import json
import os
from typing import List, Tuple, TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
try:
    from langchain.tools import StructuredTool
except ImportError:
    from langchain_core.tools import StructuredTool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph, END, START
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode

# Reuse existing business tools + context
from agent_core import (
    record_customer_interest,
    record_demo_request,
    record_phone_contact,
    record_feedback,
    load_business_context,
)

# ---- Wrap existing tools for LangGraph ----
tools = [
    StructuredTool.from_function(
        name="record_customer_interest",
        description="Record a potential customer's contact info or interest.",
        func=record_customer_interest,
    ),
    StructuredTool.from_function(
        name="record_demo_request",
        description="Log a user's request for a KolmoLabs product demo.",
        func=record_demo_request,
    ),
    StructuredTool.from_function(
        name="record_phone_contact",
        description="Store a prospect's phone number when they prefer a call.",
        func=record_phone_contact,
    ),
    StructuredTool.from_function(
        name="record_feedback",
        description="If you cannot answer from provided docs, log the user's question.",
        func=record_feedback,
    ),
]


# ---- LangGraph state ----
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1-mini")


def build_react_agent(
    persona_name: str,
    persona_description: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    top_p: float = 1.0,
    chain_of_thought: bool = False,
):
    """
    Build a manual ReAct agent graph:
      agent_node -> (conditional) tools -> agent_node -> ... -> END
    """
    ctx = load_business_context()
    intro = (
        f"You are {persona_name}, {persona_description}. "
        "Use ONLY the provided business documents as ground truth. "
        "If unsure or information is missing, call record_feedback(question). "
        "Encourage users to leave name+email for follow-up when relevant."
    )
    if chain_of_thought:
        intro += (
            " Use explicit ReAct formatting:\n"
            "Thought: ...\n"
            "Action: tool_name(args)\n"
            "Observation: ...\n"
            "Answer: ...\n"
        )

    system_text = f"{intro}\n\n--- BUSINESS DOCUMENTS (verbatim) ---\n{ctx}\n--- END DOCS ---"
    system_msg = SystemMessage(system_text)

    # Bind tools on the LLM so it can propose tool_calls
    llm = ChatOpenAI(model=model, temperature=temperature, top_p=top_p).bind_tools(tools)

    # ----- Nodes -----
    def agent_node(state: AgentState):
        # Model sees the accumulating messages (system at index 0)
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    # ----- Router -----
    def router(state: AgentState):
        last = state["messages"][-1]
        # If the model proposed tool_calls, go to tools, else END
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    # ----- Graph -----
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    # Initial state includes the system message
    init_state: AgentState = {"messages": [system_msg]}
    return graph, init_state


def run_once(graph, state: AgentState, user_text: str) -> Tuple[AgentState, str, List[str], List[str]]:
    """
    One turn through the ReAct loop using a single graph.invoke call.
    Returns (new_state, final_text, tool_summaries, error_logs).
    """
    if not isinstance(state, dict):
        state = {"messages": []}
    elif "messages" not in state or not isinstance(state["messages"], list):
        state = {"messages": list(state.get("messages", []))}
    else:
        state = {"messages": list(state["messages"])}

    baseline_messages: List[BaseMessage] = list(state["messages"])
    human_msg = HumanMessage(user_text)
    error_logs: List[str] = []

    def _invoke_with_messages(msgs: List[BaseMessage]) -> dict:
        return graph.invoke({"messages": list(msgs)}, {"configurable": {"thread_id": "react"}})

    candidate_messages = baseline_messages + [human_msg]

    try:
        result_state = _invoke_with_messages(candidate_messages)
    except InvalidUpdateError as exc:
        error_logs.append(f"InvalidUpdateError primary invoke: {exc}")
        fallback_messages = baseline_messages[:1] + [human_msg]
        try:
            result_state = _invoke_with_messages(fallback_messages)
        except InvalidUpdateError as exc2:
            error_logs.append(f"InvalidUpdateError fallback invoke: {exc2}")
            safe_reply = (
                "I'm sorry—something went wrong on my side while processing that. "
                "Could you try again with the same question?"
            )
            fallback_messages.append(AIMessage(safe_reply))
            return {"messages": fallback_messages}, safe_reply, [], error_logs

    if not isinstance(result_state, dict):
        result_messages = getattr(result_state, "messages", candidate_messages)
        result_state = {"messages": list(result_messages)}
        error_logs.append(
            f"Non-dict result_state received; coerced via messages ({type(result_state).__name__})."
        )

    result_messages: List[BaseMessage] = list(result_state.get("messages", []))

    final_text = ""
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage):
            final_text = msg.content or ""
            break

    baseline_count = len(baseline_messages)
    delta_messages = result_messages[baseline_count:]
    tool_logs: List[dict] = []

    def _capture_tool_call(call_obj):
        name = getattr(call_obj, "name", None)
        if name is None and isinstance(call_obj, dict):
            name = call_obj.get("name")
        args = getattr(call_obj, "args", None)
        if args is None and isinstance(call_obj, dict):
            args = call_obj.get("args") or call_obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if isinstance(args, dict):
            rendered_args = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        else:
            rendered_args = repr(args)
        entry = {"name": name or "unknown_tool", "args": rendered_args, "result": None}
        tool_logs.append(entry)

    for msg in delta_messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                _capture_tool_call(call)
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            for entry in reversed(tool_logs):
                if entry["result"] is None and (name is None or entry["name"] == name):
                    entry["result"] = msg.content
                    break

    tool_summaries: List[str] = []
    for entry in tool_logs:
        summary = f"{entry['name']}({entry['args']})"
        if entry["result"]:
            summary += f" -> {entry['result']}"
        tool_summaries.append(summary)

    return result_state, final_text, tool_summaries, error_logs

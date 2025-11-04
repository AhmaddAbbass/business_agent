import json
import os
from typing import List, Tuple, TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
)
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
    Tool.from_function(
        name="record_customer_interest",
        description="Record a potential customer's contact info or interest.",
        func=record_customer_interest,
    ),
    Tool.from_function(
        name="record_demo_request",
        description="Log a user's request for a KolmoLabs product demo.",
        func=record_demo_request,
    ),
    Tool.from_function(
        name="record_phone_contact",
        description="Store a prospect's phone number when they prefer a call.",
        func=record_phone_contact,
    ),
    Tool.from_function(
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

    tool_node = ToolNode(tools)

    # ----- Router -----
    def router(state: AgentState):
        last = state["messages"][-1]
        # If the model proposed tool_calls, go to tools, else END
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    # ----- Graph -----
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    # Initial state includes the system message
    init_state: AgentState = {"messages": [system_msg]}
    return graph, init_state


def run_once(graph, state: AgentState, user_text: str) -> Tuple[AgentState, str, List[str]]:
    """
    One turn through the ReAct loop. Returns (new_state, final_text, tool_summaries).
    """
    tool_logs = []
    messages = state.setdefault("messages", [])
    human_msg = HumanMessage(user_text)
    messages.append(human_msg)

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
        tool_logs.append({"name": name or "unknown_tool", "args": rendered_args, "result": None})

    existing_ids = {
        getattr(msg, "id", None) for msg in messages if getattr(msg, "id", None) is not None
    }

    # Stream events to detect tool execution and capture final AI output
    final_text = ""
    for event in graph.stream(state, {"configurable": {"thread_id": "react"}}):
        for node_name, payload in event.items():
            msgs = payload.get("messages", [])
            if msgs:
                for msg in msgs:
                    msg_id = getattr(msg, "id", None)
                    if msg_id is None or msg_id not in existing_ids:
                        messages.append(msg)
                        if msg_id is not None:
                            existing_ids.add(msg_id)
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for call in msg.tool_calls:
                            _capture_tool_call(call)
                    if isinstance(msg, ToolMessage):
                        name = getattr(msg, "name", None)
                        for log in reversed(tool_logs):
                            if log["result"] is None and (name is None or log["name"] == name):
                                log["result"] = msg.content
                                break
                if isinstance(msgs[-1], AIMessage):
                    final_text = msgs[-1].content

    if not final_text:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_text = msg.content
                break

    tool_summaries: List[str] = []
    for entry in tool_logs:
        summary = f"{entry['name']}({entry['args']})"
        if entry["result"]:
            summary += f" -> {entry['result']}"
        tool_summaries.append(summary)

    return state, (final_text or ""), tool_summaries

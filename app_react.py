from pathlib import Path
from typing import Dict

import gradio as gr

from react_agent import build_react_agent, run_once

LOGO_PATH = Path("Logo.jpg")

# Two sample personas (add more if you like)
PERSONAS: Dict[str, Dict] = {
    "Friendly Advisor": {
        "persona_name": "Friendly Advisor",
        "persona_description": "a warm, encouraging assistant for KolmoLabs prospects",
        "temperature": 0.6,
        "top_p": 1.0,
        "chain_of_thought": True,
    },
    "Strict Expert": {
        "persona_name": "Strict Expert",
        "persona_description": "a concise, formal assistant that prioritizes correctness and brevity",
        "temperature": 0.2,
        "top_p": 0.9,
        "chain_of_thought": False,
    },
}

# Keep compiled graphs cached per persona
_GRAPH_CACHE: Dict[str, tuple] = {}


def get_graph_and_state(persona_key: str):
    if persona_key in _GRAPH_CACHE:
        graph, init_messages = _GRAPH_CACHE[persona_key]
        return graph, {"messages": list(init_messages)}
    cfg = PERSONAS[persona_key]
    graph, state = build_react_agent(
        cfg["persona_name"],
        cfg["persona_description"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        chain_of_thought=cfg["chain_of_thought"],
    )
    init_messages = list(state["messages"])
    _GRAPH_CACHE[persona_key] = (graph, init_messages)
    return graph, {"messages": list(init_messages)}


def runner():
    theme = gr.themes.Soft(
        primary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="sm",
    )
    css = """
    body { background: linear-gradient(180deg, #f5f3ff 0%, #fdfcff 100%); }
    .gradio-container { max-width: 900px !important; margin: 0 auto; padding: 0.5rem 1rem !important; }
    .gr-chatbot { border-radius: 14px !important; border: 1px solid rgba(96,78,255,0.15); box-shadow: 0 4px 16px rgba(42,34,94,0.08); background: white; }
    """

    with gr.Blocks(title="KolmoLabs ReAct Assistant", theme=theme, css=css) as demo:
        persona_dd = gr.Dropdown(
            choices=list(PERSONAS.keys()),
            value="Friendly Advisor",
            label="Persona",
        )
        state_box = gr.State(None)  # holds (graph, state)

        def on_persona_change(pkey: str):
            graph, init_state = get_graph_and_state(pkey)
            # deep copy the state for a fresh session
            new_state = {"messages": list(init_state["messages"])}
            return (graph, new_state)

        persona_dd.change(
            fn=on_persona_change,
            inputs=[persona_dd],
            outputs=[state_box],
        )

        chatbot_component = gr.Chatbot(
            label="KolmoLabs ReAct Agent",
            type="messages",
            height=550,
            avatar_images=(None, str(LOGO_PATH.resolve()) if LOGO_PATH.exists() else None),
            show_copy_button=True,
        )

        def respond(message: str, history, packed, persona_key):
            if not isinstance(packed, tuple) or len(packed) != 2:
                graph, init_state = get_graph_and_state(persona_key)
                packed = (graph, init_state)

            graph, current_state = packed
            new_state, reply, tool_summaries = run_once(graph, current_state, message)
            if tool_summaries:
                formatted = "\n".join(f"- {item}" for item in tool_summaries)
                reply = f"{reply}\n\nTool activity:\n{formatted}"
            return reply, (graph, new_state)

        def respond_stream(message: str, history, packed, persona_key):
            reply, packed2 = respond(message, history, packed, persona_key)
            yield reply, packed2

        chat = gr.ChatInterface(
            fn=respond_stream,
            type="messages",
            chatbot=chatbot_component,
            textbox=gr.Textbox(
                placeholder="Ask about mission, services, pricing... I can also log feedback and capture leads.",
                lines=1,
                max_lines=3,
                autofocus=True,
            ),
            additional_inputs=[state_box, persona_dd],
            additional_outputs=[state_box],
            title="KolmoLabs ReAct Assistant",
            description="Manual ReAct loop (agent → tools → agent). Multiple personas + configs.",
            theme=theme,
            submit_btn="Send",
            stop_btn="Stop",
        )

        # initialize default persona on load
        state_box.value = on_persona_change("Friendly Advisor")

    demo.launch(server_name="127.0.0.1", share=False, show_api=False)


if __name__ == "__main__":
    runner()

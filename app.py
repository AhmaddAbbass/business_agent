from pathlib import Path
from typing import List, Tuple

import gradio as gr

from agent_core import chat_once, new_conversation

LOGO_PATH = Path("Logo.jpg")


def _agent_turn(
    agent_history: List[dict],
    user_message: str,
) -> Tuple[List[dict], str]:
    """Run a single agent turn and return updated history + reply."""
    return chat_once(agent_history, user_message)


def runner():
    """Launch the Gradio demo with a branded ChatInterface layout."""
    theme = gr.themes.Soft(
        primary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
    )
    css = """
    body {
        background: linear-gradient(180deg, #f5f3ff 0%, #ffffff 45%);
    }
    .gradio-container {
        max-width: 860px !important;
        margin: 0 auto;
    }
    .gr-chatbot {
        border-radius: 18px !important;
        border: 1px solid rgba(96, 78, 255, 0.18);
        box-shadow: 0 16px 32px rgba(42, 34, 94, 0.08);
    }
    .gr-chatbot .message {
        border-radius: 16px !important;
        border: 1px solid rgba(96, 78, 255, 0.08);
    }
    .gr-textbox textarea {
        min-height: 92px;
        font-size: 1rem;
    }
    .gradio-container footer {
        box-shadow: 0 -8px 28px rgba(32, 28, 66, 0.06);
    }
    """

    with gr.Blocks(
        title="KolmoLabs Business Assistant",
        theme=theme,
        css=css,
    ) as demo:
        agent_state = gr.State(new_conversation())

        def respond(message: str, history, agent_history):
            if not message.strip():
                return "", agent_history
            new_agent_history, reply = _agent_turn(agent_history, message)
            return reply, new_agent_history

        def respond_stream(message: str, history, agent_history):
            reply, updated_history = respond(message, history, agent_history)
            yield reply, updated_history

        chatbot_component = gr.Chatbot(
            label="KolmoLabs Assistant",
            type="messages",
            height=430,
            avatar_images=(
                None,
                str(LOGO_PATH.resolve()) if LOGO_PATH.exists() else None,
            ),
        )

        chat = gr.ChatInterface(
            fn=respond_stream,
            type="messages",
            chatbot=chatbot_component,
            textbox=gr.Textbox(
                placeholder="Ask about KolmoLabs' mission, services, pricing, or partnerships...",
                lines=2,
                autofocus=True,
                submit_btn="Send",
                stop_btn="Stop",
            ),
            additional_inputs=[agent_state],
            additional_outputs=[agent_state],
            title="KolmoLabs Business Assistant",
            description=(
                "Fast answers about Kolmogorov Neural Network services for MENA SMBs. "
                "Happy to capture leads, schedule demos, or note phone call requests."
            ),
            theme=theme,
        )

        chat.chatbot.clear(
            lambda: new_conversation(),
            outputs=[agent_state],
            queue=False,
            show_api=False,
        )

    demo.launch(server_name="127.0.0.1", server_port=5000, share=False, show_api=False)


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        print("\nServer interrupted by user. Goodbye!")

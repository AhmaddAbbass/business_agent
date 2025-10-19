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
        spacing_size="sm",
    )
    css = """
    body {
        background: linear-gradient(180deg, #f5f3ff 0%, #fdfcff 100%);
    }
    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto;
        padding: 0.5rem 1rem !important;
    }
    /* Minimize all vertical spacing */
    .contain, .gr-box, .gr-form, .gr-group {
        padding: 0 !important;
        margin: 0 !important;
        gap: 0.25rem !important;
    }
    /* Compact header */
    .gr-box h1 {
        margin: 0.5rem 0 0.25rem 0 !important;
        padding: 0 !important;
        font-size: 1.5rem !important;
        line-height: 1.2 !important;
    }
    .gr-box p {
        margin: 0 0 0.5rem 0 !important;
        padding: 0 !important;
        font-size: 0.9rem !important;
        line-height: 1.3 !important;
    }
    /* Maximize chatbot space */
    .gr-chatbot {
        border-radius: 14px !important;
        border: 1px solid rgba(96, 78, 255, 0.15);
        box-shadow: 0 4px 16px rgba(42, 34, 94, 0.08);
        background: white;
        margin: 0.5rem 0 !important;
    }
    .gr-chatbot .message {
        border-radius: 12px !important;
        padding: 0.5rem 0.75rem !important;
    }
    /* Compact input */
    .gr-textbox {
        margin: 0.5rem 0 !important;
    }
    .gr-textbox textarea {
        min-height: 60px !important;
        font-size: 0.95rem;
        border-radius: 12px !important;
        padding: 0.5rem !important;
    }
    /* Compact buttons */
    .gr-button {
        border-radius: 10px !important;
        padding: 0.4rem 1rem !important;
        margin: 0.25rem 0 !important;
    }
    /* Minimal footer */
    .gradio-container footer {
        margin-top: 0.5rem !important;
        padding: 0.5rem !important;
        box-shadow: none !important;
    }
    /* Remove gaps between form elements */
    .gr-form > * {
        margin-bottom: 0 !important;
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
            height=550,
            avatar_images=(
                None,
                str(LOGO_PATH.resolve()) if LOGO_PATH.exists() else None,
            ),
            show_copy_button=True,
        )

        chat = gr.ChatInterface(
            fn=respond_stream,
            type="messages",
            chatbot=chatbot_component,
            textbox=gr.Textbox(
                placeholder="Ask about KolmoLabs' mission, services, pricing, or partnerships...",
                lines=1,
                max_lines=3,
                autofocus=True,
                submit_btn="Send",
                stop_btn="Stop",
            ),
            additional_inputs=[agent_state],
            additional_outputs=[agent_state],
            title="KolmoLabs Business Assistant",
            description="Fast answers about Kolmogorov Neural Network services for MENA SMBs. Happy to capture leads, schedule demos, or note phone call requests.",
            theme=theme,
        )

        chat.chatbot.clear(
            lambda: new_conversation(),
            outputs=[agent_state],
            queue=False,
            show_api=False,
        )

    demo.launch(server_name="127.0.0.1", share=False, show_api=False)


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        print("\nServer interrupted by user. Goodbye!")
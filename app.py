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
    """Launch the Gradio demo with a small branded header and logo."""
    with gr.Blocks(
        title="KolmoLabs Business Assistant",
        theme=gr.themes.Soft(),
        css="""
        .branding-row {align-items: center;}
        .branding-logo {max-width: 96px; margin-right: 16px;}
        .branding-copy h1 {margin-bottom: 4px;}
        .feedback-hint {font-size: 0.85rem; color: #555;}
        """,
    ) as demo:
        agent_state = gr.State(new_conversation())
        chatbot = gr.Chatbot(label="KolmoLabs Assistant", type="tuples", height=430)
        user_box = gr.Textbox(
            placeholder="Ask about KolmoLabs' mission, services, pricing policies, etc.",
            label="Message",
            autofocus=True,
        )
        clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Row(elem_classes="branding-row"):
            if LOGO_PATH.exists():
                gr.Image(
                    value=str(LOGO_PATH.resolve()),
                    show_label=False,
                    container=False,
                    elem_classes="branding-logo",
                )
            gr.Column(
                [
                    gr.Markdown("## KolmoLabs Business Assistant"),
                    gr.Markdown(
                        "Fast answers about KolmoLabs' Kolmogorov Neural Network services for MENA SMBs."
                    ),
                ],
                elem_classes="branding-copy",
            )

        gr.Markdown(
            "Need inspiration? Try one of the quick prompts below or share your contact details if you'd like a follow-up."
        )
        gr.Examples(
            examples=[
                ["What services does KolmoLabs offer?"],
                ["We might need AI support; I'm Sara Ayoub and my email is sara@corp.com."],
                ["Do you have a Dubai office or regional partners?"],
            ],
            inputs=user_box,
        )
        gr.Markdown(
            ">(The assistant relies solely on the provided business documents. New questions are logged for follow-up.)",
            elem_classes="feedback-hint",
        )

        def respond(user_message: str, history: List[Tuple[str, str]], agent_history):
            if not user_message.strip():
                return history, agent_history

            agent_history, reply = _agent_turn(agent_history, user_message)
            updated_history = history + [(user_message, reply)]
            return updated_history, agent_history

        def clear_conversation():
            return [], new_conversation()

        user_box.submit(
            respond,
            inputs=[user_box, chatbot, agent_state],
            outputs=[chatbot, agent_state],
        ).then(
            lambda: gr.update(value=""),
            inputs=None,
            outputs=user_box,
        )
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, agent_state],
            queue=False,
        ).then(
            lambda: gr.update(value=""),
            inputs=None,
            outputs=user_box,
            queue=False,
        )

    demo.launch(server_name="127.0.0.1", server_port=5000, share=False, show_api=False)


if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        print("\nServer interrupted by user. Goodbye!")

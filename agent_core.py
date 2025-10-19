# agent_core.py
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --- load env and OpenAI client ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("AGENT_MODEL", "gpt-4.1-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- simple storage files ---
LEADS_CSV = Path("leads.csv")
FEEDBACK_LOG = Path("feedback.log")

def _append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

# ============= Tools (required by assignment) =============
def record_customer_interest(email: str, name: str, message: str = "") -> str:
    """
    Append a lead to leads.csv and print to console.
    """
    rec = f'{email},"{name}","{message.replace(chr(34), chr(39))}"'
    _append_line(LEADS_CSV, rec)
    print("[LEAD]", rec)
    return f"Recorded lead for {name} <{email}>."

def record_feedback(question: str) -> str:
    """
    Append unknown/unanswered user question to feedback.log and print to console.
    """
    _append_line(FEEDBACK_LOG, question)
    print("[FEEDBACK]", question)
    return "Logged the question for follow-up."

# Make a registry so we can dispatch tool calls
TOOL_REGISTRY = {
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
}

# OpenAI tool schemas (JSON Schema)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_customer_interest",
            "description": "Use this to record a potential customer's contact info or interest. Call this whenever a user shares an email or explicitly asks to be contacted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Customer email address."},
                    "name": {"type": "string", "description": "Customer name."},
                    "message": {"type": "string", "description": "Short note about their need or interest."}
                },
                "required": ["email", "name"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_feedback",
            "description": "If you cannot answer from the provided business documents, call this to log the user's question verbatim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question we could not answer."}
                },
                "required": ["question"]
            }
        },
    },
]

# ============= Business context loading =============
def load_business_context() -> str:
    """
    Read me/business_summary.txt and me/about_business.pdf into one context string.
    """
    parts = []
    txt_path = Path("me/business_summary.txt")
    if txt_path.exists():
        parts.append(f"[business_summary.txt]\n{txt_path.read_text(encoding='utf-8')}\n")
    pdf_path = Path("me/about_business.pdf")
    if pdf_path.exists():
        try:
            reader = PdfReader(str(pdf_path))
            pdf_text = []
            for page in reader.pages:
                t = page.extract_text() or ""
                pdf_text.append(t.strip())
            parts.append("[about_business.pdf]\n" + "\n".join(pdf_text))
        except Exception as e:
            parts.append(f"[about_business.pdf] <error reading pdf: {e}>")
    return "\n\n".join(parts).strip()

BUSINESS_CONTEXT = load_business_context()

# ============= System prompt (persona + policy) =============
SYSTEM_PROMPT = f"""
You are KolmoLabs' business assistant.

Persona & goals:
- Be helpful, concise, and on-brand for KolmoLabs, a KNN-focused AI lab serving MENA SMBs.
- Use ONLY the provided business documents (summary + PDF) as ground truth about the business.
- If information is missing or uncertain (confidence < 0.6), CALL the tool record_feedback(question).

Lead capture:
- Encourage the user to share name + email when they show interest or ask for follow-up.
- If the user provides an email (and ideally a name), CALL record_customer_interest with a short message about their interest.

Tool policy:
- Use tools only when needed; otherwise answer in plain text.
- When calling tools, keep arguments minimal and accurate.

Safety & honesty:
- Do not fabricate staff, addresses, or pricing. If unknown, log via record_feedback and offer to follow up.

Business documents (verbatim context below):
------------------------------------------------
{BUSINESS_CONTEXT}
------------------------------------------------
"""

# In-memory conversation
def new_conversation() -> List[Dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]

# Single turn with possible tool call
def chat_once(history: List[Dict], user_text: str) -> Tuple[List[Dict], str]:
    history.append({"role": "user", "content": user_text})

    # 1) Ask OpenAI with tools available
    resp = client.chat.completions.create(
        model=MODEL,
        messages=history,
        tools=OPENAI_TOOLS,
        tool_choice="auto",
        temperature=0.4,
    )

    msg = resp.choices[0].message

    # 2) If tool call, execute and return a follow-up message to the model for final say
    if msg.tool_calls:
        tool_messages = []
        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            # run the tool
            func = TOOL_REGISTRY.get(name)
            if not func:
                tool_result = f"Tool {name} not implemented."
            else:
                tool_result = func(**args)

            # record the assistant's tool call message + the tool's result message
            history.append({
                "role": "assistant",
                "tool_calls": [tc],
                "content": None
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": tool_result
            })

        history.extend(tool_messages)

        # 3) ask the model to produce the final user-facing reply
        follow = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=0.4,
        )
        final_text = follow.choices[0].message.content or ""
        history.append({"role": "assistant", "content": final_text})
        return history, final_text

    # 4) No tool call; just return text
    final_text = msg.content or ""
    history.append({"role": "assistant", "content": final_text})
    return history, final_text

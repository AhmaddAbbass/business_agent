# KolmoLabs ReAct Assistant (C4)

This project turns the KolmoLabs business assistant into a LangGraph-powered **ReAct agent** that reasons, calls structured tools, and logs every decision for analysis. The new entry point is `app_react.py`; the legacy C3 chatbot (`app.py`) is still around for reference but no longer the focus.

## TL;DR Setup
1. **Create a virtual environment**
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. **Install dependencies**  
   `pip install -r requirements.txt`
3. **Configure your API key**
   - Copy `.env.example` (or create `.env`) and set:
     ```env
     OPENAI_API_KEY="sk-your-key"
     AGENT_MODEL=gpt-4.1-mini
     ```
4. **Launch the ReAct app**  
   `python app_react.py`
   - Open the printed URL, switch personas (Friendly Advisor vs Strict Expert), and try the sample prompts.
   - Each tool call is echoed in the terminal and appended to CSV logs (`demo_requests.csv`, `phone_contacts.csv`, `leads.csv`) or `feedback.log`.
5. **Stop the server** with `Ctrl+C` when finished.

## What to Observe
- **ReAct loop**: Every assistant reply includes a “Tool activity” block when a tool is called, plus an “Errors” block if the graph had to retry.
- **Personas**: Friendly Advisor enables chain-of-thought for warmer messaging; Strict Expert is concise and deterministic.
- **Logging**: CSVs in the repo root capture every lead/demo/phone contact. All are git-ignored by default.

## Persona Experiments
Run the harness to compare personas side-by-side:
```bash
python -m experiments.react_eval
```
This writes `experiments/results.csv` with reply lengths, tool usage, and any error logs. Use it when filling out the C4 reflection.

## Legacy C3 App (Optional)
The earlier non-ReAct chatbot still lives at `app.py`. You can run it with:
```bash
python app.py
```
It serves the same content but without LangGraph’s manual reasoning loop. Useful only if you need a baseline.

## Project Structure (Highlights)
```
app_react.py           # New LangGraph UI w/ personas and tool telemetry
react_agent.py         # Build/run the manual ReAct graph
experiments/           # Persona experiment harness + CSV output
me/                    # Business docs and reflection templates
agent_core.py          # Reusable business context + tool implementations
```

## Troubleshooting
- **`InvalidUpdateError`**: The graph now self-heals and logs the error in the chat. Check the “Errors” block and `experiments/results.csv`.
- **Tool argument issues**: Inspect the terminal output; each failed call prints the raw kwargs.
- **Fresh start**: Delete the CSV/log files if you want a clean run (`demo_requests.csv`, `phone_contacts.csv`, `leads.csv`, `feedback.log`).

## Need More?
- Reflection prompts: see `me/c4_reflection_template.md`.
- Use-case summary: `me/c4_use_case.md`.
- Full write-up/PDF: `lakkis_this_folder_is_enough/c4_FULL_REPORT.pdf` (if provided for submission).

Happy experimenting! Let me know if you want scripted tests or deployment helpers.

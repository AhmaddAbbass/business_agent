# KolmoLabs Business Assistant

A lightweight Gradio-powered chatbot that answers questions about KolmoLabs, captures inbound leads, and logs follow-up requests for demos or phone calls. The project pairs a reusable agent core (`agent_core.py`) with two frontends:

* `app.py` — the production Gradio interface you can run from the terminal.
* `business_agent.ipynb` — a Jupyter notebook mirror of the same UI for quick experiments and regression checks.

## Features

- Answers questions strictly from the provided business docs (`me/business_summary.txt`, `me/about_business.pdf`).
- Logs new leads (`leads.csv`), demo requests (`demo_requests.csv`), phone contact requests (`phone_contacts.csv`), and unanswered questions (`feedback.log`).
- Branded Gradio layout with KolmoLabs logo and compact spacing.
- Optional CLI-style probes in the notebook for quick verification.

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt` (install with `pip install -r requirements.txt`).
- An OpenAI API key saved in `.env`:

```env
OPENAI_API_KEY="sk-your-key"
AGENT_MODEL=gpt-4.1-mini
```

> **Note:** The `.env` file is git-ignored. Keep your key private.

## Running the Gradio App

```bash
python app.py
```

Then open http://127.0.0.1:5000/ in a browser. Use the “Stop” button or Ctrl+C in the terminal to shut it down.

## Notebook Workflow

Open `business_agent.ipynb` in Jupyter or Colab and run all cells. The notebook:

1. Reloads `agent_core` to pick up the latest tool definitions.
2. Recreates the exact UI from `app.py` so you can launch Gradio inline.
3. Provides a “CLI-style” regression cell that prints answers and previews the tail of each log file.

## Tool & Log Mapping

| User intent                                      | Tool called                | Log file                       |
|--------------------------------------------------|----------------------------|--------------------------------|
| Share name/email for follow-up                  | `record_customer_interest` | `leads.csv`                    |
| Request a product demo                          | `record_demo_request`      | `demo_requests.csv`            |
| Prefer a phone call                             | `record_phone_contact`     | `phone_contacts.csv`           |
| Ask about unsupported / unknown info            | `record_feedback`          | `feedback.log`                 |

Each tool appends rows to its CSV/TXT file in the project root (all ignored by git).

## Quick Regression Prompts

Use either the Gradio UI or the notebook probe cell with the prompts below:

- **Knowledge only:** `What is your mission?`
- **Lead capture:** `I'm Karim from Orbit Labs. Email karim@orbit.ai—could someone reach out?`
- **Demo request:** `I need a KolmoLabs demo next Tuesday 15:00 Beirut. I'm Salma Haddad, salma.haddad@cedarcapital.ai.`
- **Phone follow-up:** `Call me tomorrow at +961-3-555555. I'm Omar Kassab and need onboarding help.`
- **Feedback escalation:** `Can you share your Dubai office address and enterprise pricing tiers?`

After each test, check the respective log file (or run the notebook preview helper) to confirm the entry was recorded.

## Project Structure

```
business_bot/
├── app.py                 # Gradio frontend
├── agent_core.py          # Agent logic, OpenAI calls, tools, prompts
├── business_agent.ipynb   # Notebook mirror of the app
├── me/                    # Business documents (summary + PDF)
├── leads.csv              # Captured email leads (ignored by git)
├── demo_requests.csv      # Captured demo requests (ignored by git)
├── phone_contacts.csv     # Captured phone callbacks (ignored by git)
├── feedback.log           # Unanswered question log (ignored by git)
├── requirements.txt       # Python dependencies
└── README.md
```

## Notes

- Log files (`leads.csv`, `demo_requests.csv`, `phone_contacts.csv`, `feedback.log`) are intentionally committed to `.gitignore`.
- If you change the logo file name or location, update `LOGO_PATH` in both `app.py` and `business_agent.ipynb`.
- Always restart your Jupyter kernel after editing `agent_core.py` so imports pick up the latest constants.

Happy hacking! Let me know if you need automated tests or deployment instructions.

# C4 Use-Case Summary — KolmoLabs ReAct Assistant

Goal: Help prospects learn about KolmoLabs (mission, services), and capture leads, demo requests, phone callbacks, or open questions via tools.

Target audience: SMB founders and ops managers in MENA seeking practical AI.

Tools:
- record_customer_interest(email, name, message)
- record_demo_request(email, name, preferred_time)
- record_phone_contact(name, phone, notes)
- record_feedback(question)

Constraints / limitations:
- Grounded strictly in provided business docs (no hallucinated pricing, staff, or addresses).
- If unknown → log via record_feedback(question) and offer follow-up.
- Encourage sharing name+email when intent suggests follow-up.

Personas:
- Friendly Advisor (warm, encouraging, CoT on)
- Strict Expert (concise, formal, CoT off)

Configurations varied:
- temperature, top_p, chain-of-thought flag

Notes:
- Manual ReAct: agent → (if tool_calls) tools → agent → ... → end.
- Evaluation logged to experiments/results.csv for comparison.

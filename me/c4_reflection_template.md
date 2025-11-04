# C4 Reflection

## 1) Which persona gave the most helpful or natural results?
Friendly Advisor felt most natural for user-facing conversations. Its higher temperature (0.6) and chain-of-thought output produced warm, detailed responses and clearly narrated its reasoning. The persona consistently triggered the right tools: `record_demo_request` captured Lina’s demo ask, `record_phone_contact` logged Omar’s call-back, and `record_customer_interest` stored Karim’s partnership lead (see `cmdlog.txt` entries and `experiments/results.csv`, rows 2–4). Strict Expert was crisp and correct, but its terse tone felt less inviting for lead capture.

## 2) Which prompt/config combination performed best for your use case?
The demo-request scenario (“I’m Lina; my email is … can I get a demo next week?”) under Friendly Advisor’s config delivered the strongest results: the agent acknowledged intent, recorded contact details via `record_demo_request`, and offered follow-up guidance without errors. Strict Expert also logged the request, but Friendly Advisor provided more customer-friendly messaging while preserving accuracy.

## 3) How well did your agent reason and use tools?
Reasoning stayed grounded in the business docs. On knowledge questions (“What is your mission?”), both personas responded directly from context. For unavailable info (Dubai/Mumbai office, enterprise pricing), the agent declined to hallucinate and logged feedback (`feedback.log`). With StructuredTool wrappers, every lead/demo/phone prompt executed the proper CSV-logging tool (evidenced in `experiments/results.csv` with `used_tools=1`), and no error blocks surfaced in the chat transcript.

## 4) Biggest challenges in implementation?
- LangGraph repeatedly raised `InvalidUpdateError` when the start-state wasn’t a dict; wiring `START → agent` and adding retry logic in `run_once` resolved it.
- LangChain’s Tool API changes caused positional-argument failures until we switched to `StructuredTool.from_function`.
- Ensuring state integrity between Gradio sessions required extra guards so `(graph, state_dict)` tuples never degraded into sentinels.
- Adding logging (tool activity + error traces) was essential for debugging the manual ReAct loop without an executor.

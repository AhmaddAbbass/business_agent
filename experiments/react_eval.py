import csv
from datetime import datetime
from pathlib import Path

from react_agent import build_react_agent, run_once

RESULTS_CSV = Path("experiments/results.csv")
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

PERSONA_MATRIX = [
    # (name, description, temperature, top_p, chain_of_thought)
    ("Friendly Advisor", "a warm, encouraging assistant", 0.6, 1.0, True),
    ("Strict Expert", "a concise, formal assistant", 0.2, 0.9, False),
]

PROMPTS = [
    "What is your mission?",
    "I'm Lina; my email is lina@example.com — can I get a demo next week?",
    "Call me at +961-3-555555, I'm Omar; need onboarding help.",
    "What are your enterprise pricing tiers and Dubai office address?",
]


def main():
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "persona",
                "temperature",
                "top_p",
                "cot",
                "prompt",
                "reply_len",
                "used_tools",
            ]
        )

        for pname, pdesc, temp, top_p, cot in PERSONA_MATRIX:
            graph, state = build_react_agent(
                persona_name=pname,
                persona_description=pdesc,
                temperature=temp,
                top_p=top_p,
                chain_of_thought=cot,
            )
            # copy initial so each prompt isolates one turn
            base_state = {"messages": list(state["messages"])}

            for prompt in PROMPTS:
                s = {"messages": list(base_state["messages"])}
                s, reply, tool_logs = run_once(graph, s, prompt)
                w.writerow(
                    [
                        datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        pname,
                        temp,
                        top_p,
                        cot,
                        prompt,
                        len(reply or ""),
                        int(bool(tool_logs)),
                    ]
                )

    print(f"Wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()

"""Hosted model to hosted model upgrades from the saved workflow v4 part A answers. No new call.

Post hoc: written after part A was scored. It has no confidence signal, so it reports what an upgrade could
give (oracle), what it gives blind (random) and what replacing the model outright gives.
"""

from __future__ import annotations

import json

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.protocol import load_protocol

RATES = [0.1, 0.25, 0.5]


def main() -> None:
    rules = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    turns = [turn for session in sessions for turn in session]
    answers, cost = {}, {}
    for name in sorted(rules["models"], key=lambda n: rules["models"][n]["params_b"]):
        rows = v4._read(v4.OUT / f"a__{name}.jsonl")
        answers[name] = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
        cost[name] = 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions)
    right = {name: np.asarray([v4._turn_right(turn, table[turn["id"]]) for turn in turns]) for name, table in answers.items()}
    success = lambda answer_of: v4._session_success(sessions, v4.decide(sessions, answer_of))
    alone = {name: success(lambda turn, t=table: t[turn["id"]]) for name, table in answers.items()}
    rng = np.random.default_rng(v4.SEED)
    out = {"alone": alone, "usd_per_1000_sessions": cost, "turn_right": {name: float(flags.mean()) for name, flags in right.items()}, "pairs": {}}
    names = list(answers)
    print("turn decision right / sessions right / $ per 1000 sessions")
    for name in names:
        print(f"  {name:24s} {100 * right[name].mean():5.1f}  {100 * alone[name]:5.1f}  ${cost[name]:.2f}")
    print("\nfrom -> to: to fixes / to breaks (% of turns) | sessions: from, to, oracle@10/25/50%, random@25%")
    for low in names:
        for high in names:
            if low == high:
                continue
            fixes = (~right[low]) & right[high]
            breaks = right[low] & (~right[high])
            entry = {"fixes": float(fixes.mean()), "breaks": float(breaks.mean()), "agree_on_intent_and_scope": float(np.mean([answers[low][t["id"]]["scope"] == answers[high][t["id"]]["scope"] and answers[low][t["id"]]["intent"] == answers[high][t["id"]]["intent"] for t in turns])), "oracle": {}, "random": {}}
            order = np.argsort(-fixes.astype(np.float64), kind="stable")
            for rate in RATES:
                k = int(round(rate * len(turns)))
                chosen = {turns[i]["id"] for i in order[:k]}
                entry["oracle"][str(rate)] = success(lambda turn: answers[high][turn["id"]] if turn["id"] in chosen else answers[low][turn["id"]])
            draws = []
            for _ in range(100):
                drawn = {turns[i]["id"] for i in rng.choice(len(turns), size=int(round(0.25 * len(turns))), replace=False)}
                draws.append(success(lambda turn: answers[high][turn["id"]] if turn["id"] in drawn else answers[low][turn["id"]]))
            entry["random"]["0.25"] = float(np.mean(draws))
            # Disagreement as a free signal once both are called: when they agree the answer is the same either way.
            both = np.asarray([answers[low][t["id"]]["scope"] == answers[high][t["id"]]["scope"] and answers[low][t["id"]]["intent"] == answers[high][t["id"]]["intent"] for t in turns])
            entry["right_when_agree"] = float(right[low][both].mean())
            entry["low_right_when_disagree"] = float(right[low][~both].mean())
            entry["high_right_when_disagree"] = float(right[high][~both].mean())
            out["pairs"][f"{low}->{high}"] = entry
            if rules["models"][low]["params_b"] < rules["models"][high]["params_b"]:
                print(
                    f"  {low:22s} -> {high:22s} {100 * entry['fixes']:4.1f} / {100 * entry['breaks']:4.1f} | {100 * alone[low]:4.1f} {100 * alone[high]:4.1f} "
                    f"{' '.join(f'{100 * v:4.1f}' for v in entry['oracle'].values())} {100 * entry['random']['0.25']:4.1f} | agree {100 * entry['agree_on_intent_and_scope']:.0f}% right when agree {100 * entry['right_when_agree']:.0f}%"
                )
    (v4.OUT / "pairs.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

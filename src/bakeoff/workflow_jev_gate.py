"""Jev as the scope gate in front of the specialist. Rules: workflow_jev_gate_protocol.json. Jev API only."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.protocol import ROOT
from bakeoff.workflow_retrieval import _domain

RULES_PATH = ROOT / "workflow_jev_gate_protocol.json"
OUT = ROOT / "artifacts" / "workflow-jev-gate"


def collect(rules: dict, name: str, domain: dict) -> dict:
    import httpx

    from bakeoff import jev

    path = OUT / f"{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row
    labels = domain["labels"]
    turns = [turn for session in domain["sessions"] for turn in session]
    questions = {
        "scope_described": {"type": "noul", "instructions": rules["scope_questions"][name]},
        "scope_listed": {"type": "noul", "instructions": rules["scope_questions"]["listed_form"].format(labels=", ".join(labels))},
        "intent": {"type": "choice", "instructions": "Which label best describes the current message?", "criteria": {label: None for label in labels}},
    }
    todo = [turn for turn in turns if turn["id"] not in done]
    key = jev._api_key(json.loads(jev.JEV_PROTOCOL_PATH.read_text()))
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=90.0) as client, path.open("a", encoding="utf-8") as handle:

        def work(turn):
            payload = v1._jev_post(client, {"model": rules["jev_request"]["model"], "state": turn["text"], "questions": questions})
            answers = payload["answers"]
            return {
                "id": turn["id"],
                "scope_described": float(answers["scope_described"]["noul"]),
                "scope_listed": float(answers["scope_listed"]["noul"]),
                "intent_probs": [float(answers["intent"]["probabilities"][label]) for label in labels],
                "model": payload.get("model"),
                "input_tokens": (payload.get("usage") or {}).get("input_tokens", 0),
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, row in enumerate(pool.map(work, todo), start=1):
                handle.write(json.dumps(row) + "\n")
                done[row["id"]] = row
                if count % 500 == 0:
                    handle.flush()
                    print(f"  {name} {count}/{len(todo)}", flush=True)
    return done


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else ["banking77", "massive", "clinc150"]
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"], "domains": {}}
    for name in only:
        domain = _domain(name)
        labels, sessions = domain["labels"], domain["sessions"]
        turns = [turn for session in sessions for turn in session]
        jev_rows = collect(rules, name, domain)
        specialists = {"encoder": domain["specialist"]}
        if name == "clinc150":
            from bakeoff import workflow_clinc as c

            clinc_rules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
            data = c._data(clinc_rules)
            specialists["encoder_oos"] = c.specialists(clinc_rules, data, sessions)["encoder_oos"]["answers"]

        def evaluate(answer_of) -> dict:
            decisions = v4.decide(sessions, answer_of)
            result = v1.evaluate(sessions, decisions, None)
            flat = [row for rows in decisions for row in rows]
            kept = float(np.mean([row["scope"] for turn, row in zip(turns, flat) if turn["in_scope"]]))
            caught = float(np.mean([not row["scope"] for turn, row in zip(turns, flat) if not turn["in_scope"]]))
            return {"session_success": result["session_success"], "session_success_ci": result["session_success_ci"], "turn_success": result["turn_success"], "node_accuracy": result["node_accuracy"], "in_scope_kept": kept, "out_of_scope_caught": caught}

        block = {"jev_model": sorted({str(row["model"]) for row in jev_rows.values()}), "jev_input_tokens_per_session": float(np.sum([row["input_tokens"] for row in jev_rows.values()]) / len(sessions)), "policies": {}}
        policies = block["policies"]
        policies["jev_alone"] = evaluate(lambda turn: {"scope": jev_rows[turn["id"]]["scope_described"] >= 0.5, "intent": int(np.argmax(jev_rows[turn["id"]]["intent_probs"]))})
        for kind, base in specialists.items():
            policies[f"{kind}_alone"] = evaluate(lambda turn: base[turn["id"]])
            policies[f"jev_gate_described+{kind}"] = evaluate(lambda turn: {"scope": jev_rows[turn["id"]]["scope_described"] >= 0.5, "intent": base[turn["id"]]["intent"]})
            policies[f"jev_gate_listed+{kind}"] = evaluate(lambda turn: {"scope": jev_rows[turn["id"]]["scope_listed"] >= 0.5, "intent": base[turn["id"]]["intent"]})
            policies[f"jev_gate_and_specialist+{kind}"] = evaluate(lambda turn: {"scope": jev_rows[turn["id"]]["scope_described"] >= 0.5 and base[turn["id"]]["scope"], "intent": base[turn["id"]]["intent"]})
        base = specialists["encoder"]
        order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
        quarter = {turn["id"] for turn in order[: len(turns) // 4]}
        for model in rules_v4["models"]:
            rows = v4._read(domain["zero_shot"] / f"a__{model}.jsonl")
            if len(rows) < len(turns):
                continue
            answers = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
            policies[f"llm_gate:{model}+encoder"] = evaluate(lambda turn: {"scope": answers[turn["id"]]["scope"], "intent": base[turn["id"]]["intent"]})
            policies[f"llm_alone:{model}"] = evaluate(lambda turn: answers[turn["id"]])
            policies[f"encoder_escalate_25pct:{model}"] = evaluate(lambda turn: answers[turn["id"]] if turn["id"] in quarter else base[turn["id"]])
        if name == "banking77":
            old = {}
            for line in (v1.OUT / "jev_typed_state.jsonl").open(encoding="utf-8"):
                row = json.loads(line)
                old[row["id"]] = row
            block["replay_scope_agreement_with_2026-10-09"] = float(np.mean([(old[t["id"]]["scope"] >= 0.5) == (jev_rows[t["id"]]["scope_described"] >= 0.5) for t in turns]))
            block["replay_intent_agreement_with_2026-10-09"] = float(np.mean([int(np.argmax(old[t["id"]]["intent_probs"])) == int(np.argmax(jev_rows[t["id"]]["intent_probs"])) for t in turns]))
        summary["domains"][name] = block
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\n{name}: Jev {block['jev_model']} {block['jev_input_tokens_per_session']:.0f} input tokens per session" + (f"; replay agreement scope {block['replay_scope_agreement_with_2026-10-09']:.3f} intent {block['replay_intent_agreement_with_2026-10-09']:.3f}" if name == "banking77" else ""), flush=True)
        for key, row in policies.items():
            print(f"  {key:48s} session {100 * row['session_success']:5.1f} [{100 * row['session_success_ci'][0]:.1f}, {100 * row['session_success_ci'][1]:.1f}] scope {100 * row['node_accuracy']['scope']:5.1f} kept {100 * row['in_scope_kept']:5.1f} caught {100 * row['out_of_scope_caught']:5.1f} intent {100 * row['node_accuracy']['intent']:5.1f}", flush=True)
    print("JEV_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()

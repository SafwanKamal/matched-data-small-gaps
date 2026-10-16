"""The workflow v2 local orchestrator on MASSIVE and CLINC150. Rules: workflow_v2_domains_protocol.json. GPU, no API."""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v2 as v2
from bakeoff import workflow_v4 as v4
from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "workflow_v2_domains_protocol.json"
OUT = ROOT / "artifacts" / "workflow-v2-domains"


def _domain(name: str) -> dict:
    if name == "massive":
        from bakeoff import workflow_massive as m

        rules = json.loads(m.RULES_PATH.read_text(encoding="utf-8"))
        loaded, sessions, oos_dev = m._sources(rules)
        spec = m.specialists(rules, loaded, sessions, oos_dev)
        return {"labels": loaded["labels"], "sessions": sessions, "encoder": spec["encoder"], "second": spec["decoder"]["answers"], "answers_dir": m.OUT}
    from bakeoff import workflow_clinc as c

    rules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
    data = c._data(rules)
    sessions = c._sessions(rules, data)
    spec = c.specialists(rules, data, sessions)
    return {"labels": data["labels"], "sessions": sessions, "encoder": spec["encoder"], "second": spec["encoder_oos"]["answers"], "answers_dir": c.OUT}


def main() -> None:
    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v2 = json.loads(v2.RULES_PATH.read_text(encoding="utf-8"))
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else ["massive", "clinc150"]
    path = OUT / "summary.json"
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": spec["protocol_id"], "domains": {}}
    for name in only:
        domain = _domain(name)
        labels, sessions = domain["labels"], domain["sessions"]
        turns = [turn for session in sessions for turn in session]
        encoder = domain["encoder"]
        rules = {**rules_v2, "prompt": spec["prompts"][name], "variants": {key: rules_v2["variants"][key] for key in spec["variants"]}}
        v2.OUT = OUT / name
        v2.OUT.mkdir(parents=True, exist_ok=True)
        actions = v2._actions(rules, sessions, labels, encoder["probs"])
        jev = {}
        for line in (ROOT / "artifacts" / "workflow-jev-gate" / f"{name}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            jev[row["id"]] = {"scope": row["scope_listed"] >= 0.5, "intent": int(np.argmax(row["intent_probs"]))}

        def answer(action: int, turn) -> dict:
            if action == 1:
                return {"scope": True, "intent": encoder["answers"][turn["id"]]["intent"]}
            if action == 2:
                return domain["second"][turn["id"]]
            if action == 3:
                return jev[turn["id"]]
            return {"scope": False, "intent": -1}

        def evaluate(answer_of) -> dict:
            result = v1.evaluate(sessions, v4.decide(sessions, answer_of), None)
            return {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}

        block = {"policies": {}, "comparators": {}}
        print(f"\n{name}", flush=True)
        for variant in spec["variants"]:
            chosen = {key: value["action"] for key, value in actions[variant].items()}
            row = evaluate(lambda turn: answer(chosen[turn["id"]], turn))
            row["action_shares"] = {str(a): float(np.mean([chosen[t["id"]] == a for t in turns])) for a in (1, 2, 3, 4)}
            marked = [t for t in turns if chosen[t["id"]] == 4]
            row["action4_precision"] = float(np.mean([not t["in_scope"] for t in marked])) if marked else None
            row["action4_recall"] = float(np.mean([chosen[t["id"]] == 4 for t in turns if not t["in_scope"]]))
            gate = evaluate(lambda turn: answer(4 if chosen[turn["id"]] == 4 else 1, turn))
            block["policies"][variant] = row
            block["policies"][f"{variant}__as_gate_only"] = gate
            print(
                f"  {variant:30s} session {100 * row['session_success']:5.1f} [{100 * row['session_success_ci'][0]:.1f}, {100 * row['session_success_ci'][1]:.1f}] scope {100 * row['node_accuracy']['scope']:.1f} intent {100 * row['node_accuracy']['intent']:.1f} "
                f"actions {' '.join(f'{k}:{v:.2f}' for k, v in row['action_shares'].items())} out precision {row['action4_precision'] or 0:.2f} recall {row['action4_recall']:.2f} | as gate only {100 * gate['session_success']:.1f}",
                flush=True,
            )
        block["comparators"]["encoder_alone_softmax"] = evaluate(lambda turn: encoder["answers"][turn["id"]])
        detectors = json.loads((ROOT / "artifacts" / "oos-detectors" / "summary.json").read_text())["domains"].get(name, {}).get("detectors", {})
        if detectors:
            best = max(detectors, key=lambda key: detectors[key]["dev_balanced_accuracy"])
            block["comparators"][f"encoder_alone_{best}"] = {"session_success": detectors[best]["session_success"]}
        rows = v4._read(domain["answers_dir"] / "a__gemma-4-31b.jsonl")
        gemma = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
        block["comparators"]["gemma_gate_encoder_intent"] = evaluate(lambda turn: {"scope": gemma[turn["id"]]["scope"], "intent": encoder["answers"][turn["id"]]["intent"]})
        retrieval = json.loads((ROOT / "artifacts" / "workflow-retrieval" / "summary.json").read_text())["domains"][name]["runs"]["gemma-4-31b__full_train"]["alone"]
        block["comparators"]["gemma_full_retrieval_alone"] = {"session_success": retrieval["session_success"], "session_success_ci": retrieval["session_success_ci"]}
        for key, row in block["comparators"].items():
            print(f"  ref {key:34s} session {100 * row['session_success']:5.1f}", flush=True)
        summary["domains"][name] = block
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print("V2_DOMAINS_DONE", flush=True)


if __name__ == "__main__":
    main()

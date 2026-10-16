"""Workflow v5: few-shot hosted models and small-to-large hosted pairs. Rules: workflow_v5_protocol.json. API only."""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff import workflow_v3 as v3
from bakeoff.data import load_items
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.train_ettin import _train_rows

RULES_PATH = ROOT / "workflow_v5_protocol.json"
OUT = ROOT / "artifacts" / "workflow-v5"
SEED = 20261012


def _few_shot_system(rules_v4, labels, k: int) -> str:
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    rows = _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"]))
    rng = np.random.default_rng(SEED)
    codes = api.codes_for(len(labels))
    by_label: dict[int, list[str]] = {}
    for text, gold in rows:
        by_label.setdefault(gold, []).append(text)
    examples = []
    for label in sorted(by_label):
        group = by_label[label]
        for index in rng.permutation(len(group))[:k]:
            examples.append((group[index], codes[label]))
    _, _, _, _, oos_dev = v1._sources()
    examples += [(text, "00") for text in oos_dev[10:14]]
    order = rng.permutation(len(examples))
    lines = "\n".join(f"{examples[i][0]} => {examples[i][1]}" for i in order)
    return v4._system_prompt(rules_v4, labels, "current_only", False) + "\n\nLabelled examples (message => code):\n" + lines


def _brief(result: dict) -> dict:
    return {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}


def part_d(rules, rules_v4, labels, sessions, client, budget) -> dict:
    subset = sessions[:150]
    turns = [turn for session in subset for turn in session]
    all_turns = [turn for session in sessions for turn in session]
    specialists = v4._specialists(all_turns)
    evaluate = lambda answer_of: _brief(v1.evaluate(subset, v4.decide(subset, answer_of), None))
    out = {"n_sessions": len(subset), "specialists": {kind: evaluate(lambda turn, b=block: b["answers"][turn["id"]]) for kind, block in specialists.items()}, "models": {}}
    print("  specialists on these sessions: " + "  ".join(f"{kind} {100 * row['session_success']:.1f}" for kind, row in out["specialists"].items()), flush=True)
    for name, shots in rules["part_d_few_shot"]["runs"].items():
        model = rules_v4["models"][name]
        tables = {0: {key: {**v4.parse(row["text"], len(labels)), "cost": row["cost"]} for key, row in v4._read(v4.OUT / f"a__{name}.jsonl").items()}}
        for k in shots:
            system = _few_shot_system(rules_v4, labels, k)
            jobs = [(turn["id"], system, rules_v4["request"]["user"]["current_only"].format(text=turn["text"])) for turn in turns]
            print(f"  part D {name} {k}-shot", flush=True)
            rows = v4._collect(client, model, jobs, rules_v4["request"]["max_tokens"]["answer"], OUT / f"d__{name}__{k}shot.jsonl", budget)
            tables[k] = {key: {**v4.parse(row["text"], len(labels)), "cost": row["cost"], "prompt_tokens": row["prompt_tokens"], "cached_tokens": row.get("cached_tokens")} for key, row in rows.items()}
        out["models"][name] = {}
        for k, table in tables.items():
            entry = {"alone": evaluate(lambda turn: table[turn["id"]]), "usd_per_1000_sessions": 1000.0 * sum(table[t["id"]]["cost"] for t in turns) / len(subset), "parse_failed": sum(table[t["id"]]["parse_failed"] for t in turns), "hybrid": {}}
            if k:
                entry["prompt_tokens_mean"] = float(np.mean([table[t["id"]]["prompt_tokens"] or 0 for t in turns]))
                entry["cached_share"] = float(sum(table[t["id"]]["cached_tokens"] or 0 for t in turns) / max(1, sum(table[t["id"]]["prompt_tokens"] or 0 for t in turns)))
            for kind, block in specialists.items():
                base = block["answers"]
                order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
                entry["hybrid"][kind] = {}
                for rate in (0.1, 0.25, 0.4, 0.6):
                    chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                    entry["hybrid"][kind][str(rate)] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else base[turn["id"]])
            out["models"][name][f"{k}-shot"] = entry
            best = {kind: max(cells.values(), key=lambda cell: cell["session_success"])["session_success"] for kind, cells in entry["hybrid"].items()}
            print(
                f"{name:20s} {k}-shot alone session {100 * entry['alone']['session_success']:.1f} intent {100 * entry['alone']['node_accuracy']['intent']:.1f} scope {100 * entry['alone']['node_accuracy']['scope']:.1f} "
                f"${entry['usd_per_1000_sessions']:.2f}/1000 | best hybrid encoder {100 * best['encoder']:.1f} decoder {100 * best['decoder']:.1f} | parse failures {entry['parse_failed']}",
                flush=True,
            )
    return out


def part_e(rules, rules_v4, labels, sessions, client, budget) -> dict:
    turns = [turn for session in sessions for turn in session]
    system = v4._system_prompt(rules_v4, labels, "current_only", False)
    targets = {}
    for name in rules_v4["models"]:
        rows = v4._read(v4.OUT / f"a__{name}.jsonl")
        targets[name] = {key: {**v4.parse(row["text"], len(labels)), "cost": row["cost"]} for key, row in rows.items()}
    success = lambda answer_of: v4._session_success(sessions, v4.decide(sessions, answer_of))
    rng = np.random.default_rng(SEED)
    rates = [0.05, 0.1, 0.25, 0.4, 0.6, 0.8]
    out = {}
    for name, provider in rules["part_e_model_pairs"]["small_models"].items():
        model = {**rules_v4["models"][name], "provider": provider}
        jobs = [(turn["id"], system, rules_v4["request"]["user"]["current_only"].format(text=turn["text"])) for turn in turns]
        print(f"  part E {name} with log-probabilities on {provider}", flush=True)
        rows = v4._collect(client, model, jobs, rules_v4["request"]["max_tokens"]["answer"], OUT / f"e__{name}.jsonl", budget, logprobs=True)
        small = {}
        for key, row in rows.items():
            steps = row.get("token_logprobs") or []
            small[key] = {**v4.parse(row["text"], len(labels)), "cost": row["cost"], "confidence": math.exp(sum(step[1] for step in steps)) if steps else float("nan")}
        missing = sum(1 for value in small.values() if value["confidence"] != value["confidence"])
        confidence = np.asarray([small[t["id"]]["confidence"] if small[t["id"]]["confidence"] == small[t["id"]]["confidence"] else 0.0 for t in turns])
        wrong = np.asarray([not v4._turn_right(t, small[t["id"]]) for t in turns])
        order = np.argsort(confidence, kind="stable")
        small_cost = sum(value["cost"] for value in small.values())
        entry = {
            "alone": _brief(v1.evaluate(sessions, v4.decide(sessions, lambda turn: small[turn["id"]]), None)),
            "usd_per_1000_sessions": 1000.0 * small_cost / len(sessions),
            "confidence_auroc_for_wrong": v3._auroc(-confidence, wrong),
            "no_logprobs": missing,
            "agreement_with_v4_part_a_answer": float(np.mean([small[t["id"]]["intent"] == targets[name][t["id"]]["intent"] and small[t["id"]]["scope"] == targets[name][t["id"]]["scope"] for t in turns])),
            "targets": {},
        }
        print(f"{name} alone session {100 * entry['alone']['session_success']:.1f} confidence AUROC {entry['confidence_auroc_for_wrong']:.3f} replies without log-probabilities {missing}", flush=True)
        for target, table in targets.items():
            if target == name:
                continue
            helps = np.asarray([wrong[i] and v4._turn_right(t, table[t["id"]]) for i, t in enumerate(turns)], dtype=np.float64)
            cells = {}
            for rate in rates:
                k = int(round(rate * len(turns)))
                chosen = {turns[i]["id"] for i in order[:k]}
                oracle = {turns[i]["id"] for i in np.argsort(-helps, kind="stable")[:k]}
                draws = []
                for _ in range(100):
                    drawn = {turns[i]["id"] for i in rng.choice(len(turns), size=k, replace=False)}
                    draws.append(success(lambda turn: table[turn["id"]] if turn["id"] in drawn else small[turn["id"]]))
                result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: table[turn["id"]] if turn["id"] in chosen else small[turn["id"]]), None)
                cells[str(rate)] = {
                    **_brief(result),
                    "usd_per_1000_sessions": 1000.0 * (small_cost + sum(table[i]["cost"] for i in chosen)) / len(sessions),
                    "random": float(np.mean(draws)),
                    "oracle": success(lambda turn: table[turn["id"]] if turn["id"] in oracle else small[turn["id"]]),
                }
            target_alone = success(lambda turn: table[turn["id"]])
            entry["targets"][target] = {"target_alone": target_alone, "target_usd_per_1000_sessions": 1000.0 * sum(v["cost"] for v in table.values()) / len(sessions), "rates": cells}
            print(f"  {name} -> {target:22s} target alone {100 * target_alone:4.1f} | by confidence " + " ".join(f"{rate}:{100 * cell['session_success']:.1f}" for rate, cell in cells.items()) + f" | random@0.25 {100 * cells['0.25']['random']:.1f} oracle@0.25 {100 * cells['0.25']['oracle']:.1f}", flush=True)
        out[name] = entry
    return out


def main() -> None:
    import httpx

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    budget = api.Budget(api_rules, run_cap=rules["run_cap_usd"])
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"]}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=90.0) as client:
        if stage in ("all", "e"):
            summary["part_e"] = part_e(rules, rules_v4, labels, sessions, client, budget)
            path.write_text(json.dumps(summary, indent=2) + "\n")
        if stage in ("all", "d"):
            summary["part_d"] = part_d(rules, rules_v4, labels, sessions, client, budget)
            path.write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"WORKFLOW_V5_DONE {stage} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    main()

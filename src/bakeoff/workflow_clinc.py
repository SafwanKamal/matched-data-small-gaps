"""The workflow ladder on CLINC150, with native out-of-scope queries. Rules: workflow_clinc_protocol.json."""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.data import Item
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "workflow_clinc_protocol.json"
OUT = ROOT / "artifacts" / "workflow-clinc"
SOURCE = ("clinc/clinc_oos", "plus", "155b9c710419136e17307b80d0a13e68cd46b4ec")


def _data(rules) -> dict:
    from datasets import load_dataset

    raw = load_dataset(SOURCE[0], SOURCE[1], revision=SOURCE[2])
    names = raw["train"].features["intent"].names
    oos = names.index("oos")
    labels = [name.replace("_", " ") for name in names if name != "oos"]
    remap = {index: position for position, index in enumerate(i for i in range(len(names)) if i != oos)}
    parts = {}
    for split in ("train", "validation", "test"):
        inside = [Item(item_id=f"clinc-{split}-{row}", row=row, text=example["text"], gold_index=remap[example["intent"]], gold_label="") for row, example in enumerate(raw[split]) if example["intent"] != oos]
        outside = [example["text"] for example in raw[split] if example["intent"] == oos]
        parts[split] = (inside, outside)
    rng = np.random.default_rng(rules["seed"])
    dev_in = [parts["validation"][0][i] for i in sorted(rng.permutation(len(parts["validation"][0]))[:400])]
    return {"labels": labels, "train_in": parts["train"][0], "train_out": parts["train"][1], "dev_in": dev_in, "dev_out": parts["validation"][1], "test_in": parts["test"][0], "test_out": parts["test"][1], "val_in_all": parts["validation"][0]}


def _sessions(rules, data):
    path = OUT / "sessions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sessions = v1.build_sessions({"seed": rules["seed"], "sessions": {"n": 400, "turns": 5}}, data["test_in"], data["test_out"])
    path.write_text(json.dumps(sessions, indent=1) + "\n", encoding="utf-8")
    return sessions


def train() -> None:
    """Fine-tune both specialists on the GPU. Skips a model that is already saved."""
    import torch

    from bakeoff import train_ettin as enc

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    data = _data(rules)
    spec = json.loads((ROOT / "program2.json").read_text())
    spec.update({"learning_rate": 1e-4, "seed": rules["seed"]})
    n = len(data["labels"])
    out_item = lambda text, i: Item(item_id=f"oos-{i}", row=i, text=text, gold_index=n, gold_label="oos")
    jobs = {
        "encoder": (data["labels"], [(item.text, item.gold_index) for item in data["train_in"]], data["val_in_all"], data["test_in"]),
        "encoder_oos": (
            data["labels"] + ["out of scope"],
            [(item.text, item.gold_index) for item in data["train_in"]] + [(text, n) for text in data["train_out"]],
            data["val_in_all"] + [out_item(text, i) for i, text in enumerate(data["dev_out"])],
            data["test_in"] + [out_item(text, i) for i, text in enumerate(data["test_out"])],
        ),
    }
    for name, (labels, rows, dev, test) in jobs.items():
        if (OUT / name / "metrics.json").exists():
            print(f"skip {name}", flush=True)
            continue
        torch.manual_seed(rules["seed"])
        stub = {"datasets": {"clinc150": {"labels": labels}}}
        materialized = {"splits": {"clinc150": {"dev": {"rows": []}}}, "items": {"clinc150": {"dev": dev, "test": test}}}
        result = enc._train_one(name, "clinc150", "log_loss", spec, stub, materialized, torch.device("cuda"), rows=rows, out_root=OUT)
        print(f"trained {name} test acc {result['scaled']['accuracy']:.4f}", flush=True)
    specialists(rules, data, _sessions(rules, data), device="cuda")
    print("CLINC_TRAIN_DONE", flush=True)


def specialists(rules, data, sessions, device: str = "cpu") -> dict:
    from bakeoff.workflow_massive import _score_texts

    labels = data["labels"]
    n = len(labels)
    turns = [turn for session in sessions for turn in session]
    sets = {"session_turns": [t["text"] for t in turns], "dev_in_scope": [item.text for item in data["dev_in"]], "dev_out_of_scope": data["dev_out"]}
    out = {}
    for name in ("encoder", "encoder_oos"):
        folder = OUT / name
        if not (folder / "metrics.json").exists():
            continue
        cache = OUT / f"{name}__scores.npz"
        if not cache.exists():
            head_labels = labels if name == "encoder" else labels + ["out of scope"]
            np.savez_compressed(cache, **{part: _score_texts("encoder", folder, texts, head_labels, device) for part, texts in sets.items()})
            print(f"scored {name}", flush=True)
        scores = dict(np.load(cache))
        gold_in = np.asarray([item.gold_index for item in data["dev_in"]], dtype=np.int64)
        if name == "encoder":
            temperature = fit_temperature(scores["dev_in_scope"], gold_in)["temperature"]
            dev_in = apply_temperature(scores["dev_in_scope"], temperature).max(axis=1)
            rule = v1._scope_threshold(dev_in, apply_temperature(scores["dev_out_of_scope"], temperature).max(axis=1))
            probs = apply_temperature(scores["session_turns"], temperature)
            answers = {turn["id"]: {"scope": bool(row.max() >= rule["tau"]), "intent": int(row.argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)}
        else:
            both = np.concatenate([scores["dev_in_scope"], scores["dev_out_of_scope"]], axis=0)
            gold = np.concatenate([gold_in, np.full(len(scores["dev_out_of_scope"]), n, dtype=np.int64)])
            temperature = fit_temperature(both, gold)["temperature"]
            dev_in = apply_temperature(scores["dev_in_scope"], temperature).max(axis=1)
            rule = {"tau": None, "dev_balanced_accuracy": float(0.5 * ((scores["dev_in_scope"].argmax(axis=1) != n).mean() + (scores["dev_out_of_scope"].argmax(axis=1) == n).mean()))}
            probs = apply_temperature(scores["session_turns"], temperature)
            answers = {turn["id"]: {"scope": bool(row.argmax() != n), "intent": int(row[:n].argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)}
        out[name] = {"answers": answers, "probs": {turn["id"]: row[:n] for turn, row in zip(turns, probs)}, "escalation_tau": float(np.quantile(dev_in, 0.25)), "rule": {**rule, "temperature": temperature}}
    return out


def _few_shot_system(rules, data, k: int) -> str:
    labels = data["labels"]
    rng = np.random.default_rng(rules["seed"])
    codes = api.codes_for(len(labels))
    by_label: dict[int, list[str]] = {}
    for item in data["train_in"]:
        by_label.setdefault(item.gold_index, []).append(item.text)
    examples = []
    for label in sorted(by_label):
        group = by_label[label]
        examples += [(group[i], codes[label]) for i in rng.permutation(len(group))[:k]]
    examples += [(text, "000") for text in data["train_out"][:4]]
    order = rng.permutation(len(examples))
    return v4._system_prompt(rules, labels, "current_only", False) + "\n\nLabelled examples (message => code):\n" + "\n".join(f"{examples[i][0]} => {examples[i][1]}" for i in order)


def main() -> None:
    import httpx

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "train":
        train()
        return
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    names = rules["models"]
    rules["models"] = {name: rules_v4["models"][name] for name in names}
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT
    data = _data(rules)
    labels = data["labels"]
    sessions = _sessions(rules, data)
    flat = [turn for session in sessions for turn in session]
    print(f"sessions {len(sessions)} turns {len(flat)} in scope {sum(t['in_scope'] for t in flat)}", flush=True)
    budget = api.Budget(api_rules, run_cap=rules["run_cap_usd"])
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"]}
    brief = lambda result: {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        answers = v4.part_a(rules, labels, sessions, client, budget, names)
        if stage == "collect":
            for name, table in answers.items():
                result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: table[turn["id"]]), None)
                print(f"{name:20s} alone session {100 * result['session_success']:.1f} " + " ".join(f"{k} {100 * v:.1f}" for k, v in result["node_accuracy"].items()), flush=True)
            budget.save()
            print(f"CLINC_COLLECT_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)
            return
        spec = specialists(rules, data, sessions)
        if "encoder" not in spec:
            raise RuntimeError("the specialists are not trained yet; run the train stage on the GPU first")
        summary["specialist_rules"] = {name: block["rule"] for name, block in spec.items()}
        summary["part_a"] = v4.score_a(rules, sessions, answers, spec)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["part_b"] = v4.part_b(rules, labels, sessions, answers, spec, client, budget)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["part_d"] = {}
        for name, runs in rules["part_d_few_shot"]["runs"].items():
            for k, extent in runs.items():
                subset = sessions if extent.startswith("all") else sessions[:150]
                turns = [turn for session in subset for turn in session]
                system = _few_shot_system(rules, data, int(k))
                jobs = [(turn["id"], system, rules["request"]["user"]["current_only"].format(text=turn["text"])) for turn in turns]
                print(f"  part D {name} {k}-shot", flush=True)
                rows = v4._collect(client, rules["models"][name], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"d__{name}__{k}shot.jsonl", budget)
                table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
                evaluate = lambda answer_of: brief(v1.evaluate(subset, v4.decide(subset, answer_of), None))
                entry = {
                    "alone": evaluate(lambda turn: table[turn["id"]]),
                    "zero_shot_alone_same_sessions": evaluate(lambda turn: answers[name][turn["id"]]),
                    "specialists_alone_same_sessions": {kind: evaluate(lambda turn, b=block: b["answers"][turn["id"]]) for kind, block in spec.items()},
                    "usd_per_1000_sessions": 1000.0 * sum(rows[t["id"]]["cost"] for t in turns) / len(subset),
                    "prompt_tokens_mean": float(np.mean([rows[t["id"]]["prompt_tokens"] or 0 for t in turns])),
                }
                entry["n_sessions"] = len(subset)
                entry["first_150_sessions_alone"] = brief(v1.evaluate(sessions[:150], v4.decide(sessions[:150], lambda turn: table[turn["id"]]), None))
                entry["hybrid"] = {}
                for kind, block in spec.items():
                    base = block["answers"]
                    order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
                    entry["hybrid"][kind] = {}
                    for rate in (0.05, 0.1, 0.25, 0.4, 0.6):
                        chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                        entry["hybrid"][kind][str(rate)] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else base[turn["id"]])
                    print(f"  few-shot target, {kind} first: " + " ".join(f"{rate}:{100 * cell['session_success']:.1f}" for rate, cell in entry["hybrid"][kind].items()), flush=True)
                summary["part_d"][f"{name}__{k}-shot"] = entry
                print(
                    f"{name} {k}-shot alone session {100 * entry['alone']['session_success']:.1f} intent {100 * entry['alone']['node_accuracy']['intent']:.1f} scope {100 * entry['alone']['node_accuracy']['scope']:.1f} "
                    f"(zero-shot {100 * entry['zero_shot_alone_same_sessions']['session_success']:.1f}; " + "; ".join(f"{kind} {100 * row['session_success']:.1f}" for kind, row in entry["specialists_alone_same_sessions"].items()) + f") ${entry['usd_per_1000_sessions']:.2f}/1000",
                    flush=True,
                )
        summary["spend_usd"] = {"project_total": budget.total}
        path.write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"WORKFLOW_CLINC_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    main()

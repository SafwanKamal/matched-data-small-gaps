"""Near-domain out-of-scope on BANKING77-OOS. Rules: workflow_near_oos_protocol.json.

Stages: 'hosted' (API only, independent of the specialist), 'train' (GPU), 'report' (processor; needs both).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v3 as v3
from bakeoff import workflow_v4 as v4
from bakeoff.data import Item
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "workflow_near_oos_protocol.json"
DATA = ROOT / "data" / "banking77-oos"
OUT = ROOT / "artifacts" / "workflow-near-oos"
SEED = 20261014


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _data() -> dict:
    train_text, train_label = _lines(DATA / "train" / "seq.in"), _lines(DATA / "train" / "label")
    labels = sorted(set(train_label))
    index = {label: i for i, label in enumerate(labels)}

    def items(split):
        return [Item(item_id=f"b77oos-{split}-{i}", row=i, text=t, gold_index=index[l], gold_label=l) for i, (t, l) in enumerate(zip(_lines(DATA / split / "seq.in"), _lines(DATA / split / "label")))]

    valid = items("valid")
    rng = np.random.default_rng(SEED)
    return {
        "labels": [label.replace("_", " ") for label in labels],
        "train": [(t, index[l]) for t, l in zip(train_text, train_label)],
        "dev_in": [valid[i] for i in sorted(rng.permutation(len(valid))[:400])],
        "test_in": items("test"),
        "dev_out": {"id": _lines(DATA / "id-oos" / "valid" / "seq.in")[:100], "ood": _lines(DATA / "ood-oos" / "valid" / "seq.in")[:100]},
        "test_out": {"id": _lines(DATA / "id-oos" / "test" / "seq.in"), "ood": _lines(DATA / "ood-oos" / "test" / "seq.in")},
    }


def _sessions(data) -> dict:
    out = {}
    for kind in ("id", "ood"):
        path = OUT / f"sessions__{kind}.json"
        if not path.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            # Both pools are cut to the same length so the generator draws the same stream and the in-scope turns match.
            sessions = v1.build_sessions({"seed": SEED, "sessions": {"n": 400, "turns": 5}}, data["test_in"], data["test_out"][kind][:1000])
            for session in sessions:
                for turn in session:
                    turn["id"] = f"{kind}-{turn['id']}"
            path.write_text(json.dumps(sessions, indent=1) + "\n", encoding="utf-8")
        out[kind] = json.loads(path.read_text(encoding="utf-8"))
    same = all(a["text"] == b["text"] for sa, sb in zip(out["id"], out["ood"]) for a, b in zip(sa, sb) if a["in_scope"] and b["in_scope"])
    print(f"in-scope turns identical across the two session sets: {same}", flush=True)
    return out


def hosted() -> None:
    import httpx

    from bakeoff.workflow_retrieval import BM25

    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    data = _data()
    sessions = _sessions(data)
    labels = data["labels"]
    codes = api.codes_for(len(labels))
    options = "\n".join(f"{code}: {label}" for code, label in zip(codes, labels))
    system = spec["hosted"]["system_prompt"].format(options=options)
    model = rules_v4["models"]["gemma-4-31b"]
    index = BM25([text for text, _ in data["train"]])
    v4.OUT = OUT
    budget = api.Budget(api_rules, run_cap=spec["hosted"]["run_cap_usd"])
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        for kind, group in sessions.items():
            turns = [turn for session in group for turn in session]
            for condition in ("zero_shot", "retrieval"):
                jobs = []
                for turn in turns:
                    if condition == "zero_shot":
                        user = f"Current message: {turn['text']}"
                    else:
                        picked = index.top(turn["text"], 20)[::-1]
                        lines = "\n".join(f"{data['train'][i][0]} => {codes[data['train'][i][1]]}" for i in picked)
                        user = f"Similar labelled examples (message => code):\n{lines}\n\nCurrent message: {turn['text']}"
                    jobs.append((turn["id"], system, user))
                print(f"  {kind} {condition}", flush=True)
                v4._collect(client, model, jobs, 24, OUT / f"gemma__{kind}__{condition}.jsonl", budget)
    budget.save()
    print(f"NEAR_OOS_HOSTED_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


def train() -> None:
    import torch

    from bakeoff import train_ettin as enc

    data = _data()
    folder = OUT / "encoder"
    if not (folder / "metrics.json").exists():
        spec = json.loads((ROOT / "program2.json").read_text())
        spec.update({"learning_rate": 1e-4, "seed": SEED})
        stub = {"datasets": {"b77oos": {"labels": data["labels"]}}}
        materialized = {"splits": {"b77oos": {"dev": {"rows": []}}}, "items": {"b77oos": {"dev": data["dev_in"], "test": data["test_in"]}}}
        torch.manual_seed(SEED)
        result = enc._train_one("encoder", "b77oos", "log_loss", spec, stub, materialized, torch.device("cuda"), rows=data["train"], out_root=OUT)
        print(f"trained encoder, in-scope test accuracy {result['scaled']['accuracy']:.4f}", flush=True)
    print("NEAR_OOS_TRAIN_DONE", flush=True)


def report() -> None:
    from bakeoff.oos_detectors import _features, _logsumexp

    data = _data()
    sessions = _sessions(data)
    labels = data["labels"]
    gold = np.asarray([item.gold_index for item in data["dev_in"]], dtype=np.int64)
    summary = {"protocol_id": json.loads(RULES_PATH.read_text(encoding="utf-8"))["protocol_id"], "sets": {}}
    for kind, group in sessions.items():
        turns = [turn for session in group for turn in session]
        in_scope = np.asarray([turn["in_scope"] for turn in turns])
        domain = {"folder": OUT / "encoder", "train": data["train"], "dev_in": [(i.text, i.gold_index) for i in data["dev_in"]], "dev_out": data["dev_out"][kind], "sessions": group}
        features = _features(f"b77oos-{kind}", domain)
        temperature = fit_temperature(features["dev_in_logits"], gold)["temperature"]
        train, train_gold = features["train_emb"], features["train_gold"]
        classes = np.unique(train_gold)
        means = np.stack([train[train_gold == c].mean(axis=0) for c in classes])
        centred = train - means[np.searchsorted(classes, train_gold)]
        precision = np.linalg.pinv(centred.T @ centred / len(train))
        unit = train / np.linalg.norm(train, axis=1, keepdims=True)
        scores = {}
        for part in ("dev_in", "dev_out", "turns"):
            logits, emb = features[f"{part}_logits"], features[f"{part}_emb"]
            maha = np.full(len(emb), np.inf)
            for mean in means:
                diff = emb - mean
                maha = np.minimum(maha, ((diff @ precision) * diff).sum(axis=1))
            sims = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ unit.T
            scores[part] = {"msp": apply_temperature(logits, temperature).max(axis=1), "energy": _logsumexp(logits), "mahalanobis": -maha, "knn": np.sort(sims, axis=1)[:, -10]}
        probs = apply_temperature(features["turns_logits"], temperature)
        intent = probs.argmax(axis=1)
        top = probs.max(axis=1)

        def evaluate(answer_of) -> dict:
            decisions = v4.decide(group, answer_of)
            result = v1.evaluate(group, decisions, None)
            flat = [row for rows in decisions for row in rows]
            return {
                "session_success": result["session_success"],
                "session_success_ci": result["session_success_ci"],
                "scope_accuracy": result["node_accuracy"]["scope"],
                "intent_accuracy": result["node_accuracy"]["intent"],
                "in_scope_kept": float(np.mean([row["scope"] for row, ok in zip(flat, in_scope) if ok])),
                "out_of_scope_caught": float(np.mean([not row["scope"] for row, ok in zip(flat, in_scope) if not ok])),
            }

        position = {turn["id"]: i for i, turn in enumerate(turns)}
        block = {"rows": {}, "auroc": {}}
        for detector in scores["turns"]:
            rule = v1._scope_threshold(scores["dev_in"][detector], scores["dev_out"][detector])
            scope = scores["turns"][detector] >= rule["tau"]
            block["auroc"][detector] = v3._auroc(scores["turns"][detector], in_scope)
            row = evaluate(lambda turn: {"scope": bool(scope[position[turn["id"]]]), "intent": int(intent[position[turn["id"]]])})
            row["dev_balanced_accuracy"] = rule["dev_balanced_accuracy"]
            block["rows"][f"encoder_{detector}"] = row
        best = max(scores["turns"], key=lambda d: block["rows"][f"encoder_{d}"]["dev_balanced_accuracy"])
        block["dev_selected_detector"] = best
        hosted_tables = {}
        for condition in ("zero_shot", "retrieval"):
            rows = v4._read(OUT / f"gemma__{kind}__{condition}.jsonl")
            if len(rows) < len(turns):
                continue
            table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
            hosted_tables[condition] = table
            block["rows"][f"gemma_{condition}_alone"] = evaluate(lambda turn, t=table: t[turn["id"]])
            block["rows"][f"gemma_{condition}_gate_encoder_intent"] = evaluate(lambda turn, t=table: {"scope": t[turn["id"]]["scope"], "intent": int(intent[position[turn["id"]]])})
        if "retrieval" in hosted_tables:
            table = hosted_tables["retrieval"]
            order = sorted(turns, key=lambda turn: top[position[turn["id"]]])
            rule = v1._scope_threshold(scores["dev_in"][best], scores["dev_out"][best])
            scope = scores["turns"][best] >= rule["tau"]
            for rate in (0.25, 0.4):
                chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                block["rows"][f"encoder_{best}_then_gemma_retrieval_{rate}"] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else {"scope": bool(scope[position[turn["id"]]]), "intent": int(intent[position[turn["id"]]])})
        summary["sets"][kind] = block
        print(f"\n{kind}: dev-selected detector {best}; AUROC " + " ".join(f"{d} {v:.3f}" for d, v in block["auroc"].items()), flush=True)
        for key, row in block["rows"].items():
            print(f"  {key:46s} session {100 * row['session_success']:5.1f} [{100 * row['session_success_ci'][0]:.1f}, {100 * row['session_success_ci'][1]:.1f}] scope {100 * row['scope_accuracy']:5.1f} kept {100 * row['in_scope_kept']:5.1f} caught {100 * row['out_of_scope_caught']:5.1f} intent {100 * row['intent_accuracy']:5.1f}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("NEAR_OOS_REPORT_DONE", flush=True)


if __name__ == "__main__":
    {"hosted": hosted, "train": train, "report": report}[sys.argv[1]]()

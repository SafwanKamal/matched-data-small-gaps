"""Encoder-side out-of-scope detectors on the three workflows. Rules: oos_detectors_protocol.json. Processor only."""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v3 as v3
from bakeoff import workflow_v4 as v4
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "oos_detectors_protocol.json"
OUT = ROOT / "artifacts" / "oos-detectors"


def _domain(name: str) -> dict:
    if name == "banking77":
        from bakeoff.data import load_items
        from bakeoff.protocol import load_protocol
        from bakeoff.train_ettin import _train_rows

        protocol = load_protocol()
        materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
        labels, _, dev, _, oos_dev = v1._sources()
        return {
            "labels": labels,
            "folder": ROOT / "artifacts" / "size-sweep" / "encoder-150m" / "lr-0.0001" / "banking77",
            "train": _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"])),
            "dev_in": [(item.text, item.gold_index) for item in dev],
            "dev_out": list(oos_dev),
            "sessions": json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8")),
            "answers_dir": ROOT / "artifacts" / "workflow-v4",
        }
    if name == "massive":
        from bakeoff import workflow_massive as m

        rules = json.loads(m.RULES_PATH.read_text(encoding="utf-8"))
        loaded, sessions, oos_dev = m._sources(rules)
        return {
            "labels": loaded["labels"],
            "folder": m.ENCODERS["encoder"],
            "train": [(item.text, item.gold_index) for item in loaded["items"]["train"]],
            "dev_in": [(item.text, item.gold_index) for item in loaded["items"]["dev"][:400]],
            "dev_out": oos_dev[:100],
            "sessions": sessions,
            "answers_dir": m.OUT,
        }
    from bakeoff import workflow_clinc as c

    rules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
    data = c._data(rules)
    return {
        "labels": data["labels"],
        "folder": c.OUT / "encoder",
        "train": [(item.text, item.gold_index) for item in data["train_in"]],
        "dev_in": [(item.text, item.gold_index) for item in data["dev_in"]],
        "dev_out": data["dev_out"],
        "sessions": c._sessions(rules, data),
        "answers_dir": c.OUT,
    }


def _features(name: str, domain: dict) -> dict:
    """Mean-pooled embeddings and logits for train, dev and session texts. Cached."""
    cache = OUT / f"{name}__features.npz"
    if cache.exists():
        return dict(np.load(cache))
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff.train_ettin import _encode

    torch.set_num_threads(10)
    folder = domain["folder"]
    tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
    encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32).eval()
    head = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"]
    weight, bias = head["weight"].numpy().astype(np.float64), head["bias"].numpy().astype(np.float64)
    turns = [turn for session in domain["sessions"] for turn in session]
    sets = {"train": [text for text, _ in domain["train"]], "dev_in": [text for text, _ in domain["dev_in"]], "dev_out": domain["dev_out"], "turns": [turn["text"] for turn in turns]}
    out = {}
    for part, texts in sets.items():
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                encoded = _encode(tokenizer, texts[start : start + 64], 128, torch.device("cpu"))
                hidden = encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                chunks.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).numpy().astype(np.float64))
        out[f"{part}_emb"] = np.concatenate(chunks, axis=0)
        out[f"{part}_logits"] = out[f"{part}_emb"] @ weight.T + bias
        print(f"  {name} {part} {len(texts)}", flush=True)
    out["train_gold"] = np.asarray([label for _, label in domain["train"]], dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def _logsumexp(x: np.ndarray) -> np.ndarray:
    top = x.max(axis=1, keepdims=True)
    return (top + np.log(np.exp(x - top).sum(axis=1, keepdims=True))).ravel()


def _scores(features: dict, temperature: float) -> dict[str, dict[str, np.ndarray]]:
    train, gold = features["train_emb"], features["train_gold"]
    classes = np.unique(gold)
    means = np.stack([train[gold == c].mean(axis=0) for c in classes])
    centred = train - means[np.searchsorted(classes, gold)]
    precision = np.linalg.pinv(centred.T @ centred / len(train))
    unit = train / np.linalg.norm(train, axis=1, keepdims=True)
    out = {}
    for part in ("dev_in", "dev_out", "turns"):
        logits, emb = features[f"{part}_logits"], features[f"{part}_emb"]
        probs = apply_temperature(logits, temperature)
        maha = np.full(len(emb), np.inf)
        for mean in means:
            diff = emb - mean
            maha = np.minimum(maha, ((diff @ precision) * diff).sum(axis=1))
        sims = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ unit.T
        out[part] = {
            "msp": probs.max(axis=1),
            "max_logit": logits.max(axis=1),
            "energy": _logsumexp(logits),
            "neg_entropy": (probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1),
            "mahalanobis": -maha,
            "knn": np.sort(sims, axis=1)[:, -10],
        }
    return out


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else ["banking77", "massive", "clinc150"]
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"], "domains": {}}
    for name in only:
        domain = _domain(name)
        sessions, labels = domain["sessions"], domain["labels"]
        turns = [turn for session in sessions for turn in session]
        features = _features(name, domain)
        gold = np.asarray([label for _, label in domain["dev_in"]], dtype=np.int64)
        temperature = fit_temperature(features["dev_in_logits"], gold)["temperature"]
        scores = _scores(features, temperature)
        intent = features["turns_logits"].argmax(axis=1)
        in_scope = np.asarray([turn["in_scope"] for turn in turns])

        def evaluate(scope: np.ndarray) -> dict:
            answers = {turn["id"]: {"scope": bool(s), "intent": int(i)} for turn, s, i in zip(turns, scope, intent)}
            result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: answers[turn["id"]]), None)
            return {"session_success": result["session_success"], "session_success_ci": result["session_success_ci"], "scope_accuracy": result["node_accuracy"]["scope"], "in_scope_kept": float(scope[in_scope].mean()), "out_of_scope_caught": float((~scope[~in_scope]).mean())}

        block = {"temperature": temperature, "detectors": {}, "reference": {}}
        print(f"\n{name}", flush=True)
        for detector in rules["detectors"]:
            rule = v1._scope_threshold(scores["dev_in"][detector], scores["dev_out"][detector])
            row = evaluate(scores["turns"][detector] >= rule["tau"])
            row["auroc"] = v3._auroc(scores["turns"][detector], in_scope)
            row["dev_balanced_accuracy"] = rule["dev_balanced_accuracy"]
            block["detectors"][detector] = row
            print(f"  {detector:12s} AUROC {row['auroc']:.3f} dev bal acc {row['dev_balanced_accuracy']:.3f} kept {100 * row['in_scope_kept']:5.1f} caught {100 * row['out_of_scope_caught']:5.1f} scope {100 * row['scope_accuracy']:5.1f} session {100 * row['session_success']:5.1f} [{100 * row['session_success_ci'][0]:.1f}, {100 * row['session_success_ci'][1]:.1f}]", flush=True)
        for model in ("gemma-4-31b", "qwen3.5-9b"):
            rows = v4._read(domain["answers_dir"] / f"a__{model}.jsonl")
            if len(rows) >= len(turns):
                answers = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
                block["reference"][f"{model}_gate"] = evaluate(np.asarray([answers[turn["id"]]["scope"] for turn in turns]))
        jev_path = ROOT / "artifacts" / "workflow-jev-gate" / f"{name}.jsonl"
        if jev_path.exists():
            jev = {json.loads(line)["id"]: json.loads(line) for line in jev_path.open(encoding="utf-8")}
            block["reference"]["jev_listed_gate"] = evaluate(np.asarray([jev[turn["id"]]["scope_listed"] >= 0.5 for turn in turns]))
        for key, row in block["reference"].items():
            print(f"  ref {key:18s} kept {100 * row['in_scope_kept']:5.1f} caught {100 * row['out_of_scope_caught']:5.1f} scope {100 * row['scope_accuracy']:5.1f} session {100 * row['session_success']:5.1f}", flush=True)
        summary["domains"][name] = block
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print("OOS_DETECTORS_DONE", flush=True)




def fractions(name: str = "banking77") -> None:
    """Post hoc companion to section 3.29's data sweep: the banking encoders trained on a fraction of the data,
    each with its dev-selected scope detector, beside the hosted model retrieving from the same fraction."""
    from bakeoff.data_fraction import _subsample

    base = _domain(name)
    banking = name == "banking77"
    sessions = base["sessions"]
    turns = [turn for session in sessions for turn in session]
    in_scope = np.asarray([turn["in_scope"] for turn in turns])
    gold = np.asarray([label for _, label in base["dev_in"]], dtype=np.int64)
    retrieval = json.loads((ROOT / "artifacts" / "workflow-retrieval" / ("summary_fractions.json" if banking else f"summary_fractions__{name}.json")).read_text())["rows"]
    out = {}
    for fraction in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        domain = dict(base)
        domain["folder"] = (ROOT / "artifacts" / "data-fraction" / "encoder" if banking else ROOT / "artifacts" / "data-fraction-domains" / name / "encoder") / f"frac-{fraction:g}"
        domain["train"] = _subsample(base["train"], fraction, 20261012)
        features = _features(f"{name}-frac-{fraction:g}", domain)
        temperature = fit_temperature(features["dev_in_logits"], gold)["temperature"]
        train, train_gold = features["train_emb"], features["train_gold"]
        classes = np.unique(train_gold)
        means = np.stack([train[train_gold == c].mean(axis=0) for c in classes])
        centred = train - means[np.searchsorted(classes, train_gold)]
        cov = centred.T @ centred / len(train)
        # With one or two examples per class the within-class covariance is empty, so it is shrunk toward the total covariance.
        total = np.cov(train.T)
        precision = np.linalg.pinv(0.5 * cov + 0.5 * total + 1e-3 * np.trace(total) / len(total) * np.eye(len(total)))
        unit = train / np.linalg.norm(train, axis=1, keepdims=True)
        k = int(max(1, min(10, np.median(np.bincount(train_gold)))))
        scores = {}
        for part in ("dev_in", "dev_out", "turns"):
            logits, emb = features[f"{part}_logits"], features[f"{part}_emb"]
            maha = np.full(len(emb), np.inf)
            for mean in means:
                diff = emb - mean
                maha = np.minimum(maha, ((diff @ precision) * diff).sum(axis=1))
            sims = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ unit.T
            scores[part] = {"msp": apply_temperature(logits, temperature).max(axis=1), "energy": _logsumexp(logits), "mahalanobis": -maha, "knn": np.sort(sims, axis=1)[:, -k]}
        intent = features["turns_logits"].argmax(axis=1)
        row = {"train_rows": len(domain["train"]), "detectors": {}}
        for detector in scores["turns"]:
            rule = v1._scope_threshold(scores["dev_in"][detector], scores["dev_out"][detector])
            scope = scores["turns"][detector] >= rule["tau"]
            answers = {turn["id"]: {"scope": bool(s), "intent": int(i)} for turn, s, i in zip(turns, scope, intent)}
            result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: answers[turn["id"]]), None)
            row["detectors"][detector] = {"dev_balanced_accuracy": rule["dev_balanced_accuracy"], "session_success": result["session_success"], "session_success_ci": result["session_success_ci"], "scope_accuracy": result["node_accuracy"]["scope"]}
        chosen = max(row["detectors"], key=lambda name: row["detectors"][name]["dev_balanced_accuracy"])
        row["dev_selected"] = chosen
        row["intent_ignoring_scope"] = float(np.mean([int(i) == turn["intent"] for turn, i in zip(turns, intent) if turn["in_scope"]]))
        out[f"{fraction:g}"] = row
        gemma = retrieval.get(f"gemma-4-31b@{fraction:g}", {}).get("retrieval_alone", {})
        best = row["detectors"][chosen]
        print(
            f"frac {fraction:5g} rows {row['train_rows']:5d} encoder intent {100 * row['intent_ignoring_scope']:.1f} | softmax {100 * row['detectors']['msp']['session_success']:.1f} | dev-selected {chosen} {100 * best['session_success']:.1f} "
            f"[{100 * best['session_success_ci'][0]:.1f}, {100 * best['session_success_ci'][1]:.1f}] | gemma retrieval same rows {100 * gemma.get('session_success', float('nan')):.1f} intent {100 * gemma.get('node_accuracy', {}).get('intent', float('nan')):.1f}",
            flush=True,
        )
    (OUT / ("summary_fractions.json" if banking else f"summary_fractions__{name}.json")).write_text(json.dumps(out, indent=2) + "\n")
    print("OOS_FRACTIONS_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fractions":
        for domain_name in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["banking77"]):
            fractions(domain_name)
    else:
        main()

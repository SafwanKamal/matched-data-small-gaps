"""Part 1 of robustness_protocol.json: two more training seeds for each specialist in the headline comparison. GPU."""

from __future__ import annotations

import json

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.data import Item
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "robustness_protocol.json"
OUT = ROOT / "artifacts" / "robustness"
SEEDS = [20261015, 20261016]


def _items(pairs) -> list:
    return [Item(item_id=str(i), row=i, text=t, gold_index=g, gold_label="") for i, (t, g) in enumerate(pairs)]


def _train(job: str, labels, rows, dev_pairs, seed: int):
    import torch

    from bakeoff import train_ettin as enc

    if (OUT / job / "metrics.json").exists():
        return
    spec = json.loads((ROOT / "program2.json").read_text())
    spec.update({"learning_rate": 1e-4, "seed": seed})
    stub = {"datasets": {"d": {"labels": labels}}}
    materialized = {"splits": {"d": {"dev": {"rows": []}}}, "items": {"d": {"dev": _items(dev_pairs), "test": _items(dev_pairs[:200])}}}
    torch.manual_seed(seed)
    enc._train_one(job, "d", "log_loss", spec, stub, materialized, torch.device("cuda"), rows=rows, out_root=OUT)


def _features(folder, train, dev_in, dev_out, texts, labels) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff.train_ettin import _encode

    tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
    encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32).to("cuda").eval()
    head = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"]
    weight, bias = head["weight"].numpy().astype(np.float64), head["bias"].numpy().astype(np.float64)
    out = {}
    for part, rows in {"train": [t for t, _ in train], "dev_in": [t for t, _ in dev_in], "dev_out": list(dev_out), "turns": texts}.items():
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                encoded = _encode(tokenizer, rows[start : start + 64], 128, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state
                hidden = hidden.float()
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                chunks.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).cpu().numpy().astype(np.float64))
        out[f"{part}_emb"] = np.concatenate(chunks, axis=0)
        out[f"{part}_logits"] = out[f"{part}_emb"] @ weight.T + bias
    out["train_gold"] = np.asarray([g for _, g in train], dtype=np.int64)
    del encoder
    torch.cuda.empty_cache()
    return out


def embedding_scores(features: dict) -> dict:
    """The two detector scores that need the embeddings (Mahalanobis, kNN), per part. bakeoff.specialist_scores saves
    them, so the analyses can rerun from the per-turn scores when the features are not available."""
    if "train_emb" not in features:
        return {part: {"mahalanobis": features[f"mahalanobis_{part}"], "knn": features[f"knn_{part}"]} for part in ("dev_in", "dev_out", "turns")}
    train, gold = features["train_emb"], features["train_gold"]
    classes = np.unique(gold)
    means = np.stack([train[gold == c].mean(axis=0) for c in classes])
    centred = train - means[np.searchsorted(classes, gold)]
    precision = np.linalg.pinv(centred.T @ centred / len(train))
    unit = train / np.linalg.norm(train, axis=1, keepdims=True)
    out = {}
    for part in ("dev_in", "dev_out", "turns"):
        emb = features[f"{part}_emb"]
        maha = np.full(len(emb), np.inf)
        for mean in means:
            diff = emb - mean
            maha = np.minimum(maha, ((diff @ precision) * diff).sum(axis=1))
        sims = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ unit.T
        out[part] = {"mahalanobis": -maha, "knn": np.sort(sims, axis=1)[:, -10]}
    return out


def _detector_scores(features: dict, temperature: float) -> dict:
    from bakeoff.oos_detectors import _logsumexp

    emb = embedding_scores(features)
    out = {}
    for part in ("dev_in", "dev_out", "turns"):
        logits = features[f"{part}_logits"]
        out[part] = {"msp": apply_temperature(logits, temperature).max(axis=1), "energy": _logsumexp(logits), "mahalanobis": emb[part]["mahalanobis"], "knn": emb[part]["knn"]}
    return out


def _session_rows(sessions, features, dev_gold) -> dict:
    turns = [turn for session in sessions for turn in session]
    temperature = fit_temperature(features["dev_in_logits"], dev_gold)["temperature"]
    scores = _detector_scores(features, temperature)
    intent = features["turns_logits"].argmax(axis=1)
    position = {turn["id"]: i for i, turn in enumerate(turns)}
    rows = {}
    for detector in scores["turns"]:
        rule = v1._scope_threshold(scores["dev_in"][detector], scores["dev_out"][detector])
        scope = scores["turns"][detector] >= rule["tau"]
        result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: {"scope": bool(scope[position[turn["id"]]]), "intent": int(intent[position[turn["id"]]])}), None)
        rows[detector] = {"dev_balanced_accuracy": rule["dev_balanced_accuracy"], "session_success": result["session_success"]}
    best = max(rows, key=lambda d: rows[d]["dev_balanced_accuracy"])
    return {"softmax": rows["msp"]["session_success"], "dev_selected": rows[best]["session_success"], "dev_selected_detector": best, "detectors": rows}


def synthetic(name: str) -> dict:
    from bakeoff.oos_detectors import _domain

    domain = _domain(name)
    dev_pairs = domain["dev_in"]
    dev_gold = np.asarray([g for _, g in dev_pairs], dtype=np.int64)
    sessions = domain["sessions"]
    texts = [turn["text"] for session in sessions for turn in session]
    out = {}
    for seed in SEEDS:
        job = f"{name}/seed-{seed}"
        _train(job, domain["labels"], domain["train"], dev_pairs, seed)
        features = _features(OUT / job, domain["train"], dev_pairs, domain["dev_out"], texts, domain["labels"])
        out[str(seed)] = _session_rows(sessions, features, dev_gold)
        print(f"{name:10s} seed {seed} softmax {100 * out[str(seed)]['softmax']:.1f} dev-selected {out[str(seed)]['dev_selected_detector']} {100 * out[str(seed)]['dev_selected']:.1f}", flush=True)
    return out


def banking77_oos() -> dict:
    from bakeoff import workflow_near_oos as wn

    data = wn._data()
    sessions = wn._sessions(data)
    dev_pairs = [(item.text, item.gold_index) for item in data["dev_in"]]
    dev_gold = np.asarray([g for _, g in dev_pairs], dtype=np.int64)
    out = {}
    for seed in SEEDS:
        job = f"banking77_oos/seed-{seed}"
        _train(job, data["labels"], data["train"], dev_pairs, seed)
        out[str(seed)] = {}
        for kind, group in sessions.items():
            texts = [turn["text"] for session in group for turn in session]
            features = _features(OUT / job, data["train"], dev_pairs, data["dev_out"][kind], texts, data["labels"])
            out[str(seed)][kind] = _session_rows(group, features, dev_gold)
            row = out[str(seed)][kind]
            print(f"banking77_oos {kind:3s} seed {seed} softmax {100 * row['softmax']:.1f} dev-selected {row['dev_selected_detector']} {100 * row['dev_selected']:.1f}", flush=True)
    return out


def multidogo() -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff import workflow_multidogo as wm
    from bakeoff.robustness_api import score_multidogo
    from bakeoff.train_ettin import EncoderClassifier, _encode

    splits = {split: wm.load(split) for split in ("train", "dev", "test")}
    pairs = lambda split: [(turn["text"], turn["label"]) for conversation in splits[split] for turn in conversation]
    dev_pairs = pairs("dev")
    dev_gold = np.asarray([g for _, g in dev_pairs], dtype=np.int64)
    test_turns = [turn for conversation in splits["test"] for turn in conversation]
    out = {}
    for seed in SEEDS:
        job = f"multidogo/seed-{seed}"
        _train(job, wm.LABELS, pairs("train"), dev_pairs, seed)
        folder = OUT / job
        tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
        encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32)
        model = EncoderClassifier(encoder, encoder.config.hidden_size, len(wm.LABELS))
        model.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"])
        model.to("cuda").eval()
        logits = {}
        for part, texts in (("dev", [t for t, _ in dev_pairs]), ("test", [t["text"] for t in test_turns])):
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(texts), 64):
                    encoded = _encode(tokenizer, texts[start : start + 64], 128, torch.device("cuda"))
                    chunks.append(model(encoded["input_ids"], encoded["attention_mask"]).float().cpu().numpy().astype(np.float64))
            logits[part] = np.concatenate(chunks, axis=0)
        del model
        torch.cuda.empty_cache()
        temperature = fit_temperature(logits["dev"], dev_gold)["temperature"]
        predicted = apply_temperature(logits["test"], temperature).argmax(axis=1)
        table = {turn["id"]: {"label": int(p), "repeat": None} for turn, p in zip(test_turns, predicted)}
        out[str(seed)] = score_multidogo(splits["test"], table)
        print(f"multidogo seed {seed} label {100 * out[str(seed)]['label_accuracy']:.1f} conversations {100 * out[str(seed)]['conversation_success']:.1f}", flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary_seeds.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": json.loads(RULES_PATH.read_text(encoding="utf-8"))["protocol_id"]}
    for name in ("banking77", "massive", "clinc150"):
        if name not in summary:
            summary[name] = synthetic(name)
            path.write_text(json.dumps(summary, indent=2) + "\n")
    if "banking77_oos" not in summary:
        summary["banking77_oos"] = banking77_oos()
        path.write_text(json.dumps(summary, indent=2) + "\n")
    if "multidogo" not in summary:
        summary["multidogo"] = multidogo()
        path.write_text(json.dumps(summary, indent=2) + "\n")
    print("ROBUSTNESS_SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()

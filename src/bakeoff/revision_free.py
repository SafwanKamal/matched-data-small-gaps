"""No-API revision work agreed in the simulated review. Rules: revision_free_protocol.json.

Stages: 'features' (GPU, float32), 'lr' (GPU), 'kplus1' (GPU), 'balanced' (GPU), 'baselines' (GPU), 'analyze' (processor).
Everything here runs on the existing session sets and is exploratory.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "revision_free_protocol.json"
OUT = ROOT / "artifacts" / "revision"
ROBUST = ROOT / "artifacts" / "robustness"
SEEDS = ["20261007", "20261015", "20261016"]  # the first is the original model of each domain
SETS = ["banking77", "massive", "clinc150", "banking77_oos/id", "banking77_oos/ood"]


def _out_pool(name: str) -> list[str]:
    """The labelled out-of-scope rows the general model's retrieval pool held (workflow_retrieval._domain)."""
    if name == "banking77":
        from datasets import load_dataset

        from bakeoff.suite import load_suite_protocol

        clinc = load_suite_protocol()["datasets"]["clinc150"]
        raw = load_dataset(clinc["hf_id"], clinc["config"], revision=clinc["revision"])["train"]
        oos = raw.features["intent"].names.index("oos")
        return [example["text"] for example in raw if example["intent"] == oos]
    if name == "massive":
        from bakeoff import workflow_massive as m

        return m._banking_out_of_scope()[1][104:]
    if name == "clinc150":
        from bakeoff import workflow_clinc as c

        return c._data(json.loads(c.RULES_PATH.read_text(encoding="utf-8")))["train_out"]
    return []


def domain(name: str) -> dict:
    """labels, train, dev_in, dev_out, sessions, the seed folders and the general models' answer files for one analysis set."""
    if name.startswith("banking77_oos"):
        from bakeoff import workflow_near_oos as wn

        kind = name.split("/")[1]
        data = wn._data()
        return {
            "labels": data["labels"],
            "train": data["train"],
            "dev_in": [(item.text, item.gold_index) for item in data["dev_in"]],
            "dev_out": data["dev_out"][kind],
            "sessions": wn._sessions(data)[kind],
            "out_pool": [],
            "folders": {"20261007": wn.OUT / "encoder", **{s: ROBUST / "banking77_oos" / f"seed-{s}" for s in SEEDS[1:]}},
            "models": {"gemma-4-31b": wn.OUT / f"gemma__{kind}__retrieval.jsonl", "qwen3.5-27b": ROBUST / f"b77oos__qwen3.5-27b__{kind}__retrieval.jsonl"},
            "zero_shot": {"gemma-4-31b": wn.OUT / f"gemma__{kind}__zero_shot.jsonl", "qwen3.5-27b": ROBUST / f"b77oos__qwen3.5-27b__{kind}__zero_shot.jsonl"},
        }
    from bakeoff.oos_detectors import _domain

    base = _domain(name)
    return {
        "labels": base["labels"],
        "train": base["train"],
        "dev_in": base["dev_in"],
        "dev_out": list(base["dev_out"]),
        "sessions": base["sessions"],
        "out_pool": _out_pool(name),
        "folders": {"20261007": base["folder"], **{s: ROBUST / name / f"seed-{s}" for s in SEEDS[1:]}},
        "models": {"gemma-4-31b": ROOT / "artifacts" / "workflow-retrieval" / f"{name}__gemma-4-31b__full_train.jsonl", "qwen3.5-27b": ROBUST / f"{name}__qwen3.5-27b__full_train.jsonl"},
    }


def _key(name: str) -> str:
    return name.replace("/", "-")


def embed_with(folder, parts: dict[str, list[str]], device: str = "cuda") -> dict:
    """Mean-pooled float32 embeddings (no autocast, no TF32) and the head's logits for each list of texts."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff.train_ettin import _encode

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
    encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32).to(device).eval()
    head = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"]
    weight, bias = head["weight"].numpy().astype(np.float64), head["bias"].numpy().astype(np.float64)
    out = {}
    for part, texts in parts.items():
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                encoded = _encode(tokenizer, texts[start : start + 64], 128, torch.device(device))
                hidden = encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                chunks.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).cpu().numpy().astype(np.float64))
        out[f"{part}_emb"] = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, weight.shape[1]))
        out[f"{part}_logits"] = out[f"{part}_emb"] @ weight.T + bias
    del encoder
    torch.cuda.empty_cache()
    return out


def features(name: str, seed: str, folder=None, tag: str | None = None) -> dict:
    """Cached float32 features of one specialist: train, dev_in, dev_out, session turns and out-of-scope pool rows."""
    cache = OUT / "features" / f"{_key(name)}__{tag or 'seed-' + seed}.npz"
    if cache.exists():
        return dict(np.load(cache))
    d = domain(name)
    parts = {
        "train": [t for t, _ in d["train"]],
        "dev_in": [t for t, _ in d["dev_in"]],
        "dev_out": list(d["dev_out"]),
        "turns": [turn["text"] for session in d["sessions"] for turn in session],
        "pool": list(d["out_pool"]),
    }
    out = embed_with(folder or d["folders"][seed], parts)
    out["train_gold"] = np.asarray([g for _, g in d["train"]], dtype=np.int64)
    out["dev_gold"] = np.asarray([g for _, g in d["dev_in"]], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def multidogo_logits(seed: str, folder=None, tag: str | None = None) -> dict:
    """Float32 dev and test logits of a MultiDoGO specialist."""
    from bakeoff import workflow_multidogo as wm

    cache = OUT / "features" / f"multidogo__{tag or 'seed-' + seed}.npz"
    if cache.exists():
        return dict(np.load(cache))
    folder = folder or (wm.OUT / "encoder" if seed == SEEDS[0] else ROBUST / "multidogo" / f"seed-{seed}")
    texts = {split: [turn["text"] for conversation in wm.load(split) for turn in conversation] for split in ("dev", "test")}
    got = embed_with(folder, texts)
    out = {"dev": got["dev_logits"], "test": got["test_logits"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def stage_features() -> None:
    for name in SETS:
        for seed in SEEDS:
            f = features(name, seed)
            print(f"features {name} seed {seed}: " + ", ".join(f"{k} {v.shape[0]}" for k, v in f.items() if k.endswith("_emb")), flush=True)
    for seed in SEEDS:
        m = multidogo_logits(seed)
        print(f"features multidogo seed {seed}: test {m['test'].shape}", flush=True)
    print("REVISION_FEATURES_DONE", flush=True)


# ---------------------------------------------------------------- training (GPU)


def _train(folder, labels, rows, dev_pairs, learning_rate: float, epochs: int, seed: int, weights=None) -> None:
    """The Program 2 encoder recipe with the given learning rate, epochs and seed; optional per-class loss weights."""
    import torch

    from bakeoff import train_ettin as enc
    from bakeoff.robustness_seeds import _items

    if (folder / "metrics.json").exists():
        return
    spec = json.loads((ROOT / "program2.json").read_text())
    spec.update({"learning_rate": learning_rate, "epochs": epochs, "seed": seed})
    stub = {"datasets": {"d": {"labels": labels}}}
    materialized = {"splits": {"d": {"dev": {"rows": []}}}, "items": {"d": {"dev": _items(dev_pairs), "test": _items(dev_pairs[:200])}}}
    original = enc.decision_loss
    if weights is not None:
        w = torch.tensor(weights, dtype=torch.float32, device="cuda")
        enc.decision_loss = lambda logits, gold, kind: torch.nn.functional.cross_entropy(logits, gold, weight=w)
    try:
        torch.manual_seed(seed)
        enc._train_one(folder.name, "d", "log_loss", spec, stub, materialized, torch.device("cuda"), rows=rows, out_root=folder.parent)
    finally:
        enc.decision_loss = original


def _training_sets() -> dict:
    """labels, train rows and dev pairs for each domain of the learning-rate sweep."""
    from bakeoff import workflow_multidogo as wm

    out = {}
    for name in ("banking77", "massive", "clinc150", "banking77_oos/id"):
        d = domain(name)
        out[name.split("/")[0]] = {"labels": d["labels"], "train": d["train"], "dev": d["dev_in"]}
    pairs = lambda split: [(t["text"], t["label"]) for c in wm.load(split) for t in c]
    out["multidogo"] = {"labels": wm.LABELS, "train": pairs("train"), "dev": pairs("dev")}
    return out


def stage_lr() -> None:
    from scipy.special import log_softmax

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))["part_g_learning_rate"]
    path = OUT / "lr_sweep.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name, data in _training_sets().items():
        for epochs in rules["grid"]["epochs"]:
            for lr in rules["grid"]["learning_rate"]:
                key = f"lr-{lr:g}__ep-{epochs}"
                if key in summary.get(name, {}):
                    continue
                folder = OUT / "lr" / name / key
                _train(folder, data["labels"], data["train"], data["dev"], lr, epochs, 20261015)
                logits = embed_with(folder, {"dev": [t for t, _ in data["dev"]]})["dev_logits"]
                gold = np.asarray([g for _, g in data["dev"]])
                row = {"dev_accuracy": float((logits.argmax(axis=1) == gold).mean()), "dev_log_loss": float(-log_softmax(logits, axis=1)[np.arange(len(gold)), gold].mean()), "train_loss": json.loads((folder / "metrics.json").read_text())["mean_train_loss"]}
                summary.setdefault(name, {})[key] = row
                path.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"lr {name:14s} {key:18s} dev acc {100 * row['dev_accuracy']:.2f} log loss {row['dev_log_loss']:.4f}", flush=True)
        best = max(summary[name], key=lambda k: (summary[name][k]["dev_accuracy"], -summary[name][k]["dev_log_loss"]))
        summary[name]["selected"] = best
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"lr {name} selected {best}", flush=True)
    print("REVISION_LR_DONE", flush=True)


def stage_kplus1() -> None:
    for name in ("banking77", "massive", "clinc150"):
        d = domain(name)
        k = len(d["labels"])
        rows = list(d["train"]) + [(text, k) for text in d["out_pool"]]
        folder = OUT / "kplus1" / name
        _train(folder, list(d["labels"]) + ["out_of_scope"], rows, d["dev_in"], 1e-4, 3, 20261007)
        f = features(name, SEEDS[0], folder=folder, tag="kplus1")
        print(f"kplus1 {name}: turns logits {f['turns_logits'].shape}", flush=True)
    print("REVISION_KPLUS1_DONE", flush=True)


def stage_balanced() -> None:
    from bakeoff import workflow_multidogo as wm

    data = _training_sets()["multidogo"]
    counts = np.bincount([g for _, g in data["train"]], minlength=len(wm.LABELS)).astype(float)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / (weights * counts).sum() * counts.sum()
    folder = OUT / "balanced" / "multidogo"
    _train(folder, wm.LABELS, data["train"], data["dev"], 1e-4, 3, 20261015, weights=weights.tolist())
    multidogo_logits(SEEDS[1], folder=folder, tag="balanced")
    print("REVISION_BALANCED_DONE", flush=True)


SETFIT_SEEDS = ("20261015", "20261016", "20261017")  # as in the protocol
MPNET = ROOT / ".cache" / "hf" / "hub" / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def _baseline_rows(turns, scope, intent, sessions) -> dict:
    c = turn_correct(turns, scope, intent)
    return {**_summary(turns, scope, intent, c), "session_success": float(_sessions_ok(sessions, scope, intent).mean())}


def _knn_scope(bank: np.ndarray, parts: dict) -> dict:
    """In-scope score: cosine similarity to the 10th-nearest bank row (or the farthest when the bank is smaller)."""
    k = min(10, len(bank))
    return {p: np.sort(e @ bank.T, axis=1)[:, -k] for p, e in parts.items()}


def stage_baselines() -> None:
    """Part H: the retriever without the general model (kNN, centroid on frozen all-mpnet-base-v2) and SetFit, on the specialist's rows."""
    from bakeoff.data_fraction import _subsample
    from bakeoff.workflow_retrieval import embed

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))["part_h_low_data_baselines"]
    path = OUT / "baselines.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in ("banking77", "massive", "clinc150"):
        d = domain(name)
        sessions = d["sessions"]
        turns = [t for s in sessions for t in s]
        fixed = {"dev_in": embed([t for t, _ in d["dev_in"]]), "dev_out": embed(list(d["dev_out"])), "turns": embed([t["text"] for t in turns])}
        for fraction in rules["fractions"]:
            rows = d["train"] if fraction == 1.0 else _subsample(d["train"], fraction, 20261012)
            key = f"{fraction:g}"
            block = summary.setdefault(name, {}).setdefault(key, {"train_rows": len(rows)})
            texts, gold = [t for t, _ in rows], np.asarray([g for _, g in rows])
            if "knn" not in block:
                bank = embed(texts)
                per_class = int(np.median(np.bincount(gold)[np.bincount(gold) > 0]))
                k = max(1, min(10, per_class))
                sims = fixed["turns"] @ bank.T
                top = np.argsort(-sims, axis=1)[:, :k]
                votes = np.zeros((len(turns), len(d["labels"])))
                for j in range(k):
                    np.add.at(votes, (np.arange(len(turns)), gold[top[:, j]]), sims[np.arange(len(turns)), top[:, j]])
                classes = np.unique(gold)
                centroids = np.stack([bank[gold == c].mean(axis=0) for c in classes])
                centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
                scope_scores = _knn_scope(bank, fixed)
                tau, _ = threshold(scope_scores["dev_in"], scope_scores["dev_out"], 0.2)
                scope = scope_scores["turns"] >= tau
                block["knn"] = _baseline_rows(turns, scope, votes.argmax(axis=1), sessions)
                block["centroid"] = _baseline_rows(turns, scope, classes[(fixed["turns"] @ centroids.T).argmax(axis=1)], sessions)
                path.write_text(json.dumps(summary, indent=2) + "\n")
            if fraction <= 0.1:
                for seed in SETFIT_SEEDS:
                    if f"setfit@{seed}" in block:
                        continue
                    block[f"setfit@{seed}"] = _setfit(texts, gold, d, turns, sessions, int(seed))
                    path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"baselines {name} {key} rows {len(rows)}: " + " | ".join(f"{k} intent {100 * v['intent_on_in_scope']:.1f} session {100 * v['session_success']:.1f}" for k, v in block.items() if isinstance(v, dict)), flush=True)
    print("REVISION_BASELINES_DONE", flush=True)


def _setfit(texts, gold, d, turns, sessions, seed: int) -> dict:
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(str(MPNET), device="cuda")
    args = TrainingArguments(output_dir=str(OUT / "setfit-tmp"), save_strategy="no", batch_size=16, num_epochs=1, num_iterations=20, seed=seed, report_to="none", show_progress_bar=False)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_dict({"text": texts, "label": gold.tolist()}))
    trainer.train()
    encode = lambda xs: model.model_body.encode(xs, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    parts = {"dev_in": encode([t for t, _ in d["dev_in"]]), "dev_out": encode(list(d["dev_out"])), "turns": encode([t["text"] for t in turns])}
    bank = encode(texts)
    scope_scores = _knn_scope(bank, parts)
    tau, _ = threshold(scope_scores["dev_in"], scope_scores["dev_out"], 0.2)
    intent = np.asarray(model.predict([t["text"] for t in turns], as_numpy=True)).astype(int)
    del model, trainer
    torch.cuda.empty_cache()
    return _baseline_rows(turns, scope_scores["turns"] >= tau, intent, sessions)


# ---------------------------------------------------------------- analysis (processor)


def threshold(dev_in: np.ndarray, dev_out: np.ndarray, rate: float | None) -> tuple[float, float]:
    """tau maximising balanced accuracy (rate None) or (1 - rate) x kept + rate x caught on dev."""
    best = None
    for tau in np.unique(np.concatenate([dev_in, dev_out])):
        kept, caught = (dev_in >= tau).mean(), (dev_out < tau).mean()
        objective = 0.5 * (kept + caught) if rate is None else (1.0 - rate) * kept + rate * caught
        if best is None or objective > best[1]:
            best = (float(tau), float(objective))
    return best


def specialist(f: dict, rate: float | None) -> dict:
    """Scope from the dev-selected detector under the given threshold rule; intent from the head."""
    from bakeoff.metrics import apply_temperature
    from bakeoff.robustness_seeds import _detector_scores
    from bakeoff.temperature import fit_temperature

    temperature = fit_temperature(f["dev_in_logits"], f["dev_gold"])["temperature"]
    scores = _detector_scores(f, temperature)
    rules = {d: threshold(scores["dev_in"][d], scores["dev_out"][d], rate) for d in scores["turns"]}
    best = max(rules, key=lambda d: rules[d][1])
    probs = apply_temperature(f["turns_logits"], temperature)
    return {
        "detector": best,
        "dev_objective": rules[best][1],
        "scope": scores["turns"][best] >= rules[best][0],
        "intent": probs.argmax(axis=1),
        "top": probs.max(axis=1),
        "scores": scores,
        "rules": rules,
    }


def knn_vote(f: dict, part: str, k: int = 10) -> np.ndarray:
    """True (in scope) unless at least half of the k nearest rows among training plus out-of-scope pool rows are pool rows."""
    if "train_emb" not in f:  # per-turn scores only (bakeoff.specialist_scores)
        return f[f"knn_vote_{part}"]
    bank = np.concatenate([f["train_emb"], f["pool_emb"]])
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    is_pool = np.r_[np.zeros(len(f["train_emb"]), bool), np.ones(len(f["pool_emb"]), bool)]
    emb = f[f"{part}_emb"] / np.linalg.norm(f[f"{part}_emb"], axis=1, keepdims=True)
    top = np.argsort(-(emb @ bank.T), axis=1)[:, :k]
    return is_pool[top].sum(axis=1) < (k + 1) // 2


def adb(f: dict, seed: int = 20261015) -> dict[str, np.ndarray]:
    """Adaptive decision boundary (Zhang et al. 2021) on frozen embeddings: in scope when inside the nearest centroid's learned radius."""
    import torch

    torch.manual_seed(seed)
    x = torch.tensor(f["train_emb"], dtype=torch.float64)
    y = torch.tensor(f["train_gold"])
    classes = torch.unique(y)
    centroids = torch.stack([x[y == c].mean(dim=0) for c in classes])
    index = torch.searchsorted(classes, y)
    raw = torch.zeros(len(classes), dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([raw], lr=0.05)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(100):
        for batch in torch.randperm(len(x), generator=generator).split(256):
            distance = (x[batch] - centroids[index[batch]]).norm(dim=1)
            radius = torch.nn.functional.softplus(raw)[index[batch]]
            outside = (distance > radius).double()
            loss = (outside * (distance - radius) + (1 - outside) * (radius - distance)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    radius = torch.nn.functional.softplus(raw).detach().numpy()
    out = {}
    for part in ("dev_in", "dev_out", "turns"):
        emb, c = f[f"{part}_emb"], centroids.numpy()
        distance = np.sqrt(np.clip((emb**2).sum(axis=1)[:, None] - 2 * emb @ c.T + (c**2).sum(axis=1)[None], 0, None))
        nearest = distance.argmin(axis=1)
        out[part] = distance[np.arange(len(emb)), nearest] <= radius[nearest]
    return out


def turn_correct(turns, scope, intent) -> np.ndarray:
    return np.asarray([(bool(s) == t["in_scope"]) and (not t["in_scope"] or int(i) == t["intent"]) for t, s, i in zip(turns, scope, intent)])


def sign_flip(diff: np.ndarray, clusters: np.ndarray, draws: int = 10000, seed: int = 20261015) -> dict:
    """Paired sign-flip permutation test and percentile bootstrap interval, both by cluster."""
    keys, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    observed = sums.sum()
    flips = rng.choice([-1.0, 1.0], size=(draws, len(sums)))
    p = (1 + np.sum(np.abs(flips @ sums) >= abs(observed) - 1e-12)) / (draws + 1)
    picks = rng.integers(0, len(sums), size=(2000, len(sums)))
    boot = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1) * 100
    return {"difference_points": float(100 * diff.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "p": float(p), "clusters": int(len(keys))}


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * ps[key]))
        out[key] = running
    return out


def _llm(path, n_labels: int, turns) -> dict:
    from bakeoff import workflow_v4 as v4

    rows = v4._read(path)
    parsed = [v4.parse(rows[t["id"]]["text"], n_labels) for t in turns]
    return {"scope": np.asarray([p["scope"] for p in parsed]), "intent": np.asarray([p["intent"] for p in parsed]), "rows": [rows[t["id"]] for t in turns]}


def _sessions_ok(sessions, scope, intent) -> np.ndarray:
    from bakeoff import workflow_v4 as v4
    from bakeoff.paired_headline import session_flags

    position = {t["id"]: i for i, t in enumerate(t for s in sessions for t in s)}
    return session_flags(sessions, v4.decide(sessions, lambda t: {"scope": bool(scope[position[t["id"]]]), "intent": int(intent[position[t["id"]]])}))


def _summary(turns, scope, intent, correct) -> dict:
    in_scope = np.asarray([t["in_scope"] for t in turns])
    gold = np.asarray([t["intent"] if t["in_scope"] else -1 for t in turns])
    return {
        "turn_correct": float(correct.mean()),
        "scope_accuracy": float((np.asarray(scope, bool) == in_scope).mean()),
        "in_scope_kept": float(np.asarray(scope, bool)[in_scope].mean()),
        "out_of_scope_caught": float((~np.asarray(scope, bool))[~in_scope].mean()),
        "intent_on_in_scope": float((np.asarray(intent)[in_scope] == gold[in_scope]).mean()),
    }


def analyze_set(name: str) -> dict:
    d = domain(name)
    sessions, labels = d["sessions"], d["labels"]
    turns = [t for s in sessions for t in s]
    in_scope = np.asarray([t["in_scope"] for t in turns])
    by_intent = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{n}" for n, t in enumerate(turns)])
    by_session = np.asarray([f"s{k}" for k, s in enumerate(sessions) for _ in s])
    feats = {seed: features(name, seed) for seed in SEEDS}
    llms = {model: _llm(path, len(labels), turns) for model, path in d["models"].items()}
    out = {"n_turns": len(turns), "n_sessions": len(sessions), "rules": {}}
    for rule_name, rate in (("balanced", None), ("prior_0.2", 0.2), ("prior_0.1", 0.1), ("prior_0.3", 0.3)):
        block = {"specialist": {}, "systems": {}, "tests": {}}
        spec = {seed: specialist(feats[seed], rate) for seed in SEEDS}
        correct = {seed: turn_correct(turns, spec[seed]["scope"], spec[seed]["intent"]) for seed in SEEDS}
        for seed in SEEDS:
            row = _summary(turns, spec[seed]["scope"], spec[seed]["intent"], correct[seed])
            row["detector"] = spec[seed]["detector"]
            row["session_success"] = float(_sessions_ok(sessions, spec[seed]["scope"], spec[seed]["intent"]).mean())
            block["specialist"][seed] = row
        mean_correct = np.mean([correct[s] for s in SEEDS], axis=0)
        block["specialist"]["mean"] = {k: float(np.mean([block["specialist"][s][k] for s in SEEDS])) for k in ("turn_correct", "scope_accuracy", "in_scope_kept", "out_of_scope_caught", "intent_on_in_scope", "session_success")}
        for model, llm in llms.items():
            llm_correct = turn_correct(turns, llm["scope"], llm["intent"])
            row = _summary(turns, llm["scope"], llm["intent"], llm_correct)
            row["session_success"] = float(_sessions_ok(sessions, llm["scope"], llm["intent"]).mean())
            block["systems"][f"{model} alone"] = row
            block["tests"][f"specialist vs {model}"] = {"by_intent": sign_flip(mean_correct - llm_correct, by_intent), "by_session": sign_flip(mean_correct - llm_correct, by_session), "per_seed": {s: sign_flip(correct[s].astype(float) - llm_correct, by_intent) for s in SEEDS}}
            # Decomposition and the two designs found in review, per seed then averaged.
            designs = {"model scope + specialist intent": [], "specialist scope + model intent": [], "combination (model scope; specialist intent if p>=0.9)": [], "equipped cascade 0.4": []}
            for seed in SEEDS:
                s = spec[seed]
                confident = s["top"] >= 0.9
                escalate = np.zeros(len(turns), bool)
                escalate[np.argsort(s["top"], kind="stable")[: int(round(0.4 * len(turns)))]] = True
                variants = {
                    "model scope + specialist intent": (llm["scope"], s["intent"]),
                    "specialist scope + model intent": (s["scope"], llm["intent"]),
                    "combination (model scope; specialist intent if p>=0.9)": (llm["scope"], np.where(confident, s["intent"], llm["intent"])),
                    "equipped cascade 0.4": (np.where(escalate, llm["scope"], s["scope"]), np.where(escalate, llm["intent"], s["intent"])),
                }
                for key, (sc, it) in variants.items():
                    c = turn_correct(turns, sc, it)
                    designs[key].append((c, _sessions_ok(sessions, sc, it).mean()))
            for key, rows in designs.items():
                c = np.mean([r[0] for r in rows], axis=0)
                block["systems"][f"{model}: {key}"] = {"turn_correct": float(c.mean()), "session_success": float(np.mean([r[1] for r in rows]))}
                block["tests"][f"{model}: {key} vs {model} alone"] = {"by_intent": sign_flip(c - llm_correct, by_intent)}
        out["rules"][rule_name] = block
    # Symmetric data: the specialist given the general model's out-of-scope pool rows (prior-weighted rule).
    if d["out_pool"]:
        sym = {}
        for seed in SEEDS:
            f = feats[seed]
            s = specialist(f, 0.2)
            vote = {part: knn_vote(f, part) for part in ("dev_in", "dev_out", "turns")}
            options = {
                "detector": {p: s["scores"][p][s["detector"]] >= s["rules"][s["detector"]][0] for p in ("dev_in", "dev_out", "turns")},
                "knn_vote": vote,
            }
            options["either"] = {p: options["detector"][p] & vote[p] for p in vote}
            objective = {k: 0.8 * v["dev_in"].mean() + 0.2 * (~v["dev_out"]).mean() for k, v in options.items()}
            chosen = max(objective, key=objective.get)
            scope = options[chosen]["turns"]
            c = turn_correct(turns, scope, s["intent"])
            sym[seed] = {**_summary(turns, scope, s["intent"], c), "rule": chosen, "session_success": float(_sessions_ok(sessions, scope, s["intent"]).mean()), "correct": c}
        mean_c = np.mean([sym[s].pop("correct") for s in SEEDS], axis=0)
        out["symmetric_knn"] = {"per_seed": sym, "tests": {f"vs {m}": sign_flip(mean_c - turn_correct(turns, l["scope"], l["intent"]), by_intent) for m, l in llms.items()}}
    # ADB on each seed's embeddings.
    rows, cs = {}, []
    for seed in SEEDS:
        f = feats[seed]
        s = specialist(f, 0.2)
        scope = adb(f)["turns"]
        c = turn_correct(turns, scope, s["intent"])
        rows[seed] = {**_summary(turns, scope, s["intent"], c), "session_success": float(_sessions_ok(sessions, scope, s["intent"]).mean())}
        cs.append(c)
    out["adb"] = {"per_seed": rows, "tests": {f"vs {m}": sign_flip(np.mean(cs, axis=0) - turn_correct(turns, l["scope"], l["intent"]), by_intent) for m, l in llms.items()}}
    return out


def stage_analyze() -> None:
    summary = {"protocol_id": json.loads(RULES_PATH.read_text(encoding="utf-8"))["protocol_id"], "status": "exploratory", "sets": {}}
    for name in SETS:
        summary["sets"][name] = analyze_set(name)
        print(f"analyzed {name}", flush=True)
    for rule in ("balanced", "prior_0.2"):
        ps = {f"{name} | {key}": summary["sets"][name]["rules"][rule]["tests"][key]["by_intent"]["p"] for name in SETS for key in summary["sets"][name]["rules"][rule]["tests"] if key.startswith("specialist vs")}
        adjusted = holm(ps)
        for label, value in adjusted.items():
            name, key = label.split(" | ")
            summary["sets"][name]["rules"][rule]["tests"][key]["by_intent"]["holm_p"] = value
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in SETS:
        for rule in ("balanced", "prior_0.2"):
            b = summary["sets"][name]["rules"][rule]
            sp = b["specialist"]["mean"]
            print(f"\n{name} [{rule}] specialist (3 seeds): turn {100 * sp['turn_correct']:.1f} scope {100 * sp['scope_accuracy']:.1f} kept {100 * sp['in_scope_kept']:.1f} caught {100 * sp['out_of_scope_caught']:.1f} intent {100 * sp['intent_on_in_scope']:.1f} session {100 * sp['session_success']:.1f}")
            for key, row in b["systems"].items():
                extra = f" scope {100 * row['scope_accuracy']:.1f} kept {100 * row['in_scope_kept']:.1f} caught {100 * row['out_of_scope_caught']:.1f} intent {100 * row['intent_on_in_scope']:.1f}" if "scope_accuracy" in row else ""
                print(f"  {key:70s} turn {100 * row['turn_correct']:.1f}{extra} session {100 * row['session_success']:.1f}")
            for key, t in b["tests"].items():
                x = t["by_intent"]
                print(f"  TEST {key:66s} {x['difference_points']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] p={x['p']:.4f}" + (f" holm={x['holm_p']:.4f}" if "holm_p" in x else "") + (f" | by session p={t['by_session']['p']:.4f}" if "by_session" in t else ""))
        for extra in ("symmetric_knn", "adb"):
            if extra in summary["sets"][name]:
                e = summary["sets"][name][extra]
                means = {k: np.mean([r[k] for r in e["per_seed"].values()]) for k in ("turn_correct", "scope_accuracy", "session_success")}
                print(f"  {extra:14s} turn {100 * means['turn_correct']:.1f} scope {100 * means['scope_accuracy']:.1f} session {100 * means['session_success']:.1f} | " + " ".join(f"{k} {v['difference_points']:+.2f} p={v['p']:.4f}" for k, v in e["tests"].items()))
    print("REVISION_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    {"features": stage_features, "lr": stage_lr, "kplus1": stage_kplus1, "balanced": stage_balanced, "baselines": stage_baselines, "analyze": stage_analyze}[sys.argv[1]]()

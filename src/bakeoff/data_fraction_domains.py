"""Training-data sweep on MASSIVE and CLINC150, joined to the hosted answers. Rules: data_fraction_domains_protocol.json."""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.data import Item
from bakeoff.data_fraction import _subsample
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "data_fraction_domains_protocol.json"
OUT = ROOT / "artifacts" / "data-fraction-domains"
RATES = [0.1, 0.25, 0.4, 0.6]


def _domain(name: str) -> dict:
    if name == "massive":
        from bakeoff import workflow_massive as m
        from bakeoff.massive import load_primary_protocol

        rules = json.loads(m.RULES_PATH.read_text(encoding="utf-8"))
        loaded, sessions, oos_dev = m._sources(rules)
        return {
            "labels": loaded["labels"],
            "train": [(item.text, item.gold_index) for item in loaded["items"]["train"]],
            "dev_items": loaded["items"]["dev"],
            "test_items": loaded["items"]["test"],
            "dev_in": loaded["items"]["dev"][:400],
            "dev_out": oos_dev[:100],
            "sessions": sessions,
            "answers_dir": m.OUT,
            "full": {"encoder@1": ("encoder", m.ENCODERS["encoder"]), "decoder@1": ("decoder", m.DECODER)},
            "decoder_spec": load_primary_protocol()["ettin_decoder"],
        }
    from bakeoff import workflow_clinc as c

    rules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
    data = c._data(rules)
    return {
        "labels": data["labels"],
        "train": [(item.text, item.gold_index) for item in data["train_in"]],
        "dev_items": data["val_in_all"],
        "test_items": data["test_in"],
        "dev_in": data["dev_in"],
        "dev_out": data["dev_out"],
        "sessions": c._sessions(rules, data),
        "answers_dir": c.OUT,
        "full": {"encoder@1": ("encoder", c.OUT / "encoder")},
        "decoder_spec": None,
    }


def _jobs(rules, name: str) -> list[tuple[str, float]]:
    jobs = [("encoder", fraction) for fraction in rules["fractions"]["encoder"]]
    if name == "massive":
        jobs += [("decoder", fraction) for fraction in rules["fractions"]["massive_decoder"]]
    return jobs


def _score(kind: str, folder, texts: list[str], labels, decoder_spec) -> np.ndarray:
    import torch

    if kind == "encoder":
        from bakeoff.workflow_massive import _score_texts

        return _score_texts("encoder", folder, texts, labels, "cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bakeoff.train_ettin_decoder import _score_items

    saved = json.loads((folder / "metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(folder / "model")
    model = AutoModelForCausalLM.from_pretrained(folder / "model", dtype=torch.float32)
    model.config.use_cache = False
    model.to("cuda")
    items = [Item(item_id=str(i), row=i, text=text, gold_index=0, gold_label="") for i, text in enumerate(texts)]
    scores, _, _ = _score_items(model, tokenizer, items, labels, decoder_spec, torch.device("cuda"), saved["batch_size"], saved["label_chunk_size"])
    del model
    torch.cuda.empty_cache()
    return scores


def train(name: str) -> None:
    import torch

    from bakeoff import train_ettin as enc
    from bakeoff import train_ettin_decoder as dec

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    domain = _domain(name)
    labels = domain["labels"]
    stub = {"datasets": {name: {"labels": labels}}}
    materialized = {"splits": {name: {"dev": {"rows": []}}}, "items": {name: {"dev": domain["dev_items"], "test": domain["test_items"]}}}
    turns = [turn for session in domain["sessions"] for turn in session]
    sets = {"session_turns": [t["text"] for t in turns], "dev_in_scope": [item.text for item in domain["dev_in"]], "dev_out_of_scope": domain["dev_out"]}
    for kind, fraction in _jobs(rules, name):
        job = f"{name}/{kind}/frac-{fraction:g}"
        cache = OUT / "scores" / f"{name}__{kind}@{fraction:g}.npz"
        if cache.exists():
            print(f"skip {job}", flush=True)
            continue
        if not (OUT / job / "metrics.json").exists():
            rows = _subsample(domain["train"], fraction, rules["seed"])
            if kind == "encoder":
                spec = json.loads((ROOT / "program2.json").read_text())
                spec["learning_rate"] = 1e-4
            else:
                spec = dict(domain["decoder_spec"])
            spec["seed"] = rules["seed"]
            spec["epochs"] = max(3, math.ceil(300 / math.ceil(len(rows) / spec["batch_size"])))
            print(f"start {job} rows {len(rows)} epochs {spec['epochs']}", flush=True)
            torch.manual_seed(rules["seed"])
            module = enc if kind == "encoder" else dec
            result = module._train_one(job, name, "log_loss", spec, stub, materialized, torch.device("cuda"), rows=rows, out_root=OUT)
            print(f"done {job} rows {len(rows)} acc {result['scaled']['accuracy']:.4f}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{part: _score(kind, OUT / job, texts, labels, domain["decoder_spec"]) for part, texts in sets.items()})
        print(f"scored {job}", flush=True)
    print(f"DOMAIN_TRAIN_DONE {name}", flush=True)


def join(name: str) -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    domain = _domain(name)
    labels, sessions = domain["labels"], domain["sessions"]
    turns = [turn for session in sessions for turn in session]
    gold = np.asarray([item.gold_index for item in domain["dev_in"]], dtype=np.int64)
    counts = {}
    for _, label in domain["train"]:
        counts[label] = counts.get(label, 0) + 1

    def block(scores: dict) -> dict:
        temperature = fit_temperature(scores["dev_in_scope"], gold)["temperature"]
        rule = v1._scope_threshold(apply_temperature(scores["dev_in_scope"], temperature).max(axis=1), apply_temperature(scores["dev_out_of_scope"], temperature).max(axis=1))
        probs = apply_temperature(scores["session_turns"], temperature)
        return {turn["id"]: {"scope": bool(row.max() >= rule["tau"]), "intent": int(row.argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)}

    specialists, sizes = {}, {}
    for kind, fraction in _jobs(rules, name):
        cache = OUT / "scores" / f"{name}__{kind}@{fraction:g}.npz"
        if cache.exists():
            specialists[f"{kind}@{fraction:g}"] = block(dict(np.load(cache)))
            sizes[f"{kind}@{fraction:g}"] = sum(max(1, int(round(fraction * n))) for n in counts.values())
    for key, (kind, folder) in domain["full"].items():
        cache = domain["answers_dir"] / f"{key.split('@')[0]}__scores.npz"
        if cache.exists():
            specialists[key] = block(dict(np.load(cache)))
            sizes[key] = len(domain["train"])
    models = {}
    for model in rules_v4["models"]:
        rows = v4._read(domain["answers_dir"] / f"a__{model}.jsonl")
        if len(rows) == len(turns):
            models[model] = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
    brief = lambda result: {"session_success": result["session_success"], "session_success_ci": result["session_success_ci"], "turn_success": result["turn_success"], "node_accuracy": result["node_accuracy"]}
    summary = {"protocol_id": rules["protocol_id"], "domain": name, "models_alone": {}, "specialists": {}}
    for model, answers in models.items():
        summary["models_alone"][model] = brief(v1.evaluate(sessions, v4.decide(sessions, lambda turn: answers[turn["id"]]), None))
    in_scope = [turn for turn in turns if turn["in_scope"]]
    for key, base in specialists.items():
        order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
        row = {"train_rows": sizes[key], "alone": brief(v1.evaluate(sessions, v4.decide(sessions, lambda turn: base[turn["id"]]), None)), "intent_ignoring_scope": float(np.mean([base[t["id"]]["intent"] == t["intent"] for t in in_scope])), "hybrid": {}}
        for model, answers in models.items():
            row["hybrid"][model] = {}
            for rate in RATES:
                chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                row["hybrid"][model][str(rate)] = brief(v1.evaluate(sessions, v4.decide(sessions, lambda turn: answers[turn["id"]] if turn["id"] in chosen else base[turn["id"]]), None))
        summary["specialists"][key] = row
    (OUT / f"summary__{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{name} models alone: " + "  ".join(f"{model} {100 * row['session_success']:.1f}" for model, row in summary["models_alone"].items()))
    for key, row in sorted(summary["specialists"].items(), key=lambda item: (item[0].split("@")[0], item[1]["train_rows"])):
        best = {model: max(cells.items(), key=lambda item: item[1]["session_success"]) for model, cells in row["hybrid"].items()}
        print(
            f"  {key:14s} rows {row['train_rows']:6d} alone {100 * row['alone']['session_success']:5.1f} scope {100 * row['alone']['node_accuracy']['scope']:5.1f} intent ignoring scope {100 * row['intent_ignoring_scope']:5.1f} | best hybrid: "
            + "  ".join(f"{model.split('-')[0]}{rules_v4['models'][model]['params_b']}B {100 * cell['session_success']:.1f}@{rate}" for model, (rate, cell) in best.items() if model in ("qwen3.5-9b", "gemma-4-31b", "qwen3.5-122b-a10b"))
        )
    print(f"DOMAIN_JOIN_DONE {name}")


if __name__ == "__main__":
    {"train": train, "join": join}[sys.argv[1]](sys.argv[2])

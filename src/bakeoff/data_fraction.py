"""Specialist training-data and size sweep joined to the workflow v4 ladder. Rules: data_fraction_protocol.json."""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.data import Item, load_items
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "data_fraction_protocol.json"
OUT = ROOT / "artifacts" / "data-fraction"
SWEEP = ROOT / "artifacts" / "size-sweep"
HYBRID_RATES = [0.1, 0.25, 0.4, 0.6]


def _subsample(rows, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    by_label: dict[int, list] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    out = []
    for label in sorted(by_label):
        group = by_label[label]
        keep = max(1, int(round(fraction * len(group))))
        out.extend(group[i] for i in rng.permutation(len(group))[:keep])
    return out


def train() -> None:
    import torch

    from bakeoff import train_ettin as enc
    from bakeoff import train_ettin_decoder as dec

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    excluded = set(materialized["splits"]["banking77"]["dev"]["rows"])
    full = enc._train_rows("banking77", protocol, excluded)
    device = torch.device("cuda")
    for kind, module, spec_file in (("encoder", enc, "program2.json"), ("decoder", dec, "program2_decoder.json")):
        for fraction in rules["fractions"][kind]:
            job = f"{kind}/frac-{fraction:g}"
            if (OUT / job / "metrics.json").exists():
                print(f"skip {job}", flush=True)
                continue
            rows = _subsample(full, fraction, rules["seed"])
            spec = json.loads((ROOT / spec_file).read_text())
            spec["seed"] = rules["seed"]
            if kind == "encoder":
                spec["learning_rate"] = 1e-4
                batch = spec["batch_size"]
            else:
                batch = spec["per_dataset"]["banking77"]["batch_size"]
            spec["epochs"] = max(3, math.ceil(300 / math.ceil(len(rows) / batch)))
            print(f"start {job} rows {len(rows)} epochs {spec['epochs']}", flush=True)
            torch.manual_seed(rules["seed"])
            torch.cuda.reset_peak_memory_stats()
            result = module._train_one(job, "banking77", "log_loss", spec, protocol, materialized, device, rows=rows, out_root=OUT)
            print(f"done {job} rows {len(rows)} acc {result['scaled']['accuracy']:.4f}", flush=True)
    score_gpu()
    print("DATA_FRACTION_TRAIN_DONE", flush=True)


def _specialist_dirs(rules) -> dict:
    out = {}
    for fraction in rules["fractions"]["encoder"]:
        out[f"encoder-150m@{fraction:g}"] = ("encoder", OUT / "encoder" / f"frac-{fraction:g}")
    out["encoder-150m@1"] = ("encoder", SWEEP / "encoder-150m" / "lr-0.0001" / "banking77")
    for size in ("17m", "32m", "68m"):
        out[f"encoder-{size}@1"] = ("encoder", SWEEP / f"encoder-{size}" / "lr-0.0001" / "banking77")
    for fraction in rules["fractions"]["decoder"]:
        out[f"decoder-150m@{fraction:g}"] = ("decoder", OUT / "decoder" / f"frac-{fraction:g}")
    return out


def _texts():
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    labels, _, dev, _, oos_dev = v1._sources()
    turns = [turn for session in sessions for turn in session]
    return sessions, labels, dev, {"session_turns": [t["text"] for t in turns], "dev_in_scope": [item.text for item in dev], "dev_out_of_scope": list(oos_dev)}


def score_gpu() -> None:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from bakeoff import train_ettin as enc
    from bakeoff import train_ettin_decoder as dec

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    _, labels, _, texts = _texts()
    device = torch.device("cuda")
    for name, (kind, folder) in _specialist_dirs(rules).items():
        cache = OUT / "scores" / f"{name}.npz"
        if cache.exists():
            continue
        if not folder.exists():
            print(f"missing {name}: {folder}", flush=True)
            continue
        scores = {}
        if kind == "encoder":
            tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
            encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32)
            model = enc.EncoderClassifier(encoder, encoder.config.hidden_size, len(labels))
            model.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"])
            model.to(device).eval()
            for part, rows in texts.items():
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(rows), 64):
                        encoded = enc._encode(tokenizer, rows[start : start + 64], 128, device)
                        chunks.append(model(encoded["input_ids"], encoded["attention_mask"]).cpu().numpy().astype(np.float64))
                scores[part] = np.concatenate(chunks, axis=0)
        else:
            saved = json.loads((folder / "metrics.json").read_text())
            tokenizer = AutoTokenizer.from_pretrained(folder / "model")
            model = AutoModelForCausalLM.from_pretrained(folder / "model", dtype=torch.float32)
            model.config.use_cache = False
            model.to(device)
            spec = json.loads((ROOT / "program2_decoder.json").read_text())
            for part, rows in texts.items():
                items = [Item(item_id=str(i), row=i, text=text, gold_index=0, gold_label="") for i, text in enumerate(rows)]
                scores[part], _, _ = dec._score_items(model, tokenizer, items, labels, spec, device, saved["batch_size"], saved["label_chunk_size"])
        del model
        torch.cuda.empty_cache()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **scores)
        print(f"scored {name}", flush=True)


def _block(scores: dict, dev, turns) -> dict:
    gold = np.asarray([item.gold_index for item in dev], dtype=np.int64)
    temperature = fit_temperature(scores["dev_in_scope"], gold)["temperature"]
    rule = v1._scope_threshold(apply_temperature(scores["dev_in_scope"], temperature).max(axis=1), apply_temperature(scores["dev_out_of_scope"], temperature).max(axis=1))
    probs = apply_temperature(scores["session_turns"], temperature)
    return {turn["id"]: {"scope": bool(row.max() >= rule["tau"]), "intent": int(row.argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)}


def join() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    v4_rules = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    sessions, labels, dev, _ = _texts()
    turns = [turn for session in sessions for turn in session]
    specialists = {}
    for name in _specialist_dirs(rules):
        cache = OUT / "scores" / f"{name}.npz"
        if cache.exists():
            specialists[name] = _block(dict(np.load(cache)), dev, turns)
    specialists["decoder-150m@1"] = {key: value for key, value in v4._specialists(turns)["decoder"]["answers"].items()}
    models = {}
    for name in v4_rules["models"]:
        rows = v4._read(v4.OUT / f"a__{name}.jsonl")
        if len(rows) == len(turns):
            models[name] = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
    brief = lambda result: {"session_success": result["session_success"], "session_success_ci": result["session_success_ci"], "turn_success": result["turn_success"], "node_accuracy": result["node_accuracy"]}
    summary = {"protocol_id": rules["protocol_id"], "models_alone": {}, "specialists": {}}
    for name, answers in models.items():
        summary["models_alone"][name] = brief(v1.evaluate(sessions, v4.decide(sessions, lambda turn: answers[turn["id"]]), None))
    for name, base in specialists.items():
        order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
        row = {"alone": brief(v1.evaluate(sessions, v4.decide(sessions, lambda turn: base[turn["id"]]), None)), "hybrid": {}}
        for model, answers in models.items():
            row["hybrid"][model] = {}
            for rate in HYBRID_RATES:
                chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: answers[turn["id"]] if turn["id"] in chosen else base[turn["id"]]), None)
                row["hybrid"][model][str(rate)] = brief(result)
        summary["specialists"][name] = row
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("models alone: " + "  ".join(f"{name} {100 * row['session_success']:.1f}" for name, row in summary["models_alone"].items()))
    for name, row in summary["specialists"].items():
        best = {model: max(cells.items(), key=lambda item: item[1]["session_success"]) for model, cells in row["hybrid"].items()}
        print(
            f"{name:22s} alone {100 * row['alone']['session_success']:5.1f} intent {100 * row['alone']['node_accuracy']['intent']:5.1f} scope {100 * row['alone']['node_accuracy']['scope']:5.1f} | best hybrid: "
            + "  ".join(f"{model.split('-')[0]}{v4_rules['models'][model]['params_b']}B {100 * cell['session_success']:.1f}@{rate}" for model, (rate, cell) in best.items())
        )
    print("DATA_FRACTION_JOIN_DONE")


if __name__ == "__main__":
    {"train": train, "score": score_gpu, "join": join}[sys.argv[1]]()

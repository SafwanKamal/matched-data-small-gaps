"""Whole official Banking77 test split. Frozen rules are in full_banking77_protocol.json."""

from __future__ import annotations

import json
import sys

import httpx
import numpy as np
import torch

from bakeoff import hybrid, jev
from bakeoff.data import Item, _labels_for, _load_raw, _rows
from bakeoff.metrics import apply_temperature, score_probabilities
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

PROTOCOL_PATH = ROOT / "full_banking77_protocol.json"
OUT = ROOT / "artifacts" / "banking77-full"
SEED = 20261008


def load_full_test(protocol: dict, spec: dict) -> tuple[list[Item], list[str]]:
    raw = _load_raw(protocol)["banking77"]
    labels = _labels_for("banking77", raw, protocol)
    items = [
        Item(item_id=spec["item_id"].format(row=row), row=row, text=text, gold_index=gold, gold_label=labels[gold])
        for row, text, gold in _rows("banking77", raw, "test")
    ]
    counts = np.bincount([item.gold_index for item in items], minlength=len(labels))
    if len(items) != spec["expected_n"] or set(counts.tolist()) != {spec["per_class_n"]}:
        raise RuntimeError(f"unexpected Banking77 test split: n={len(items)} class counts {sorted(set(counts.tolist()))}")
    return items, labels


def _temperature(artifact) -> float:
    return float(json.loads((artifact / "metrics.json").read_text())["temperature"]["temperature"])


def score_encoder(artifact, items, labels) -> np.ndarray:
    return hybrid.score_encoder(artifact, items, hybrid._encoder_spec(), len(labels))


def score_decoder(artifact, items, labels) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bakeoff.train_ettin_decoder import _score_items

    spec = hybrid._decoder_spec()
    saved = json.loads((artifact / "metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(artifact / "model")
    model = AutoModelForCausalLM.from_pretrained(artifact / "model", dtype=torch.float32)
    model.config.use_cache = False
    model.to("cuda")
    scores, _, _ = _score_items(model, tokenizer, items, labels, spec, torch.device("cuda"), saved["batch_size"], saved["label_chunk_size"])
    del model
    torch.cuda.empty_cache()
    return scores


def score_zeroshot(name: str, items, labels, protocol: dict) -> np.ndarray:
    from bakeoff.systems import build_systems

    system = next(candidate for candidate in build_systems(protocol) if candidate.name == name)
    system.load()
    rows = []
    for index, item in enumerate(items, start=1):
        scores, _ = system.score(item.text, labels)
        rows.append(scores)
        if index % 500 == 0:
            print(f"  {name} {index}/{len(items)}", flush=True)
    system.close()
    return np.stack(rows, axis=0)


def _dev_scores(path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return (
        np.asarray([row["scores"] for row in rows], dtype=np.float64),
        np.asarray([row["gold_index"] for row in rows], dtype=np.int64),
    )


def score_jev(items, labels) -> np.ndarray:
    spec = json.loads(jev.JEV_PROTOCOL_PATH.read_text())
    path = OUT / "jev__test.jsonl"
    with httpx.Client(headers={"Authorization": f"Bearer {jev._api_key(spec)}"}, timeout=60.0) as client:
        for attempt in range(4):
            rows = jev._collect(client, spec, items, labels, "banking77", "test-full", path)
            errors = sum(row["error"] is not None for row in rows)
            print(f"  jev pass {attempt + 1}: {errors} rows in error", flush=True)
            if not errors:
                break
    if errors:
        raise RuntimeError(f"{errors} Jev rows still in error after 4 passes")
    return np.asarray([row["scores"] for row in rows], dtype=np.float64)


def macro_f1(choice: np.ndarray, gold: np.ndarray, n_labels: int) -> float:
    scores = []
    for label in range(n_labels):
        tp = float(((choice == label) & (gold == label)).sum())
        fp = float(((choice == label) & (gold != label)).sum())
        fn = float(((choice != label) & (gold == label)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def _interval(values: np.ndarray, rng) -> list[float]:
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(2000)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(probs: np.ndarray, gold: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    scored = score_probabilities(probs, gold)
    scored.pop("risk_coverage", None)
    scored.pop("ece_bins", None)
    choice = probs.argmax(axis=1)
    scored["macro_f1"] = macro_f1(choice, gold, probs.shape[1])
    scored["accuracy_ci"] = _interval((choice == gold).astype(np.float64), rng)
    scored["brier_ci"] = _interval(hybrid.item_brier(probs, gold), rng)
    return scored


def main() -> None:
    spec = json.loads(PROTOCOL_PATH.read_text())
    protocol = load_protocol()
    items, labels = load_full_test(protocol, spec)
    gold = np.asarray([item.gold_index for item in items], dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    probs = {}
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol_id": spec["protocol_id"], "n": len(items), "systems": {}}
    for name, source in spec["systems"].items():
        if only and name not in only:
            continue
        cache = OUT / f"{name}__scores.npz"
        if cache.exists():
            scores = np.load(cache)["scores"]
        else:
            print(f"scoring {name}", flush=True)
            if name.startswith("ettin-encoder"):
                scores = score_encoder(ROOT / source, items, labels)
            elif name.startswith("ettin-decoder"):
                scores = score_decoder(ROOT / source, items, labels)
            elif name == "jev":
                scores = score_jev(items, labels)
            else:
                scores = score_zeroshot(name, items, labels, protocol)
            np.savez_compressed(cache, scores=scores, gold=gold)
        if name.startswith("ettin"):
            temperature = _temperature(ROOT / source)
        elif name == "jev":
            temperature = fit_temperature(*_dev_scores(ROOT / "artifacts" / "jev" / "banking77__dev.jsonl"))["temperature"]
        else:
            temperature = fit_temperature(*_dev_scores(ROOT / "artifacts" / "decisions" / f"{name}__banking77__dev.jsonl"))["temperature"]
        probs[name] = apply_temperature(scores, temperature)
        summary["systems"][name] = {"temperature": temperature, "scaled": summarize(probs[name], gold)}
        if name == "jev":
            summary["systems"][name]["as_delivered"] = summarize(apply_temperature(scores, 1.0), gold)
        row = summary["systems"][name]["scaled"]
        print(f"{name}: acc {row['accuracy']:.4f} macro-F1 {row['macro_f1']:.4f} brier {row['brier']:.4f}", flush=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    cascade = spec["cascade"]
    if cascade["gate"] in probs and cascade["strong"] in probs:
        cheap, strong = probs[cascade["gate"]], probs[cascade["strong"]]
        rules = json.loads((ROOT / "artifacts" / "hybrid" / "thresholds.json").read_text())["datasets"]["banking77"][cascade["gate"]]
        control = hybrid.load_hybrid_protocol()["controls"]
        rows = []
        for rule in rules:
            mask = hybrid.escalate_mask(hybrid.confidence_of(cheap), rule)
            mixed = hybrid.hybrid_probs(cheap, strong, mask)
            called = int(mask.sum())
            random = hybrid.random_control(cheap, strong, gold, called, control["random_draws"], control["random_seed"])
            random_briers = random.pop("briers")
            brier = float(hybrid.item_brier(mixed, gold).mean())
            rows.append(
                {
                    "rate": rule["rate"],
                    "strong_call_fraction": called / len(gold),
                    "accuracy": float((mixed.argmax(axis=1) == gold).mean()),
                    "brier": brier,
                    "random": random,
                    "random_draws_at_or_below_observed": float(np.mean(random_briers <= brier)),
                    "oracle": hybrid.oracle_control(cheap, strong, gold, called),
                }
            )
            print(f"cascade rate {rule['rate']:.2f} called {called / len(gold):.3f} acc {rows[-1]['accuracy']:.4f} brier {brier:.4f} random {random['brier_mean']:.4f}", flush=True)
        summary["cascade"] = {"gate": cascade["gate"], "strong": cascade["strong"], "rates": rows}
        diff = hybrid.item_brier(cheap, gold) - hybrid.item_brier(strong, gold)
        rng = np.random.default_rng(SEED)
        cheap_right, strong_right = cheap.argmax(axis=1) == gold, strong.argmax(axis=1) == gold
        summary["encoder_vs_decoder"] = {
            "brier_difference_encoder_minus_decoder": float(diff.mean()),
            "ci": _interval(diff, rng),
            "encoder_only_right": int((cheap_right & ~strong_right).sum()),
            "decoder_only_right": int((~cheap_right & strong_right).sum()),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

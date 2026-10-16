"""Confidence cascade from checkpoints that already exist.

Thresholds are fit on dev confidence only. Test metrics are a separate command
and refuse to run until artifacts/hybrid/thresholds.json exists for this protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bakeoff.metrics import apply_temperature, schema_valid, score_probabilities
from bakeoff.protocol import ROOT, load_protocol

HYBRID_PATH = ROOT / "hybrid_protocol.json"
STORE_MEDIA = ROOT / "artifacts" / "hybrid" / "plots"
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def load_hybrid_protocol() -> dict:
    return json.loads(HYBRID_PATH.read_text())


def fit_threshold(confidence: np.ndarray, rate: float) -> dict:
    """Choose tau from dev confidence. Labels are not an argument."""
    values = np.asarray(confidence, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        raise ValueError("empty dev split")
    k = int(round(float(rate) * n))
    if k <= 0:
        return {"rate": float(rate), "k": 0, "mode": "none", "tau": None, "dev_n": n}
    if k >= n:
        return {"rate": float(rate), "k": n, "mode": "all", "tau": None, "dev_n": n}
    tau = float(np.sort(values)[k - 1])
    return {"rate": float(rate), "k": k, "mode": "leq", "tau": tau, "dev_n": n}


def escalate_mask(confidence: np.ndarray, rule: dict) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64)
    if rule["mode"] == "none":
        return np.zeros(values.shape, dtype=bool)
    if rule["mode"] == "all":
        return np.ones(values.shape, dtype=bool)
    if rule["mode"] != "leq":
        raise ValueError(rule["mode"])
    return values <= float(rule["tau"])


def hybrid_probs(cheap: np.ndarray, strong: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cheap = np.asarray(cheap, dtype=np.float64)
    strong = np.asarray(strong, dtype=np.float64)
    if cheap.shape != strong.shape:
        raise ValueError(f"shape {cheap.shape} != {strong.shape}")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (cheap.shape[0],):
        raise ValueError("mask length")
    out = np.array(cheap, copy=True)
    out[mask] = strong[mask]
    return out


def confidence_of(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs, dtype=np.float64).max(axis=1)


def agreement_table(cheap: np.ndarray, strong: np.ndarray) -> dict:
    cheap = np.asarray(cheap, dtype=np.float64)
    strong = np.asarray(strong, dtype=np.float64)
    cheap_choice = np.argmax(cheap, axis=1)
    strong_choice = np.argmax(strong, axis=1)
    agree = cheap_choice == strong_choice
    conf = confidence_of(cheap)
    bins = []
    for lo, hi in BINS:
        if hi >= 1.0:
            take = (conf >= lo) & (conf <= hi)
        else:
            take = (conf >= lo) & (conf < hi)
        n = int(take.sum())
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "agreement": float(agree[take].mean()) if n else None,
            }
        )
    return {"n": int(len(agree)), "agreement": float(agree.mean()), "bins": bins}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def zeroshot_probs(gate: dict, dataset: str, split: str, items) -> np.ndarray:
    path = ROOT / gate["decisions"].format(dataset=dataset, split=split)
    by_id = {row["item_id"]: row for row in _read_jsonl(path)}
    chosen = []
    for item in items:
        row = by_id[item.item_id]
        if int(row["gold_index"]) != int(item.gold_index):
            raise RuntimeError(f"gold mismatch {item.item_id}")
        probs = np.asarray(row["probs_scaled"], dtype=np.float64)
        if not schema_valid(probs, len(probs)):
            raise RuntimeError(f"invalid simplex {item.item_id}")
        chosen.append(probs)
    return np.stack(chosen, axis=0)


def _temperature(metrics_path: Path) -> float:
    payload = json.loads(metrics_path.read_text())
    return float(payload["temperature"]["temperature"])


def ettin_test_probs(artifact: Path, temperature: float, items) -> np.ndarray:
    stored = np.load(artifact / "test_scores.npz")
    gold = stored["gold"]
    if gold.shape != (len(items),) or not np.array_equal(gold, [item.gold_index for item in items]):
        raise RuntimeError(f"saved test gold does not match the frozen split at {artifact}")
    return apply_temperature(stored["scores"], temperature)


def _require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(name)
    return torch.device("cuda")


def _check_close(fresh: np.ndarray, saved: np.ndarray, what: str) -> None:
    if fresh.shape != saved.shape or not np.allclose(fresh, saved, rtol=0.02, atol=0.02):
        diff = float(np.max(np.abs(fresh - saved))) if fresh.shape == saved.shape else float("inf")
        raise RuntimeError(f"{what} reload check failed, max abs diff {diff}")


def _load_encoder(artifact: Path, n_labels: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff.train_ettin import EncoderClassifier

    device = _require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(artifact / "encoder")
    encoder = AutoModel.from_pretrained(artifact / "encoder", dtype=torch.float32)
    model = EncoderClassifier(encoder, encoder.config.hidden_size, n_labels).to(device)
    blob = torch.load(artifact / "head.pt", map_location="cpu", weights_only=True)
    model.head.load_state_dict(blob["head"])
    model.eval()
    return model, tokenizer, device


def score_encoder(artifact: Path, items, spec: dict, n_labels: int) -> np.ndarray:
    from bakeoff.train_ettin import _score_items

    model, tokenizer, device = _load_encoder(artifact, n_labels)
    scores, gold, _ = _score_items(model, tokenizer, items, spec, device, spec["eval_batch_size"])
    if not np.array_equal(gold, [item.gold_index for item in items]):
        raise RuntimeError("encoder gold order")
    del model
    import torch

    torch.cuda.empty_cache()
    return scores


def check_decoder(artifact: Path, test_items, labels: list[str], spec: dict) -> None:
    """Reload check on the first 8 saved test logits. Dev scores are not required."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bakeoff.train_ettin_decoder import _score_items

    device = _require_gpu()
    tokenizer = AutoTokenizer.from_pretrained(artifact / "model")
    model = AutoModelForCausalLM.from_pretrained(artifact / "model", dtype=torch.float32)
    model.config.use_cache = False
    model.to(device)
    model.eval()
    scores, _, _ = _score_items(
        model,
        tokenizer,
        test_items[:8],
        labels,
        spec,
        device,
        spec["eval_batch_size"],
        spec["label_chunk_size"],
    )
    saved = np.load(artifact / "test_scores.npz")["scores"][:8]
    _check_close(scores, saved, "decoder")
    del model
    torch.cuda.empty_cache()


def _encoder_spec() -> dict:
    return json.loads((ROOT / "program2.json").read_text())


def _decoder_spec() -> dict:
    return json.loads((ROOT / "program2_decoder.json").read_text())


def write_thresholds() -> Path:
    """Recompute missing Ettin dev logits and write taus. Does not score the test cascade."""
    protocol = load_hybrid_protocol()
    choice = load_protocol()
    from bakeoff.data import load_items

    materialized = load_items(ROOT / "artifacts" / "splits.json", choice)
    out_root = ROOT / "artifacts" / "hybrid" / "dev"
    encoder_spec = _encoder_spec()
    decoder_spec = _decoder_spec()
    rules = {"protocol_id": protocol["protocol_id"], "datasets": {}}
    for dataset in protocol["datasets"]:
        labels = choice["datasets"][dataset]["labels"]
        dev_items = materialized["items"][dataset]["dev"]
        test_items = materialized["items"][dataset]["test"]
        strong_art = ROOT / protocol["strong"]["artifact"].format(dataset=dataset)
        check_decoder(strong_art, test_items, labels, decoder_spec)
        gate_rules = {}
        for gate in protocol["gates"]:
            dest = out_root / dataset / f"{gate['id']}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if gate["kind"] == "zeroshot":
                probs = zeroshot_probs(gate, dataset, "dev", dev_items)
                np.savez_compressed(dest, probs=probs)
            else:
                artifact = ROOT / gate["artifact"].format(dataset=dataset)
                if not dest.exists():
                    fresh = score_encoder(artifact, test_items[:8], encoder_spec, len(labels))
                    _check_close(fresh, np.load(artifact / "test_scores.npz")["scores"][:8], "encoder")
                    scores = score_encoder(artifact, dev_items, encoder_spec, len(labels))
                    np.savez_compressed(
                        dest,
                        scores=scores,
                        gold=np.asarray([item.gold_index for item in dev_items], dtype=np.int64),
                    )
                stored = np.load(dest)
                temperature = _temperature(artifact / "metrics.json")
                probs = apply_temperature(stored["scores"], temperature)
            conf = confidence_of(probs)
            gate_rules[gate["id"]] = [
                fit_threshold(conf, rate) | {"dev_escalated": _realized(conf, rate)}
                for rate in protocol["escalation_rates"]
            ]
        rules["datasets"][dataset] = gate_rules
    path = ROOT / "artifacts" / "hybrid" / "thresholds.json"
    path.write_text(json.dumps(rules, indent=2) + "\n")
    return path


def _realized(confidence: np.ndarray, rate: float) -> int:
    rule = fit_threshold(confidence, rate)
    return int(escalate_mask(confidence, rule).sum())


def item_brier(probs: np.ndarray, gold: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(gold)), np.asarray(gold, dtype=np.int64)] = 1.0
    return ((probs - one_hot) ** 2).sum(axis=1)


def random_control(cheap: np.ndarray, strong: np.ndarray, gold: np.ndarray, called: int, draws: int, seed: int) -> dict:
    """Brier and accuracy when the same number of items is escalated at random."""
    n = len(gold)
    cheap_brier = item_brier(cheap, gold)
    strong_brier = item_brier(strong, gold)
    cheap_right = np.argmax(cheap, axis=1) == gold
    strong_right = np.argmax(strong, axis=1) == gold
    rng = np.random.default_rng(seed)
    briers = np.empty(draws, dtype=np.float64)
    accuracies = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=called, replace=False)] = True
        briers[draw] = np.where(mask, strong_brier, cheap_brier).mean()
        accuracies[draw] = np.where(mask, strong_right, cheap_right).mean()
    return {
        "draws": draws,
        "brier_mean": float(briers.mean()),
        "brier_lo": float(np.percentile(briers, 2.5)),
        "brier_hi": float(np.percentile(briers, 97.5)),
        "accuracy_mean": float(accuracies.mean()),
        "briers": briers,
    }


def oracle_control(cheap: np.ndarray, strong: np.ndarray, gold: np.ndarray, called: int) -> dict:
    """Best possible choice of the same number of items, using the gold labels. A bound, not a method."""
    gain = item_brier(cheap, gold) - item_brier(strong, gold)
    mask = np.zeros(len(gold), dtype=bool)
    if called:
        mask[np.argsort(-gain, kind="stable")[:called]] = True
    probs = hybrid_probs(cheap, strong, mask)
    return {"brier": float(item_brier(probs, gold).mean()), "accuracy": float((np.argmax(probs, axis=1) == gold).mean())}


def _metric_row(probs: np.ndarray, gold: np.ndarray) -> dict:
    scored = score_probabilities(probs, gold)
    keep = ("n", "accuracy", "brier", "log_loss", "ece", "coverage_at_5pct_error", "aurc")
    return {key: scored[key] for key in keep}


def _gate_test_probs(gate: dict, dataset: str, items, choice_labels: list[str]) -> np.ndarray:
    del choice_labels
    if gate["kind"] == "zeroshot":
        return zeroshot_probs(gate, dataset, "test", items)
    artifact = ROOT / gate["artifact"].format(dataset=dataset)
    temperature = _temperature(artifact / "metrics.json")
    return ettin_test_probs(artifact, temperature, items)


def evaluate() -> Path:
    """Apply the frozen taus to test. Refuses to run without thresholds.json."""
    protocol = load_hybrid_protocol()
    threshold_path = ROOT / "artifacts" / "hybrid" / "thresholds.json"
    if not threshold_path.exists():
        raise RuntimeError("thresholds.json is missing; test metrics stay uncomputed")
    thresholds = json.loads(threshold_path.read_text())
    if thresholds["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("thresholds were fit for a different protocol")
    choice = load_protocol()
    from bakeoff.data import load_items

    materialized = load_items(ROOT / "artifacts" / "splits.json", choice)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "role": protocol["role"],
        "datasets": {},
    }
    record_root = ROOT / "artifacts" / "hybrid" / "decisions"
    for dataset, spec in protocol["datasets"].items():
        items = materialized["items"][dataset]["test"]
        gold = np.asarray([item.gold_index for item in items], dtype=np.int64)
        labels = choice["datasets"][dataset]["labels"]
        strong_art = ROOT / protocol["strong"]["artifact"].format(dataset=dataset)
        strong = ettin_test_probs(strong_art, _temperature(strong_art / "metrics.json"), items)
        strong_metrics = _metric_row(strong, gold)
        gates = {}
        for gate in protocol["gates"]:
            cheap = _gate_test_probs(gate, dataset, items, labels)
            if cheap.shape != strong.shape:
                raise RuntimeError(f"{gate['id']} {cheap.shape} != {strong.shape}")
            rules = thresholds["datasets"][dataset][gate["id"]]
            by_rate = {float(rule["rate"]): rule for rule in rules}
            rates = []
            masks = {}
            for rate in protocol["escalation_rates"]:
                rule = by_rate[float(rate)]
                mask = escalate_mask(confidence_of(cheap), rule)
                probs = hybrid_probs(cheap, strong, mask)
                row = _metric_row(probs, gold)
                called = int(mask.sum())
                row.update(
                    {
                        "rate": float(rate),
                        "tau": rule["tau"],
                        "mode": rule["mode"],
                        "strong_calls": called,
                        "strong_call_fraction": called / len(items),
                        "strong_calls_saved": 1.0 - called / len(items),
                        "score_drop": row["brier"] - strong_metrics["brier"],
                    }
                )
                control = protocol["controls"]
                random = random_control(cheap, strong, gold, called, control["random_draws"], control["random_seed"])
                random_briers = random.pop("briers")
                row["random"] = random
                row["brier_gain_over_random"] = random["brier_mean"] - row["brier"]
                row["random_draws_at_or_below_observed"] = float(np.mean(random_briers <= row["brier"]))
                row["oracle"] = oracle_control(cheap, strong, gold, called)
                rates.append(row)
                masks[float(rate)] = mask
            record = record_root / dataset
            record.mkdir(parents=True, exist_ok=True)
            path = record / f"{gate['id']}.jsonl"
            with path.open("w") as handle:
                cheap_choice = np.argmax(cheap, axis=1)
                strong_choice = np.argmax(strong, axis=1)
                conf = confidence_of(cheap)
                for index, item in enumerate(items):
                    escalated = [rate for rate, mask in masks.items() if bool(mask[index])]
                    handle.write(
                        json.dumps(
                            {
                                "item_id": item.item_id,
                                "gold_index": int(item.gold_index),
                                "cheap_confidence": float(conf[index]),
                                "cheap_choice": int(cheap_choice[index]),
                                "strong_choice": int(strong_choice[index]),
                                "escalated_rates": escalated,
                            }
                        )
                        + "\n"
                    )
            gates[gate["id"]] = {
                "agreement": agreement_table(cheap, strong),
                "rates": rates,
            }
        summary["datasets"][dataset] = {
            "role": spec["role"],
            "n": int(len(items)),
            "strong": strong_metrics,
            "gates": gates,
        }
    path = ROOT / "artifacts" / "hybrid" / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    write_plots(summary)
    return path


def write_plots(summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STORE_MEDIA.mkdir(parents=True, exist_ok=True)
    colors = {
        "ettin-encoder-log_loss": "#1f4e79",
        "ettin-encoder-brier": "#5b8fb8",
        "deberta-v3-large-zeroshot-v2.0-c": "#c47b2b",
        "gliclass-base-v3.0": "#2f6f4e",
        "qwen3-0.6b": "#8c3a4b",
    }
    short = {
        "ettin-encoder-log_loss": "Encoder, log loss",
        "ettin-encoder-brier": "Encoder, Brier",
        "deberta-v3-large-zeroshot-v2.0-c": "DeBERTa -c",
        "gliclass-base-v3.0": "GLiClass",
        "qwen3-0.6b": "Qwen3-0.6B",
    }
    titles = {"banking77": "Banking77", "sst2": "SST-2"}
    for dataset, block in summary["datasets"].items():
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        strong_brier = block["strong"]["brier"]
        axis.axhline(strong_brier, color="#222222", linewidth=1.0, linestyle="--", label="Always strong")
        for gate_id, gate in block["gates"].items():
            xs = [row["strong_call_fraction"] for row in gate["rates"]]
            ys = [row["brier"] for row in gate["rates"]]
            axis.plot(xs, ys, marker="o", color=colors[gate_id], label=short[gate_id])
            axis.plot(xs, [row["random"]["brier_mean"] for row in gate["rates"]], color=colors[gate_id], linewidth=0.8, linestyle=":")
        axis.set_xlabel("Fraction of items sent to the Ettin decoder")
        axis.set_ylabel("Brier")
        axis.set_title(titles.get(dataset, dataset))
        axis.set_xlim(-0.02, 1.02)
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(STORE_MEDIA / f"hybrid-brier-{dataset}.png", dpi=140)
        plt.close(figure)
    block = summary["datasets"]["banking77"]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(len(BINS))
    width = 0.15
    for offset, (gate_id, gate) in enumerate(block["gates"].items()):
        heights = [bin_row["agreement"] if bin_row["agreement"] is not None else 0.0 for bin_row in gate["agreement"]["bins"]]
        axis.bar(positions + (offset - 2) * width, heights, width=width, color=colors[gate_id], label=short[gate_id])
    axis.set_xticks(positions)
    axis.set_xticklabels(["[0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1]"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Argmax agreement with the decoder")
    axis.set_xlabel("Cheap-model confidence")
    axis.set_title("Banking77")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(STORE_MEDIA / "hybrid-agreement-banking77.png", dpi=140)
    plt.close(figure)


def main() -> None:
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "thresholds":
        path = write_thresholds()
        print(f"wrote {path}", flush=True)
        return
    if command == "evaluate":
        path = evaluate()
        print(f"wrote {path}", flush=True)
        return
    raise SystemExit("usage: python -m bakeoff.hybrid thresholds|evaluate")


if __name__ == "__main__":
    main()

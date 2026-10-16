"""Paired tests between the 150M Ettin encoder and decoder at their dev-selected learning rates.

Reads the saved test scores of the size sweep and of the original runs. Post hoc: written after the size
sweep's accuracies were seen, to put intervals on a comparison FINDINGS already states.
"""

from __future__ import annotations

import json
from math import comb

import numpy as np

from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT

SWEEP = ROOT / "artifacts" / "size-sweep"
PAIRS = {
    "banking77": {
        "encoder_1e-4": SWEEP / "encoder-150m" / "lr-0.0001" / "banking77",
        "decoder_1e-4": SWEEP / "decoder-150m" / "lr-0.0001" / "banking77",
        "decoder_2e-5": ROOT / "artifacts" / "program2-decoder" / "banking77-log_loss",
        "encoder_2e-5": ROOT / "artifacts" / "program2" / "banking77-log_loss",
    },
    "massive_en": {
        "encoder_1e-4": SWEEP / "encoder-150m" / "lr-0.0001" / "massive_en",
        "decoder_1e-4": SWEEP / "decoder-150m" / "lr-0.0001" / "massive_en",
        "decoder_2e-5": ROOT / "artifacts" / "primary" / "ettin-decoder",
        "encoder_2e-5": ROOT / "artifacts" / "primary" / "ettin-encoder",
    },
    "sst2": {
        "encoder_1e-4": SWEEP / "encoder-150m" / "lr-0.0001" / "sst2",
        "decoder_2e-5": ROOT / "artifacts" / "program2-decoder" / "sst2-log_loss",
        "encoder_2e-5": ROOT / "artifacts" / "program2" / "sst2-log_loss",
    },
}


def _load(folder):
    blob = np.load(folder / "test_scores.npz")
    metrics = json.loads((folder / ("sweep.json" if (folder / "sweep.json").exists() else "metrics.json")).read_text())
    probs = apply_temperature(blob["scores"], metrics["temperature"]["temperature"])
    gold = blob["gold"]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(gold)), gold] = 1.0
    return probs.argmax(axis=1) == gold, ((probs - one_hot) ** 2).sum(axis=1), gold


def _mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = min(1.0, 2.0 * sum(comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2.0**n) if n else 1.0
    return {"only_first_right": only_a, "only_second_right": only_b, "exact_p": p}


def main() -> None:
    rng = np.random.default_rng(20261013)
    out = {}
    for dataset, folders in PAIRS.items():
        loaded = {name: _load(folder) for name, folder in folders.items()}
        comparisons = [("encoder_1e-4", name) for name in loaded if name != "encoder_1e-4"]
        out[dataset] = {"n": int(len(loaded["encoder_1e-4"][0])), "accuracy": {name: float(v[0].mean()) for name, v in loaded.items()}, "brier": {name: float(v[1].mean()) for name, v in loaded.items()}, "pairs": {}}
        for first, second in comparisons:
            a, b = loaded[first], loaded[second]
            if not np.array_equal(a[2], b[2]):
                raise RuntimeError(f"{dataset}: {first} and {second} were scored on different items")
            n = len(a[0])
            picks = rng.integers(0, n, size=(5000, n))
            acc = (a[0][picks].mean(axis=1) - b[0][picks].mean(axis=1)) * 100
            brier = a[1][picks].mean(axis=1) - b[1][picks].mean(axis=1)
            row = {
                "accuracy_difference_points": float((a[0].mean() - b[0].mean()) * 100),
                "accuracy_difference_ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
                "brier_difference": float(a[1].mean() - b[1].mean()),
                "brier_difference_ci": [float(np.percentile(brier, 2.5)), float(np.percentile(brier, 97.5))],
                "mcnemar": _mcnemar(a[0], b[0]),
            }
            out[dataset]["pairs"][f"{first} minus {second}"] = row
            print(
                f"{dataset:11s} {first} minus {second}: accuracy {row['accuracy_difference_points']:+.1f} [{row['accuracy_difference_ci'][0]:+.1f}, {row['accuracy_difference_ci'][1]:+.1f}] "
                f"Brier {row['brier_difference']:+.3f} [{row['brier_difference_ci'][0]:+.3f}, {row['brier_difference_ci'][1]:+.3f}] McNemar {row['mcnemar']['only_first_right']}/{row['mcnemar']['only_second_right']} p={row['mcnemar']['exact_p']:.3f}"
            )
    (SWEEP / "paired_tuned.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

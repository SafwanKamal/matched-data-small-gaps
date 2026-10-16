"""Choice metrics for the frozen Program 1 protocol.

Numeric constants here are checked against protocol.json by the unit test.
"""

from __future__ import annotations

import numpy as np

N_BINS = 15
SELECTIVE_ERROR = 0.05
PROB_FLOOR = 1e-12
SUM_TOL = 1e-4
NEG_TOL = 1e-6


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    return softmax(np.asarray(scores, dtype=np.float64) / temperature)


def schema_valid(probs: np.ndarray, n_labels: int) -> bool:
    values = np.asarray(probs, dtype=np.float64)
    if values.shape != (n_labels,):
        return False
    if not np.isfinite(values).all():
        return False
    if np.any(values < -NEG_TOL):
        return False
    if abs(float(values.sum()) - 1.0) > SUM_TOL:
        return False
    return True


def accuracy(probs: np.ndarray, gold: np.ndarray) -> float:
    pred = np.argmax(np.asarray(probs, dtype=np.float64), axis=1)
    return float(np.mean(pred == np.asarray(gold, dtype=np.int64)))


def multiclass_brier(probs: np.ndarray, gold: np.ndarray) -> float:
    """Mean over items of the sum of squared errors across classes."""
    values = np.asarray(probs, dtype=np.float64)
    one_hot = np.zeros_like(values)
    gold = np.asarray(gold, dtype=np.int64)
    one_hot[np.arange(len(gold)), gold] = 1.0
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def log_loss(probs: np.ndarray, gold: np.ndarray) -> float:
    """Mean natural-log loss. Gold probability is clipped to [1e-12, 1]."""
    values = np.asarray(probs, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    chosen = np.clip(values[np.arange(len(gold)), gold], PROB_FLOOR, 1.0)
    return float(np.mean(-np.log(chosen)))


def expected_calibration_error(probs: np.ndarray, gold: np.ndarray) -> dict:
    """Top-label ECE with 15 equal-width bins on [0, 1].

    Bin i for i < 14 is [i/15, (i+1)/15). The last bin is [14/15, 1].
    Empty bins are omitted from the sum.
    """
    values = np.asarray(probs, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    confidence = values.max(axis=1)
    correct = (np.argmax(values, axis=1) == gold).astype(np.float64)
    n = len(gold)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    bins = []
    ece = 0.0
    for i in range(N_BINS):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < N_BINS - 1:
            mask = (confidence >= lo) & (confidence < hi)
        else:
            mask = (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        row = {"bin": i, "lo": lo, "hi": hi, "count": count, "accuracy": None, "confidence": None}
        if count:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            ece += (count / n) * abs(acc - conf)
            row["accuracy"] = acc
            row["confidence"] = conf
        bins.append(row)
    top = bins[-1]
    return {
        "ece": float(ece),
        "bins": bins,
        "n": n,
        "top_bin_mass": float(top["count"] / n) if n else None,
    }


def selective_stats(probs: np.ndarray, gold: np.ndarray, max_error: float = SELECTIVE_ERROR) -> dict:
    """Coverage at a fixed error and the area under the risk-coverage curve.

    Items are ranked by top-label probability, highest first. Ties keep the
    original order. Coverage at 5% error is the largest prefix whose error
    rate is at most 0.05. If no prefix qualifies, coverage is 0.
    AURC is the trapezoid of risk against coverage k/n for k = 1..n.
    """
    values = np.asarray(probs, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    n = int(len(gold))
    if n == 0:
        return {
            "coverage_at_5pct_error": None,
            "selected_n": 0,
            "confidence_threshold": None,
            "aurc": None,
            "coverage": [],
            "risk": [],
        }
    confidence = values.max(axis=1)
    correct = np.argmax(values, axis=1) == gold
    order = np.argsort(-confidence, kind="mergesort")
    errors = (~correct[order]).astype(np.float64)
    ks = np.arange(1, n + 1, dtype=np.float64)
    risk = np.cumsum(errors) / ks
    coverage = ks / n
    ok = np.flatnonzero(risk <= max_error)
    if len(ok) == 0:
        selected = 0
        cov = 0.0
        threshold = None
    else:
        selected = int(ok[-1] + 1)
        cov = float(selected / n)
        threshold = float(confidence[order][selected - 1])
    return {
        "coverage_at_5pct_error": cov,
        "selected_n": selected,
        "confidence_threshold": threshold,
        "aurc": float(np.trapezoid(risk, coverage)),
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }


def score_probabilities(probs: np.ndarray, gold: np.ndarray) -> dict:
    ece = expected_calibration_error(probs, gold)
    selective = selective_stats(probs, gold)
    return {
        "n": int(len(gold)),
        "accuracy": accuracy(probs, gold),
        "brier": multiclass_brier(probs, gold),
        "log_loss": log_loss(probs, gold),
        "ece": ece["ece"],
        "top_bin_mass": ece["top_bin_mass"],
        "ece_bins": ece["bins"],
        "coverage_at_5pct_error": selective["coverage_at_5pct_error"],
        "selective_selected_n": selective["selected_n"],
        "selective_confidence_threshold": selective["confidence_threshold"],
        "aurc": selective["aurc"],
        "risk_coverage": {"coverage": selective["coverage"], "risk": selective["risk"]},
    }

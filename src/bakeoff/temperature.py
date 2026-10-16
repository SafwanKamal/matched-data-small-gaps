"""One-parameter temperature scaling fit on a development split."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from bakeoff.metrics import PROB_FLOOR, apply_temperature

TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
TEMPERATURE_XATOL = 1e-4


def mean_nll(scores: np.ndarray, gold: np.ndarray, temperature: float) -> float:
    probs = apply_temperature(scores, temperature)
    gold = np.asarray(gold, dtype=np.int64)
    chosen = np.clip(probs[np.arange(len(gold)), gold], PROB_FLOOR, 1.0)
    return float(np.mean(-np.log(chosen)))


def fit_temperature(scores: np.ndarray, gold: np.ndarray) -> dict:
    """Minimize mean NLL on dev. Bounds are part of the protocol."""
    scores = np.asarray(scores, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    if scores.ndim != 2 or len(scores) != len(gold) or len(gold) == 0:
        raise ValueError("scores and gold must be a non-empty aligned pair")
    if not np.isfinite(scores).all():
        raise ValueError("temperature fit received a non-finite score")

    result = minimize_scalar(
        lambda temperature: mean_nll(scores, gold, float(temperature)),
        bounds=(TEMPERATURE_MIN, TEMPERATURE_MAX),
        method="bounded",
        options={"xatol": TEMPERATURE_XATOL},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"temperature fit failed: {result.message}")
    temperature = float(result.x)
    return {
        "temperature": temperature,
        "dev_nll": float(result.fun),
        "dev_nll_at_1": mean_nll(scores, gold, 1.0),
        "n": int(len(gold)),
        "bounds": [TEMPERATURE_MIN, TEMPERATURE_MAX],
        "at_lower_bound": temperature <= TEMPERATURE_MIN + 2 * TEMPERATURE_XATOL,
        "at_upper_bound": temperature >= TEMPERATURE_MAX - 2 * TEMPERATURE_XATOL,
    }

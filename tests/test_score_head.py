"""Ordinal score definitions. No model and no SST-5 rows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bakeoff.train_score import coral_distribution, expectation_mae, ranked_probability_score  # noqa: E402


def test_ranked_probability_score_is_zero_when_perfect() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gold = np.array([0, 2])
    assert ranked_probability_score(probs, gold) == 0.0
    assert expectation_mae(probs, gold) == 0.0


def test_coral_monotone_sigmoids_need_no_repair() -> None:
    logits = torch.tensor([[2.0, 0.0, -2.0]])
    probs, repaired = coral_distribution(logits)
    assert not bool(repaired.any())
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(probs >= 0)

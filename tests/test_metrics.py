"""Unit checks for the frozen metric definitions. No model and no test labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bakeoff.metrics import (  # noqa: E402
    N_BINS,
    PROB_FLOOR,
    SELECTIVE_ERROR,
    accuracy,
    expected_calibration_error,
    log_loss,
    multiclass_brier,
    schema_valid,
    score_probabilities,
    selective_stats,
    softmax,
)
from bakeoff.temperature import (  # noqa: E402
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    fit_temperature,
    mean_nll,
)


def test_perfect_predictions() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    gold = np.array([0, 1])
    scored = score_probabilities(probs, gold)
    assert scored["accuracy"] == 1.0
    assert scored["brier"] == 0.0
    assert scored["log_loss"] == 0.0
    assert scored["ece"] == 0.0
    assert scored["coverage_at_5pct_error"] == 1.0


def test_brier_and_log_loss_and_ece() -> None:
    probs = np.array([[0.8, 0.2]])
    gold = np.array([0])
    assert abs(multiclass_brier(probs, gold) - 0.08) < 1e-12
    assert abs(log_loss(probs, gold) - (-np.log(0.8))) < 1e-12
    ece = expected_calibration_error(probs, gold)
    assert abs(ece["ece"] - 0.2) < 1e-12
    occupied = [row for row in ece["bins"] if row["count"]]
    assert len(occupied) == 1
    assert occupied[0]["bin"] == 12
    assert occupied[0]["lo"] == 0.8


def test_log_loss_floor() -> None:
    probs = np.array([[0.0, 1.0]])
    value = log_loss(probs, np.array([0]))
    assert np.isfinite(value)
    assert abs(value - (-np.log(PROB_FLOOR))) < 1e-9


def test_schema_and_tie() -> None:
    assert schema_valid(np.array([0.5, 0.5]), 2)
    assert not schema_valid(np.array([0.5, 0.6]), 2)
    assert not schema_valid(np.array([-0.1, 1.1]), 2)
    assert not schema_valid(np.array([np.nan, 1.0]), 2)
    assert int(np.argmax(np.array([0.5, 0.5]))) == 0
    assert accuracy(np.array([[0.5, 0.5]]), np.array([0])) == 1.0
    assert accuracy(np.array([[0.5, 0.5]]), np.array([1])) == 0.0


def test_selective_coverage() -> None:
    probs = np.zeros((100, 2))
    gold = np.zeros(100, dtype=np.int64)
    for index in range(90):
        confidence = 0.99 - index * 0.001
        probs[index] = [confidence, 1.0 - confidence]
    for offset, index in enumerate(range(90, 100)):
        confidence = 0.80 - offset * 0.01
        probs[index] = [1.0 - confidence, confidence]
    stats = selective_stats(probs, gold, max_error=SELECTIVE_ERROR)
    assert stats["selected_n"] == 94
    assert abs(stats["coverage_at_5pct_error"] - 0.94) < 1e-12


def test_temperature_recovers_scale() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2000, 3))
    probabilities = softmax(logits)
    gold = np.array([rng.choice(3, p=row) for row in probabilities])
    scores = logits * 2.0
    fitted = fit_temperature(scores, gold)
    assert TEMPERATURE_MIN < fitted["temperature"] < TEMPERATURE_MAX
    assert abs(fitted["temperature"] - 2.0) < 0.25
    assert fitted["dev_nll"] < mean_nll(scores, gold, 1.0)


def test_protocol_file_matches_code() -> None:
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["protocol_id"] == "program-1-choice-2026-10-07"
    assert protocol["seed"] == 20261007
    assert protocol["ece"]["bins"] == N_BINS == 15
    assert protocol["selective"]["max_error"] == SELECTIVE_ERROR == 0.05
    assert protocol["log_loss"]["floor"] == PROB_FLOOR
    assert protocol["temperature"]["bounds"] == [TEMPERATURE_MIN, TEMPERATURE_MAX]
    sst2 = protocol["datasets"]["sst2"]
    ag = protocol["datasets"]["ag_news"]
    banking = protocol["datasets"]["banking77"]
    assert sst2["test_n"] == 872 and sst2["dev_n"] == 400
    assert sst2["labels"] == ["negative", "positive"]
    assert sst2["test_is_full_split"] is True
    assert ag["test_n"] == 1000 and ag["dev_n"] == 400
    assert ag["labels"] == ["World", "Sports", "Business", "Sci/Tech"]
    assert banking["test_n"] == 500 and banking["dev_n"] == 400
    assert len(banking["labels"]) == 77
    assert banking["labels"][0] == "activate my card"
    deberta = protocol["systems"]["deberta-v3-large-zeroshot-v2.0-c"]
    gliclass = protocol["systems"]["gliclass-base-v3.0"]
    qwen = protocol["systems"]["qwen3-0.6b"]
    assert deberta["multi_label"] is False
    assert deberta["hypothesis_template"] == "This example is {}."
    assert gliclass["classification"] == "single-label"
    assert gliclass["max_labels_per_forward"] == 25
    assert qwen["enable_thinking"] is False
    assert qwen["length_normalization"] == "mean_token_logprob"
    assert qwen["generate"] is False
    assert qwen["label_batch_size"] == 4


def main() -> None:
    test_perfect_predictions()
    test_brier_and_log_loss_and_ece()
    test_log_loss_floor()
    test_schema_and_tie()
    test_selective_coverage()
    test_temperature_recovers_scale()
    test_protocol_file_matches_code()
    print("metric tests passed")


if __name__ == "__main__":
    main()

"""Threshold and cascade rules. No GPU and no test labels."""

from __future__ import annotations

import numpy as np

from bakeoff.hybrid import agreement_table, escalate_mask, fit_threshold, hybrid_probs


def test_threshold_endpoints_and_order_statistic() -> None:
    confidence = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.8, 0.5, 0.7, 0.6, 0.55])
    none = fit_threshold(confidence, 0.0)
    assert none["mode"] == "none" and none["k"] == 0
    assert not escalate_mask(confidence, none).any()
    every = fit_threshold(confidence, 1.0)
    assert every["mode"] == "all" and every["k"] == 10
    assert escalate_mask(confidence, every).all()
    rule = fit_threshold(confidence, 0.25)
    assert rule["k"] == 2
    assert rule["tau"] == 0.2
    mask = escalate_mask(confidence, rule)
    assert mask.tolist() == [True, False, True, False, False, False, False, False, False, False]


def test_ties_at_tau_all_escalate() -> None:
    confidence = np.array([0.2, 0.2, 0.2, 0.9])
    rule = fit_threshold(confidence, 0.25)
    assert rule["k"] == 1
    assert rule["tau"] == 0.2
    assert int(escalate_mask(confidence, rule).sum()) == 3


def test_hybrid_copies_the_selected_row() -> None:
    cheap = np.array([[0.8, 0.2], [0.3, 0.7]])
    strong = np.array([[0.1, 0.9], [0.6, 0.4]])
    mixed = hybrid_probs(cheap, strong, np.array([False, True]))
    assert np.allclose(mixed[0], cheap[0])
    assert np.allclose(mixed[1], strong[1])


def test_agreement_bins_are_half_open_until_the_last() -> None:
    cheap = np.array([[0.2, 0.8], [1.0, 0.0], [0.6, 0.4]])
    strong = np.array([[0.1, 0.9], [0.0, 1.0], [0.6, 0.4]])
    table = agreement_table(cheap, strong)
    assert abs(table["agreement"] - (2 / 3)) < 1e-12
    filled = [row for row in table["bins"] if row["n"]]
    assert [(row["lo"], row["n"]) for row in filled] == [(0.6, 1), (0.8, 2)]


if __name__ == "__main__":
    test_threshold_endpoints_and_order_statistic()
    test_ties_at_tau_all_escalate()
    test_hybrid_copies_the_selected_row()
    test_agreement_bins_are_half_open_until_the_last()
    print("ok")

"""Plots from per-item decisions. Numbers are recomputed with the metric functions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bakeoff.metrics import expected_calibration_error, selective_stats

COLORS = {
    "deberta-v3-large-zeroshot-v2.0-c": "#1f4e79",
    "gliclass-base-v3.0": "#c47b2b",
    "qwen3-0.6b": "#2f6f4e",
}
SHORT = {
    "deberta-v3-large-zeroshot-v2.0-c": "DeBERTa-v3-large -c",
    "gliclass-base-v3.0": "GLiClass base v3",
    "qwen3-0.6b": "Qwen3-0.6B",
}
DATASETS = ["sst2", "ag_news", "banking77"]
DATASET_TITLES = {"sst2": "SST-2", "ag_news": "AG News", "banking77": "Banking77"}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _valid(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    kept = [row for row in rows if row.get("schema_valid_scaled")]
    probs = np.asarray([row["probs_scaled"] for row in kept], dtype=np.float64)
    gold = np.asarray([row["gold_index"] for row in kept], dtype=np.int64)
    return probs, gold


def write_plots(decisions: Path, destinations: list[Path], protocol: dict) -> list[str]:
    systems = list(protocol["systems"])
    reliability = plt.figure(figsize=(11.2, 8.4))
    risk_fig = plt.figure(figsize=(11.2, 3.6))
    for row_index, dataset in enumerate(DATASETS):
        risk_ax = risk_fig.add_subplot(1, 3, row_index + 1)
        for col_index, system in enumerate(systems):
            path = decisions / f"{system}__{dataset}__test.jsonl"
            axis = reliability.add_subplot(3, 3, row_index * 3 + col_index + 1)
            axis.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title(f"{DATASET_TITLES[dataset]} · {SHORT[system]}", fontsize=9)
            if row_index == 2:
                axis.set_xlabel("Top-label confidence")
            if col_index == 0:
                axis.set_ylabel("Accuracy in bin")
            if not path.exists():
                axis.text(0.5, 0.5, "no run", ha="center", va="center")
                continue
            probs, gold = _valid(_read_jsonl(path))
            if len(gold) == 0:
                axis.text(0.5, 0.5, "no valid items", ha="center", va="center")
                continue
            ece = expected_calibration_error(probs, gold)
            xs, ys, sizes = [], [], []
            for bin_row in ece["bins"]:
                if bin_row["count"]:
                    xs.append(bin_row["confidence"])
                    ys.append(bin_row["accuracy"])
                    sizes.append(18 + 40 * bin_row["count"] / len(gold))
            axis.scatter(xs, ys, s=sizes, color=COLORS[system], zorder=3)
            selective = selective_stats(probs, gold)
            risk_ax.plot(
                selective["coverage"],
                selective["risk"],
                color=COLORS[system],
                label=SHORT[system],
                linewidth=1.6,
            )
        risk_ax.axhline(0.05, color="#888888", linewidth=0.8, linestyle="--")
        risk_ax.set_xlim(0, 1)
        risk_ax.set_ylim(0, 1)
        risk_ax.set_title(DATASET_TITLES[dataset])
        risk_ax.set_xlabel("Coverage")
        if row_index == 0:
            risk_ax.set_ylabel("Risk")
            risk_ax.legend(fontsize=8, frameon=False)
    reliability.tight_layout()
    risk_fig.tight_layout()

    latency_fig, latency_ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.25
    positions = np.arange(len(DATASETS))
    for offset, system in enumerate(systems):
        p50, p95 = [], []
        for dataset in DATASETS:
            path = decisions / f"{system}__{dataset}__test.jsonl"
            if not path.exists():
                p50.append(0.0)
                p95.append(0.0)
                continue
            values = [row["latency_ms"] for row in _read_jsonl(path) if row.get("latency_ms") is not None]
            if not values:
                p50.append(0.0)
                p95.append(0.0)
                continue
            array = np.asarray(values, dtype=np.float64)
            p50.append(float(np.percentile(array, 50, method="linear")))
            p95.append(float(np.percentile(array, 95, method="linear")))
        latency_ax.bar(
            positions + (offset - 1) * width,
            p50,
            width=width,
            color=COLORS[system],
            label=f"{SHORT[system]} p50",
        )
        latency_ax.scatter(
            positions + (offset - 1) * width,
            p95,
            color="#222222",
            s=16,
            zorder=3,
        )
    latency_ax.set_xticks(positions)
    latency_ax.set_xticklabels([DATASET_TITLES[name] for name in DATASETS])
    latency_ax.set_ylabel("Batch-1 latency (ms)")
    latency_ax.set_title("p50 bars, p95 points")
    latency_ax.legend(fontsize=8, frameon=False)
    latency_fig.tight_layout()

    written = []
    figures = {
        "program-1-reliability.png": reliability,
        "program-1-risk-coverage.png": risk_fig,
        "program-1-latency.png": latency_fig,
    }
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            path = destination / name
            figure.savefig(path, dpi=140)
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"plot was not written: {path}")
            written.append(str(path))
    plt.close(reliability)
    plt.close(risk_fig)
    plt.close(latency_fig)
    return written

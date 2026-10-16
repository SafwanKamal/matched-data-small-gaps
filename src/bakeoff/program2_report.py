"""Compare Ettin finetunes with the Program 1 zero-shot table. Numbers come from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from bakeoff.protocol import ROOT

STORE_DOC = ROOT / "docs" / "program-2-encoder.md"
ZERO_SHOT = {
    "deberta-v3-large-zeroshot-v2.0-c": "DeBERTa-v3-large -c",
    "gliclass-base-v3.0": "GLiClass base v3",
    "qwen3-0.6b": "Qwen3-0.6B",
}
DATASETS = {"banking77": "Banking77", "sst2": "SST-2"}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _append_run(lines: list[str], title: str, system: str, run: dict) -> None:
    scaled = run["scaled"]
    lines.append(
        f"| {title} | {system} | {run['loss']} | {scaled.get('n')} | {_fmt(scaled.get('accuracy'))} | "
        f"{_fmt(scaled.get('brier'))} | {_fmt(scaled.get('log_loss'))} | {_fmt(scaled.get('ece'))} | "
        f"{_fmt(scaled.get('coverage_at_5pct_error'))} | {_fmt((run.get('latency') or {}).get('p50_ms'), 2)} | "
        f"{_fmt(run['temperature']['temperature'], 3)} |"
    )


def _plot(summary: dict, decoder: dict | None, zero: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for axis, (dataset, title) in zip(axes, DATASETS.items(), strict=True):
        names = []
        values = []
        for name, label in ZERO_SHOT.items():
            cell = zero["systems"].get(name, {}).get("datasets", {}).get(dataset)
            if not cell:
                continue
            names.append(label)
            values.append(cell["scaled"]["brier"])
        for run in summary["runs"]:
            if run["dataset"] == dataset:
                names.append(f"Ettin enc {run['loss']}")
                values.append(run["scaled"]["brier"])
        if decoder:
            for run in decoder["runs"]:
                if run["dataset"] == dataset:
                    names.append(f"Ettin dec {run['loss']}")
                    values.append(run["scaled"]["brier"])
        axis.barh(range(len(values)), values, color="#3d5a40")
        axis.set_yticks(range(len(values)), names, fontsize=8)
        axis.set_xlabel("Scaled Brier")
        axis.set_title(title)
        axis.invert_yaxis()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_program2_report() -> Path:
    summary = json.loads((ROOT / "artifacts" / "program2" / "summary.json").read_text())
    zero = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    decoder_path = ROOT / "artifacts" / "program2-decoder" / "summary.json"
    decoder = json.loads(decoder_path.read_text()) if decoder_path.exists() else None
    lines = [
        "# Program 2: Ettin encoder and paired decoder",
        "",
        "Same RTX 5070 and the same frozen Program 1 test rows. "
        "`jhu-clsp/ettin-encoder-150m` with a mean-pooled linear head. "
        "The 400 Program 1 dev rows were held out of training and used only to fit temperature. "
        "Three epochs, learning rate 2e-5, no test-set model selection. "
        "Log loss and Brier are separate runs. No GRPO and no RLCD.",
        "",
        "## Test metrics after temperature scaling",
        "",
        "| Dataset | System | Loss | N | Accuracy | Brier | Log loss | ECE | Coverage @ 5% error | p50 ms | T |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, title in DATASETS.items():
        for name, label in ZERO_SHOT.items():
            cell = zero["systems"].get(name, {}).get("datasets", {}).get(dataset)
            if not cell:
                continue
            scaled = cell["scaled"]
            lines.append(
                f"| {title} | {label} | zero-shot | {scaled.get('n')} | {_fmt(scaled.get('accuracy'))} | "
                f"{_fmt(scaled.get('brier'))} | {_fmt(scaled.get('log_loss'))} | {_fmt(scaled.get('ece'))} | "
                f"{_fmt(scaled.get('coverage_at_5pct_error'))} | {_fmt(cell['latency'].get('p50_ms'), 2)} | "
                f"{_fmt(cell['temperature']['temperature'], 3)} |"
            )
        for run in summary["runs"]:
            if run["dataset"] == dataset:
                _append_run(lines, title, "Ettin encoder 150M", run)
        if decoder:
            for run in decoder["runs"]:
                if run["dataset"] == dataset:
                    _append_run(lines, title, "Ettin decoder 150M", run)
    lines += [
        "",
        "## Run record",
        "",
        "| Job | Train N | Dev held out | Batch | Peak MiB | Train loss | Brier unscaled | Brier scaled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    record_runs = [("encoder", run) for run in summary["runs"]]
    if decoder:
        record_runs.extend(("decoder", run) for run in decoder["runs"])
    for kind, run in record_runs:
        lines.append(
            f"| {kind} {run['dataset']} {run['loss']} | {run['train_n']} | {run['excluded_dev_n']} | "
            f"{run['batch_size']} | {run['peak_allocated_mib']:.0f} | {_fmt(run['mean_train_loss'])} | "
            f"{_fmt(run['unscaled'].get('brier'))} | {_fmt(run['scaled'].get('brier'))} |"
        )
    failures = list(summary["failures"])
    if decoder:
        lines += [
            "",
            "## Decoder",
            "",
            "Paired `jhu-clsp/ettin-decoder-150m` closed-set readout. "
            "Token ids are concatenated: bos, the text truncated to 128 tokens, a newline, then the label encoded alone. "
            "The score is the mean token log-probability of those label ids, and the simplex is a softmax over labels. "
            "No tokens are generated. Same losses, same dev holdout, and same three epochs as the encoder. "
            "Batch size 8. A batch of 16 ran out of memory on a real Banking77 batch before any test metric.",
            "",
        ]
        if decoder["failures"]:
            failures.extend(decoder["failures"])
    if failures:
        lines += ["", "## Failures", ""]
        for failure in failures:
            tail = failure["traceback"].strip().splitlines()
            lines.append(f"- {failure['job']}: {tail[-1] if tail else 'unknown'}")
    plot_path = STORE_DOC.parent.parent / "media" / "program-2-brier.png"
    _plot(summary, decoder, zero, plot_path)
    lines += ["", f"Scaled Brier plot: `{plot_path}`.", ""]
    STORE_DOC.parent.mkdir(parents=True, exist_ok=True)
    STORE_DOC.write_text("\n".join(lines))
    return STORE_DOC


if __name__ == "__main__":
    print(write_program2_report())

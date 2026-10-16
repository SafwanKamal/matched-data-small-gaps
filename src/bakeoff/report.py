"""Write the results note from metrics.json. It does not recompute scores."""

from __future__ import annotations

from pathlib import Path

SYSTEM_ORDER = [
    "deberta-v3-large-zeroshot-v2.0-c",
    "gliclass-base-v3.0",
    "qwen3-0.6b",
]
SYSTEM_LABEL = {
    "deberta-v3-large-zeroshot-v2.0-c": "DeBERTa-v3-large -c",
    "gliclass-base-v3.0": "GLiClass base v3",
    "qwen3-0.6b": "Qwen3-0.6B",
}
DATASET_ORDER = ["sst2", "ag_news", "banking77"]
DATASET_LABEL = {"sst2": "SST-2", "ag_news": "AG News", "banking77": "Banking77"}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _dataset_cell(payload: dict, system: str, dataset: str) -> dict | None:
    block = payload["systems"].get(system)
    if not block:
        return None
    return block["datasets"].get(dataset)


def write_report(payload: dict, path: Path, plot_paths: list[str], protocol: dict) -> None:
    del protocol  # the frozen file is cited by id; the table is taken from the run payload
    gpu = payload["gpu_before"]
    lines = [
        "# Program 1 bakeoff",
        "",
        "Inference-only Choice comparison on one GPU. No rationale, no cascade, and no training. "
        "Per-item decisions and the frozen protocol are in the repository. "
        "Table entries are rounded from `artifacts/metrics.json`.",
        "",
        "## GPU",
        "",
        f"`nvidia-smi` name: **{gpu['nvidia_smi_name']}**. "
        f"PyTorch device: **{gpu['torch_device_name']}**. "
        f"Driver {gpu['driver_version']}, PyTorch CUDA {gpu['torch_cuda']}. "
        f"Total memory {gpu['memory_total_mib']:.0f} MiB. "
        f"Before the first model load, nvidia-smi reported {gpu['memory_used_mib']:.0f} MiB used "
        f"and {gpu['memory_free_mib']:.0f} MiB free. The used figure includes the desktop compositor. "
        "Each system asserted that its parameters were on CUDA.",
        "",
        "| System | Dtype | Weights allocated (MiB) | Peak allocated (MiB) | nvidia-smi used after load (MiB) | nvidia-smi used after run (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in SYSTEM_ORDER:
        system = payload["systems"].get(name)
        if not system or "weight_mib" not in system:
            lines.append(f"| {SYSTEM_LABEL[name]} | n/a | n/a | n/a | n/a | n/a |")
            continue
        fallback = " (float32 fallback)" if system.get("dtype_fallback") else ""
        lines.append(
            f"| {SYSTEM_LABEL[name]} | {system['dtype']}{fallback} | "
            f"{system['weight_mib']:.0f} | {system['peak_allocated_mib']:.0f} | "
            f"{system['nvidia_smi_after_load']['memory_used_mib']:.0f} | "
            f"{system['nvidia_smi_after_run']['memory_used_mib']:.0f} |"
        )
    lines += [
        "",
        "Peak allocated is `torch.cuda.max_memory_allocated` for that system's forwards, reset after the weights loaded. "
        "nvidia-smi used still includes the desktop process.",
        "",
        "## Sample sizes",
        "",
        "The dev split is a disjoint official train subsample and was used only to fit one temperature. "
        "The test rows were fixed before any test metric was computed.",
        "",
        "| Dataset | Test source | Test N | Dev N | Labels | Source sizes (dev pool / test pool) | Exact text overlap |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    sources = {
        "sst2": "GLUE validation, every labeled row",
        "ag_news": "official test, fixed subsample",
        "banking77": "official test, fixed subsample",
    }
    for name in DATASET_ORDER:
        sample = payload["samples"][name]
        source_n = sample["source_n"]
        lines.append(
            f"| {DATASET_LABEL[name]} | {sources[name]} | {sample['test_n']} | {sample['dev_n']} | "
            f"{sample['n_labels']} | {source_n['dev']} / {source_n['test']} | "
            f"{sample['exact_text_overlap_dev_test']} |"
        )
    lines += [
        "",
        "SST-2 uses GLUE validation because the GLUE test split has no gold labels. "
        "Banking77 is `legacy-datasets/banking77` (10003 train / 3080 test). "
        "`datasets` 5.1.0 refuses the original `PolyAI/banking77` loading script, so the run uses that parquet mirror of the same split.",
        "",
        "## Metric table",
        "",
        "Primary numbers use the dev-fitted temperature. Accuracy is argmax, and ties take the lowest label index. "
        "Brier is the mean over items of the sum of squared errors across classes. "
        "Log loss is mean natural log, with the gold probability clipped at 1e-12. "
        "ECE is 15 equal-width bins of top-label confidence on [0, 1]; the last bin includes 1 and empty bins are omitted. "
        "Selective coverage is the largest confidence-ranked prefix whose error rate is at most 5%. "
        "Schema validity is the fraction of test items whose scaled distribution is a finite simplex on the frozen labels. "
        "Latency is batch-1 milliseconds per text on this RTX 5070. Label scoring happens inside that call. "
        "Twenty dev items are warmed up and excluded. p50 and p95 are linear percentiles. Output tokens are 0.",
        "",
        "| Dataset | System | N | Accuracy | Brier | Log loss | ECE | Coverage @ 5% error | Schema validity | p50 ms | p95 ms | T |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASET_ORDER:
        for name in SYSTEM_ORDER:
            cell = _dataset_cell(payload, name, dataset)
            if cell is None:
                lines.append(
                    f"| {DATASET_LABEL[dataset]} | {SYSTEM_LABEL[name]} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            scaled = cell["scaled"]
            lines.append(
                "| {dataset} | {system} | {n} | {acc} | {brier} | {loss} | {ece} | {cov} | {schema} | {p50} | {p95} | {temp} |".format(
                    dataset=DATASET_LABEL[dataset],
                    system=SYSTEM_LABEL[name],
                    n=scaled.get("n", 0),
                    acc=_fmt(scaled.get("accuracy")),
                    brier=_fmt(scaled.get("brier")),
                    loss=_fmt(scaled.get("log_loss")),
                    ece=_fmt(scaled.get("ece")),
                    cov=_fmt(scaled.get("coverage_at_5pct_error")),
                    schema=_fmt(cell.get("schema_validity")),
                    p50=_fmt(cell["latency"].get("p50_ms"), 2),
                    p95=_fmt(cell["latency"].get("p95_ms"), 2),
                    temp=_fmt(cell["temperature"]["temperature"], 3),
                )
            )
    lines += [
        "",
        "Same test items before temperature scaling:",
        "",
        "| Dataset | System | Accuracy | Brier | Log loss | ECE | Coverage @ 5% error | AURC | Top-bin mass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASET_ORDER:
        for name in SYSTEM_ORDER:
            cell = _dataset_cell(payload, name, dataset)
            if cell is None:
                lines.append(f"| {DATASET_LABEL[dataset]} | {SYSTEM_LABEL[name]} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            raw = cell["unscaled"]
            lines.append(
                f"| {DATASET_LABEL[dataset]} | {SYSTEM_LABEL[name]} | {_fmt(raw.get('accuracy'))} | "
                f"{_fmt(raw.get('brier'))} | {_fmt(raw.get('log_loss'))} | {_fmt(raw.get('ece'))} | "
                f"{_fmt(raw.get('coverage_at_5pct_error'))} | {_fmt(raw.get('aurc'))} | {_fmt(raw.get('top_bin_mass'))} |"
            )
    lines += [
        "",
        "Temperature-scaled selective detail. AURC is the trapezoid of the risk-coverage curve. "
        "Top-bin mass is the fraction of items whose confidence falls in the last ECE bin, [14/15, 1].",
        "",
        "| Dataset | System | AURC | Top-bin mass | Selected N | Confidence threshold | Dev NLL | Dev NLL at T=1 | Truncated test items |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASET_ORDER:
        for name in SYSTEM_ORDER:
            cell = _dataset_cell(payload, name, dataset)
            if cell is None:
                lines.append(f"| {DATASET_LABEL[dataset]} | {SYSTEM_LABEL[name]} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            scaled = cell["scaled"]
            fitted = cell["temperature"]
            lines.append(
                f"| {DATASET_LABEL[dataset]} | {SYSTEM_LABEL[name]} | {_fmt(scaled.get('aurc'))} | "
                f"{_fmt(scaled.get('top_bin_mass'))} | {scaled.get('selective_selected_n', 'n/a')} | "
                f"{_fmt(scaled.get('selective_confidence_threshold'))} | {_fmt(fitted.get('dev_nll'))} | "
                f"{_fmt(fitted.get('dev_nll_at_1'))} | {cell.get('n_truncated', 'n/a')} |"
            )
    lines += [
        "",
        "## What the table separates",
        "",
    ]
    lines.extend(_reading(payload))
    lines += [
        "",
        "## Protocol used for this table",
        "",
        f"Protocol id `{payload['protocol_id']}`. SHA-256 of `protocol.json`: `{payload['protocol_sha256']}`. "
        f"Git HEAD at the start of the run: `{payload['git_head']}`. "
        "Libraries: "
        + ", ".join(f"{name} {version}" for name, version in payload.get("libraries", {}).items())
        + ".",
        "",
        "Shared label strings. DeBERTa-v3-large-zeroshot-v2.0-c uses `multi_label` false: a softmax over the entailment logit of `This example is {label}.`, truncation `only_first` at 512. "
        "GLiClass base v3.0 is single-label softmax over the library's label logits, float32, max length 512, no prompt and no examples. "
        "Its config scores at most 25 labels per forward, so Banking77 logits are concatenated from contiguous chunks of 25 and then softmaxed once. "
        "Qwen3-0.6B is a closed-set softmax of length-normalized continuation log-probabilities. "
        "The chat template is called with `enable_thinking` false, which inserts an empty think block as prompt text. `generate` is not called.",
        "",
        "Revisions: "
        + ", ".join(f"`{spec['hf_id']}` `{spec['revision'][:12]}`" for spec in payload_systems_revisions(payload)),
        "",
        "## Plots",
        "",
    ]
    for plot in plot_paths:
        lines.append(f"- `{plot}`")
    if payload["failures"]:
        lines += ["", "## Failures", ""]
        for failure in payload["failures"]:
            stage = failure.get("stage", "")
            where = " / ".join(str(failure.get(key)) for key in ("system", "dataset") if failure.get(key))
            tail = failure.get("traceback", "").strip().splitlines()
            last = tail[-1] if tail else "unknown"
            lines.append(f"- {where} ({stage}): {last}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"results note was not written: {path}")


def payload_systems_revisions(payload: dict) -> list[dict]:
    del payload
    from bakeoff.protocol import load_protocol

    protocol = load_protocol()
    revisions = []
    for spec in protocol["systems"].values():
        revisions.append(spec)
    for spec in protocol["datasets"].values():
        revisions.append({"hf_id": spec["hf_id"], "revision": spec["revision"]})
    return revisions


def _reading(payload: dict) -> list[str]:
    sentences = []
    for dataset in DATASET_ORDER:
        ranked = []
        for name in SYSTEM_ORDER:
            cell = _dataset_cell(payload, name, dataset)
            if cell is None or "brier" not in cell["scaled"]:
                continue
            ranked.append((cell["scaled"]["brier"], cell["scaled"]["accuracy"], cell["latency"]["p50_ms"], name))
        if not ranked:
            sentences.append(f"{DATASET_LABEL[dataset]} did not produce a scored test set.")
            continue
        best_brier = min(ranked, key=lambda item: item[0])
        best_acc = max(ranked, key=lambda item: item[1])
        fastest = min(ranked, key=lambda item: item[2])
        sentences.append(
            f"On {DATASET_LABEL[dataset]}, {SYSTEM_LABEL[best_brier[3]]} has the lowest temperature-scaled Brier "
            f"({best_brier[0]:.4f}). {SYSTEM_LABEL[best_acc[3]]} has the highest accuracy ({best_acc[1]:.4f}). "
            f"{SYSTEM_LABEL[fastest[3]]} has the lowest batch-1 p50 ({fastest[2]:.2f} ms)."
        )
    sentences.append(
        "Schema validity is a property of the simplex, not of the gold label. "
        "A valid row can still be the wrong class. Output tokens stayed at 0, so none of these rows billed a rationale."
    )
    return sentences

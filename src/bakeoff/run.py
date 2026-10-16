"""Run the frozen Program 1 bakeoff. Test metrics are computed after temperature is fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from bakeoff.data import load_items, write_splits
from bakeoff.metrics import apply_temperature, schema_valid, score_probabilities
from bakeoff.plots import write_plots
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.report import write_report
from bakeoff.systems import build_systems
from bakeoff.temperature import fit_temperature

MEDIA_DIR = ROOT / "docs" / "figures"
RESULTS_PATH = ROOT / "docs" / "program-1-bakeoff.md"


def gpu_snapshot() -> dict:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, total, used, free, driver = [part.strip() for part in query.split(",")]
    return {
        "nvidia_smi_name": name,
        "memory_total_mib": float(total),
        "memory_used_mib": float(used),
        "memory_free_mib": float(free),
        "driver_version": driver,
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")


def _score_items(system, items, labels: list[str], timed: bool) -> list[dict]:
    rows = []
    for index, item in enumerate(items, start=1):
        latency_ms = None
        truncated = False
        error = None
        score_list = None
        try:
            if timed:
                torch.cuda.synchronize()
                started = time.perf_counter()
            scores, truncated = system.score(item.text, labels)
            if timed:
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - started) * 1000.0
            if np.isfinite(scores).all():
                score_list = scores.tolist()
            else:
                error = "non-finite scores"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "dataset": None,
                "split": None,
                "system": system.name,
                "item_id": item.item_id,
                "row": item.row,
                "gold_index": item.gold_index,
                "gold_label": item.gold_label,
                "scores": score_list,
                "truncated": bool(truncated),
                "latency_ms": latency_ms,
                "output_tokens": 0,
                "error": error,
            }
        )
        if index % 100 == 0:
            print(f"  {system.name} {index}/{len(items)}", flush=True)
    return rows


def _finalize(rows: list[dict], labels: list[str], temperature: float | None) -> list[dict]:
    for row in rows:
        if row["scores"] is None:
            row["probs_unscaled"] = None
            row["probs_scaled"] = None
            row["schema_valid_unscaled"] = False
            row["schema_valid_scaled"] = False
            row["choice_index_unscaled"] = None
            row["choice_label_unscaled"] = None
            row["choice_index_scaled"] = None
            row["choice_label_scaled"] = None
            continue
        scores = np.asarray(row["scores"], dtype=np.float64)
        unscaled = apply_temperature(scores, 1.0)
        scaled = apply_temperature(scores, temperature) if temperature is not None else None
        row["probs_unscaled"] = unscaled.tolist()
        row["probs_scaled"] = None if scaled is None else scaled.tolist()
        row["schema_valid_unscaled"] = schema_valid(unscaled, len(labels))
        row["schema_valid_scaled"] = False if scaled is None else schema_valid(scaled, len(labels))
        if row["schema_valid_unscaled"]:
            choice = int(np.argmax(unscaled))
            row["choice_index_unscaled"] = choice
            row["choice_label_unscaled"] = labels[choice]
        else:
            row["choice_index_unscaled"] = None
            row["choice_label_unscaled"] = None
        if row["schema_valid_scaled"]:
            choice = int(np.argmax(scaled))
            row["choice_index_scaled"] = choice
            row["choice_label_scaled"] = labels[choice]
        else:
            row["choice_index_scaled"] = None
            row["choice_label_scaled"] = None
    return rows


def _metrics_from_rows(rows: list[dict], scaled: bool) -> dict:
    key = "probs_scaled" if scaled else "probs_unscaled"
    valid_key = "schema_valid_scaled" if scaled else "schema_valid_unscaled"
    kept = [row for row in rows if row[valid_key]]
    if not kept:
        return {"n": 0}
    probs = np.asarray([row[key] for row in kept], dtype=np.float64)
    gold = np.asarray([row["gold_index"] for row in kept], dtype=np.int64)
    return score_probabilities(probs, gold)


def _latency(rows: list[dict]) -> dict:
    values = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
        "mean_ms": float(array.mean()),
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run(protocol: dict, out_dir: Path, media_dir: Path, results_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run on CPU")
    splits_path = out_dir / "splits.json"
    materialized = load_items(splits_path, protocol)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = out_dir / "decisions"
    if decisions.exists():
        for child in decisions.glob("*.jsonl"):
            child.unlink()
    decisions.mkdir(parents=True, exist_ok=True)

    warmup_n = protocol["latency"]["warmup"]
    gpu_before = gpu_snapshot()
    print(json.dumps({"gpu_before": gpu_before}), flush=True)
    if gpu_before["torch_device_name"] != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(f"expected RTX 5070, found {gpu_before['torch_device_name']}")

    systems_out = {}
    failures = []
    for system in build_systems(protocol):
        print(f"loading {system.name}", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            system.load()
            device_name = system.device_name()
            after_load = gpu_snapshot()
            weight_mib = torch.cuda.memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            failures.append({"system": system.name, "stage": "load", "traceback": traceback.format_exc()})
            print(failures[-1]["traceback"], flush=True)
            system.close()
            continue
        system_result = {
            "device_name": device_name,
            "parameters_on_cuda": True,
            "dtype": system.dtype_name,
            "dtype_fallback": system.dtype_fallback,
            "weight_mib": weight_mib,
            "nvidia_smi_after_load": after_load,
            "datasets": {},
        }
        dev_rows = {}
        try:
            for dataset_name, spec in protocol["datasets"].items():
                labels = spec["labels"]
                items = materialized["items"][dataset_name]["dev"]
                print(f"dev {system.name} {dataset_name} n={len(items)}", flush=True)
                try:
                    rows = _score_items(system, items, labels, timed=False)
                except Exception:
                    failures.append(
                        {
                            "system": system.name,
                            "dataset": dataset_name,
                            "stage": "dev",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(failures[-1]["traceback"], flush=True)
                    continue
                nonfinite = [row for row in rows if row["error"] == "non-finite scores"]
                if (
                    len(nonfinite) == len(rows)
                    and rows
                    and hasattr(system, "fallback_to_float32")
                    and not system.dtype_fallback
                ):
                    print(f"non-finite dev scores; reloading {system.name} in float32", flush=True)
                    system.fallback_to_float32()
                    system_result["dtype"] = system.dtype_name
                    system_result["dtype_fallback"] = True
                    rows = _score_items(system, items, labels, timed=False)
                finite = [row for row in rows if row["scores"] is not None]
                if len(finite) != len(rows):
                    failures.append(
                        {
                            "system": system.name,
                            "dataset": dataset_name,
                            "stage": "dev",
                            "traceback": f"{len(rows) - len(finite)} dev items had no finite scores",
                        }
                    )
                if len(finite) < 50:
                    failures.append(
                        {
                            "system": system.name,
                            "dataset": dataset_name,
                            "stage": "dev",
                            "traceback": f"only {len(finite)} finite dev scores; temperature was not fit",
                        }
                    )
                    continue
                scores = np.asarray([row["scores"] for row in finite], dtype=np.float64)
                gold = np.asarray([row["gold_index"] for row in finite], dtype=np.int64)
                fitted = fit_temperature(scores, gold)
                _finalize(rows, labels, fitted["temperature"])
                for row in rows:
                    row["dataset"] = dataset_name
                    row["split"] = "dev"
                    _append_jsonl(decisions / f"{system.name}__{dataset_name}__dev.jsonl", row)
                dev_rows[dataset_name] = (rows, fitted)
                print(
                    f"temperature {system.name} {dataset_name} T={fitted['temperature']:.4f} "
                    f"dev_nll={fitted['dev_nll']:.4f}",
                    flush=True,
                )
            _write_json(
                out_dir / f"temperatures-{system.name}.json",
                {name: fitted for name, (_, fitted) in dev_rows.items()},
            )
            for dataset_name, (rows, fitted) in dev_rows.items():
                labels = protocol["datasets"][dataset_name]["labels"]
                test_items = materialized["items"][dataset_name]["test"]
                try:
                    print(f"warmup {system.name} {dataset_name}", flush=True)
                    for item in materialized["items"][dataset_name]["dev"][:warmup_n]:
                        torch.cuda.synchronize()
                        system.score(item.text, labels)
                        torch.cuda.synchronize()
                    print(f"test {system.name} {dataset_name} n={len(test_items)}", flush=True)
                    test_rows = _score_items(system, test_items, labels, timed=True)
                    _finalize(test_rows, labels, fitted["temperature"])
                    path = decisions / f"{system.name}__{dataset_name}__test.jsonl"
                    for row in test_rows:
                        row["dataset"] = dataset_name
                        row["split"] = "test"
                        _append_jsonl(path, row)
                    scaled = _metrics_from_rows(test_rows, scaled=True)
                    unscaled = _metrics_from_rows(test_rows, scaled=False)
                    scaled.pop("risk_coverage", None)
                    unscaled.pop("risk_coverage", None)
                    n_test = len(test_rows)
                    n_valid = sum(1 for row in test_rows if row["schema_valid_scaled"])
                    system_result["datasets"][dataset_name] = {
                        "temperature": fitted,
                        "scaled": scaled,
                        "unscaled": unscaled,
                        "latency": _latency(test_rows),
                        "schema_validity": n_valid / n_test if n_test else None,
                        "schema_validity_unscaled": (
                            sum(1 for row in test_rows if row["schema_valid_unscaled"]) / n_test if n_test else None
                        ),
                        "n_test": n_test,
                        "n_valid_scaled": n_valid,
                        "n_truncated": sum(1 for row in test_rows if row["truncated"]),
                        "output_tokens": 0,
                        "dev_n": len(rows),
                    }
                except Exception:
                    failures.append(
                        {
                            "system": system.name,
                            "dataset": dataset_name,
                            "stage": "test",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(failures[-1]["traceback"], flush=True)
        except Exception:
            failures.append({"system": system.name, "stage": "run", "traceback": traceback.format_exc()})
            print(failures[-1]["traceback"], flush=True)
        system_result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
        system_result["nvidia_smi_after_run"] = gpu_snapshot()
        systems_out[system.name] = system_result
        system.close()

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
        "git_head": _git_head(),
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "libraries": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "gliclass": importlib.metadata.version("gliclass"),
            "datasets": importlib.metadata.version("datasets"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "samples": {
            name: {
                "test_n": spec["test_n"],
                "dev_n": spec["dev_n"],
                "n_labels": len(spec["labels"]),
                "exact_text_overlap_dev_test": materialized["splits"][name]["exact_text_overlap_dev_test"],
                "source_n": materialized["splits"][name]["source_n"],
            }
            for name, spec in materialized["splits"].items()
        },
        "systems": systems_out,
        "failures": failures,
    }
    _write_json(out_dir / "metrics.json", payload)
    _write_json(out_dir / "failures.json", {"failures": failures})
    plot_paths = write_plots(decisions, [out_dir / "plots", media_dir], protocol)
    write_report(payload, results_path, plot_paths, protocol)
    print(f"wrote {out_dir / 'metrics.json'}", flush=True)
    print(f"wrote {results_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-splits", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--media", type=Path, default=MEDIA_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.freeze_splits:
        write_splits(args.out / "splits.json", protocol)
        print(f"wrote {args.out / 'splits.json'}", flush=True)
        return
    run(protocol, args.out, args.media, args.results)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

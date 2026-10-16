"""Score zero-shot systems on the wider Choice suite. Frozen rules are in suite_protocol.json."""

from __future__ import annotations

import json
import shutil
import sys
import time

import numpy as np

from bakeoff.full_banking77 import summarize
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.suite import OUT, load_suite, load_suite_protocol
from bakeoff.temperature import fit_temperature


def _out_of_scope(probs: np.ndarray, in_scope: np.ndarray) -> dict:
    """Confidence placed on inputs that have no correct option, next to in-scope test confidence."""
    top = probs.max(axis=1)
    reference = in_scope.max(axis=1)
    ranks = np.concatenate([top, reference]).argsort().argsort().astype(np.float64)
    # Probability that an in-scope item is more confident than an out-of-scope item (ties count half via ranks).
    auroc = (ranks[len(top) :].sum() - len(reference) * (len(reference) - 1) / 2) / (len(top) * len(reference))
    return {
        "n": int(len(top)),
        "mean_top_probability": float(top.mean()),
        "share_at_or_above_0.9": float((top >= 0.9).mean()),
        "share_at_or_above_0.5": float((top >= 0.5).mean()),
        "in_scope_mean_top_probability": float(reference.mean()),
        "auroc_in_scope_vs_out_of_scope_by_confidence": float(auroc),
    }


def _report(system: str, dataset: str, block: dict, scores: dict, extra: dict) -> dict:
    gold_dev = np.asarray([item.gold_index for item in block["dev"]], dtype=np.int64)
    gold_test = np.asarray([item.gold_index for item in block["test"]], dtype=np.int64)
    fitted = fit_temperature(scores["dev"], gold_dev)
    scaled = apply_temperature(scores["test"], fitted["temperature"])
    row = {
        "n_options": len(block["labels"]),
        "dev_n": len(gold_dev),
        "test_n": len(gold_test),
        "temperature": fitted,
        "scaled": summarize(scaled, gold_test),
        "unscaled": summarize(apply_temperature(scores["test"], 1.0), gold_test),
    }
    if "oos" in scores:
        row["out_of_scope_scaled"] = _out_of_scope(apply_temperature(scores["oos"], fitted["temperature"]), scaled)
        row["out_of_scope_unscaled"] = _out_of_scope(apply_temperature(scores["oos"], 1.0), apply_temperature(scores["test"], 1.0))
    row.update(extra)
    print(f"{system} {dataset}: acc {row['scaled']['accuracy']:.4f} F1 {row['scaled']['macro_f1']:.4f} brier {row['scaled']['brier']:.4f} unscaled brier {row['unscaled']['brier']:.4f}", flush=True)
    return row


def _write(system: str, summary: dict) -> None:
    (OUT / system).mkdir(parents=True, exist_ok=True)
    (OUT / system / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_jev() -> None:
    import httpx

    from bakeoff import jev

    spec = json.loads(jev.JEV_PROTOCOL_PATH.read_text())
    suite = load_suite()
    out = OUT / "jev"
    out.mkdir(parents=True, exist_ok=True)
    # Banking77 was already requested under the same rule; those rows are reused, not requested again.
    seeded = {
        out / "banking77_full__dev.jsonl": ROOT / "artifacts" / "jev" / "banking77__dev.jsonl",
        out / "banking77_full__test.jsonl": ROOT / "artifacts" / "banking77-full" / "jev__test.jsonl",
    }
    for target, source in seeded.items():
        if not target.exists() and source.exists():
            shutil.copyfile(source, target)
    summary = {"protocol_id": load_suite_protocol()["protocol_id"], "system": "jev", "datasets": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {jev._api_key(spec)}"}, timeout=90.0) as client:
        for dataset, block in suite.items():
            scores = {}
            usage = 0
            latencies = []
            for split in [name for name in ("dev", "test", "oos") if name in block]:
                path = out / f"{dataset}__{split}.jsonl"
                for attempt in range(4):
                    rows = jev._collect(client, spec, block[split], block["labels"], dataset, split, path)
                    errors = [row["error"] for row in rows if row["error"] is not None]
                    if not errors:
                        break
                    print(f"  jev {dataset} {split} pass {attempt + 1}: {len(errors)} errors, e.g. {errors[0][:100]}", flush=True)
                if errors:
                    raise RuntimeError(f"jev {dataset} {split}: {len(errors)} rows still in error")
                scores[split] = np.asarray([row["scores"] for row in rows], dtype=np.float64)
                usage += sum((row.get("usage") or {}).get("input_tokens", 0) for row in rows)
                if split == "test":
                    latencies = [row["latency_ms"] for row in rows]
            extra = {"input_tokens": int(usage), "latency_p50_ms": float(np.percentile(latencies, 50)), "latency_note": "HTTPS round trip, not comparable to local GPU"}
            summary["datasets"][dataset] = _report("jev", dataset, block, scores, extra)
            _write("jev", summary)


def run_llm(system: str) -> None:
    import torch

    from bakeoff.readout import READOUT_PATH, IndexReadout

    suite_spec = load_suite_protocol()
    system_spec = suite_spec["systems"][system]
    spec = json.loads(READOUT_PATH.read_text())
    spec["model"] = {"hf_id": system_spec["hf_id"], "revision": system_spec["revision"], "load_in_4bit": system_spec.get("load_in_4bit", False)}
    suite = load_suite()
    out = OUT / system
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol_id": suite_spec["protocol_id"], "system": system, "datasets": {}}
    readout = None
    for dataset, block in suite.items():
        if dataset in summary["datasets"]:
            print(f"skip {system} {dataset}", flush=True)
            continue
        if dataset in system_spec.get("skip_datasets", []):
            print(f"not run for {system}: {dataset}", flush=True)
            continue
        if readout is None:
            readout = IndexReadout(spec, prefix_batch=system_spec["prefix_batch"])
        labels = block["labels"]
        raw = {}
        latencies = []
        truncated = 0
        for split in [name for name in ("dev", "test", "oos") if name in block]:
            cache = out / f"{dataset}__{split}.npz"
            if cache.exists():
                raw[split] = np.load(cache)["index"]
                continue
            print(f"{system} {dataset} {split} n={len(block[split])} options={len(labels)}", flush=True)
            rows = []
            for index, item in enumerate(block[split], start=1):
                torch.cuda.synchronize()
                started = time.perf_counter()
                scores, was_truncated = readout.score(item.text, labels)
                torch.cuda.synchronize()
                if split == "test":
                    latencies.append((time.perf_counter() - started) * 1000.0)
                    truncated += int(was_truncated)
                if not np.isfinite(scores).all():
                    raise RuntimeError(f"non-finite score for {item.item_id}")
                rows.append(scores)
                if index % 250 == 0:
                    print(f"  {index}/{len(block[split])}", flush=True)
            raw[split] = np.stack(rows, axis=0)
            np.savez_compressed(cache, index=raw[split], latency_ms=np.asarray(latencies if split == "test" else [], dtype=np.float64))
        prior = raw["dev"].mean(axis=0)
        scores = {split: values - prior for split, values in raw.items()}
        test_cache = np.load(out / f"{dataset}__test.npz")
        timing = test_cache["latency_ms"] if "latency_ms" in test_cache.files and len(test_cache["latency_ms"]) else np.asarray(latencies, dtype=np.float64)
        extra = {
            "readout": "index_prior",
            "n_truncated_test": truncated,
            "output_tokens": 0,
            "latency": {
                "n": int(timing.size),
                "p50_ms": float(np.percentile(timing, 50)) if timing.size else None,
                "p95_ms": float(np.percentile(timing, 95)) if timing.size else None,
                "note": "includes the first few calls; no separate warmup",
            },
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        }
        summary["datasets"][dataset] = _report(system, dataset, block, scores, extra)
        _write(system, summary)
    print(f"SUITE_DONE {system}", flush=True)


def _jsonl_scores(path, items) -> np.ndarray:
    by_id = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_id[row["item_id"]] = row["scores"]
    return np.asarray([by_id[item.item_id] for item in items], dtype=np.float64)


def run_gliclass_cpu() -> None:
    """GLiClass on the processor, float32 as in Program 1. Scores only; latency is not timed here."""
    import torch

    from bakeoff.protocol import load_protocol
    from bakeoff.systems import GLiClassChoice

    name = "gliclass-base-v3.0"
    suite_spec = load_suite_protocol()
    torch.set_num_threads(suite_spec["systems"][name]["cpu_threads"])
    suite = load_suite()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    # Rows already scored on the GPU under the same float32 rule are reused.
    reuse = {
        ("banking77_full", "dev"): ROOT / "artifacts" / "decisions" / f"{name}__banking77__dev.jsonl",
        ("massive_en", "dev"): ROOT / "artifacts" / "primary" / "zeroshot" / name / "dev.jsonl",
        ("massive_en", "test"): ROOT / "artifacts" / "primary" / "zeroshot" / name / "test.jsonl",
    }
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol_id": suite_spec["protocol_id"], "system": name, "device": "cpu float32 for new rows", "datasets": {}}
    system = GLiClassChoice(load_protocol()["systems"][name], device="cpu")
    system.load()
    check = np.load(ROOT / "artifacts" / "banking77-full" / f"{name}__scores.npz")["scores"]
    block = suite["banking77_full"]
    fresh = np.stack([system.score(item.text, block["labels"])[0] for item in block["test"][:40]], axis=0)
    summary["cpu_vs_gpu_check"] = {
        "items": 40,
        "max_abs_score_difference": float(np.abs(fresh - check[:40]).max()),
        "argmax_changes": int((fresh.argmax(axis=1) != check[:40].argmax(axis=1)).sum()),
    }
    print(f"cpu vs gpu on 40 Banking77 items: {summary['cpu_vs_gpu_check']}", flush=True)
    for dataset, block in suite.items():
        if dataset in summary["datasets"]:
            print(f"skip {name} {dataset}", flush=True)
            continue
        scores = {}
        truncated = 0
        for split in [part for part in ("dev", "test", "oos") if part in block]:
            cache = out / f"{dataset}__{split}.npz"
            if cache.exists():
                scores[split] = np.load(cache)["scores"]
                continue
            if (dataset, split) in reuse:
                scores[split] = _jsonl_scores(reuse[(dataset, split)], block[split])
            elif (dataset, split) == ("banking77_full", "test"):
                scores[split] = check
            else:
                print(f"{name} cpu {dataset} {split} n={len(block[split])} options={len(block['labels'])}", flush=True)
                rows = []
                for index, item in enumerate(block[split], start=1):
                    row, was_truncated = system.score(item.text, block["labels"])
                    rows.append(row)
                    truncated += int(was_truncated and split == "test")
                    if index % 250 == 0:
                        print(f"  {index}/{len(block[split])}", flush=True)
                scores[split] = np.stack(rows, axis=0)
            np.savez_compressed(cache, scores=scores[split])
        summary["datasets"][dataset] = _report(name, dataset, block, scores, {"n_truncated_test_new_rows": truncated, "output_tokens": 0, "latency": "not timed on cpu"})
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SUITE_DONE {name}", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "jev":
        run_jev()
    elif command == "gliclass-cpu":
        run_gliclass_cpu()
    elif command == "llm" and len(sys.argv) > 2:
        run_llm(sys.argv[2])
    else:
        raise SystemExit("usage: python -m bakeoff.suite_run jev | llm <system>")

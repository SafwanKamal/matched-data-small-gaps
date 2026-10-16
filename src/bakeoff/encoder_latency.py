"""Batch-1 latency for Program 2 encoder heads already saved on disk.

Uses the Program 1 timing rule: one text per call, twenty unscored dev warmups,
cuda synchronize, linear p50 and p95 on the test split.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from bakeoff.data import load_items
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.train_ettin import PROGRAM2_PATH, EncoderClassifier, _encode


def _time_job(job: str, spec: dict, protocol: dict, materialized, device) -> dict:
    metrics_path = ROOT / "artifacts" / "program2" / job / "metrics.json"
    record = json.loads(metrics_path.read_text())
    dataset = record["dataset"]
    labels = protocol["datasets"][dataset]["labels"]
    encoder_dir = metrics_path.parent / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(encoder_dir, dtype=torch.bfloat16)
    model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
    head = torch.load(metrics_path.parent / "head.pt", map_location="cpu", weights_only=False)
    model.head.load_state_dict(head["head"])
    model.eval()
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(torch.cuda.get_device_name(0))
    items = materialized["items"][dataset]
    warmup = protocol["latency"]["warmup"]
    timed = []
    with torch.inference_mode():
        for item in items["dev"][:warmup]:
            encoded = _encode(tokenizer, [item.text], spec["max_length"], device)
            model(encoded["input_ids"], encoded["attention_mask"])
            torch.cuda.synchronize()
        for item in items["test"]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            encoded = _encode(tokenizer, [item.text], spec["max_length"], device)
            model(encoded["input_ids"], encoded["attention_mask"])
            torch.cuda.synchronize()
            timed.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(timed, dtype=np.float64)
    record["latency"] = {
        "n": int(array.size),
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{job} p50 {record['latency']['p50_ms']:.2f} p95 {record['latency']['p95_ms']:.2f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return record


def main() -> None:
    spec = json.loads(PROGRAM2_PATH.read_text())
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    device = torch.device("cuda")
    summary_path = ROOT / "artifacts" / "program2" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    updated = []
    for dataset in spec["datasets"]:
        for kind in spec["losses"]:
            job = f"{dataset}-{kind}"
            updated.append(_time_job(job, spec, protocol, materialized, device))
    if summary is not None:
        by_key = {(row["dataset"], row["loss"]): row for row in updated}
        for row in summary["runs"]:
            fresh = by_key.get((row["dataset"], row["loss"]))
            if fresh and "latency" in fresh:
                row["latency"] = fresh["latency"]
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"updated {summary_path}", flush=True)


if __name__ == "__main__":
    main()

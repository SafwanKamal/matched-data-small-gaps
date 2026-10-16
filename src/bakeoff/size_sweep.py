"""Ettin size sweep. Frozen rules are in size_sweep_protocol.json."""

from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np
import torch

from bakeoff import train_ettin as enc
from bakeoff import train_ettin_decoder as dec
from bakeoff.data import load_items
from bakeoff.massive import load_massive, load_primary_protocol
from bakeoff.protocol import ROOT, load_protocol

RULES_PATH = ROOT / "size_sweep_protocol.json"
OUT = ROOT / "artifacts" / "size-sweep"


def _data(dataset: str):
    """labels, a protocol stub, the materialized dev and test items, and the training rows (None = Program 2 loader)."""
    if dataset == "massive_en":
        primary = load_primary_protocol()
        loaded = load_massive(primary)
        stub = {"datasets": {dataset: {"labels": loaded["labels"]}}}
        materialized = {"splits": {dataset: {"dev": {"rows": []}}}, "items": {dataset: {"dev": loaded["items"]["dev"], "test": loaded["items"]["test"]}}}
        rows = [(item.text, item.gold_index) for item in loaded["items"]["train"]]
        return stub, materialized, rows
    protocol = load_protocol()
    return protocol, load_items(ROOT / "artifacts" / "splits.json", protocol), None


def _spec(kind: str, dataset: str, size: dict, rate: float, seed: int) -> dict:
    if dataset == "massive_en":
        spec = dict(load_primary_protocol()[f"ettin_{kind}"])
    else:
        spec = json.loads((ROOT / ("program2.json" if kind == "encoder" else "program2_decoder.json")).read_text())
    spec.update({"model": size[kind], "revision": size[f"{kind}_revision"], "learning_rate": rate, "seed": seed})
    return spec


def _encoder_latency(job_dir, spec: dict, labels, items, warmup: int) -> tuple[dict, int]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(job_dir / "encoder")
    encoder = AutoModel.from_pretrained(job_dir / "encoder", dtype=torch.float32)
    model = enc.EncoderClassifier(encoder, encoder.config.hidden_size, len(labels))
    model.head.load_state_dict(torch.load(job_dir / "head.pt", map_location="cpu", weights_only=True)["head"])
    model.to("cuda").eval()
    device = torch.device("cuda")
    timed = []
    with torch.inference_mode():
        for item in items["dev"][:warmup]:
            encoded = enc._encode(tokenizer, [item.text], spec["max_length"], device)
            model(encoded["input_ids"], encoded["attention_mask"])
            torch.cuda.synchronize()
        for item in items["test"]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            encoded = enc._encode(tokenizer, [item.text], spec["max_length"], device)
            model(encoded["input_ids"], encoded["attention_mask"])
            torch.cuda.synchronize()
            timed.append((time.perf_counter() - started) * 1000.0)
    parameters = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    array = np.asarray(timed)
    return {"n": int(array.size), "p50_ms": float(np.percentile(array, 50)), "p95_ms": float(np.percentile(array, 95))}, parameters


def run(kind: str, size_names: list[str], rates: list[float], datasets: list[str] | None = None) -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    warmup = load_protocol()["latency"]["warmup"]
    for rate in rates:
        for size_name in size_names:
            for dataset in datasets or rules["datasets"]:
                job = f"{kind}-{size_name}/lr-{rate:g}/{dataset}"
                done = OUT / job / "sweep.json"
                if done.exists():
                    print(f"skip {job}", flush=True)
                    continue
                print(f"start {job}", flush=True)
                started = time.perf_counter()
                try:
                    protocol, materialized, rows = _data(dataset)
                    spec = _spec(kind, dataset, rules["sizes"][size_name], rate, rules["seed"])
                    torch.manual_seed(rules["seed"])
                    torch.cuda.reset_peak_memory_stats()
                    module = enc if kind == "encoder" else dec
                    result = module._train_one(job, dataset, rules["loss"], spec, protocol, materialized, device, rows=rows, out_root=OUT)
                    labels = protocol["datasets"][dataset]["labels"]
                    if kind == "encoder":
                        result["latency"], result["parameters"] = _encoder_latency(OUT / job, spec, labels, materialized["items"][dataset], warmup)
                    else:
                        from transformers import AutoModelForCausalLM

                        saved = AutoModelForCausalLM.from_pretrained(OUT / job / "model")
                        result["parameters"] = sum(p.numel() for p in saved.parameters())
                        del saved
                    result.update({"protocol_id": rules["protocol_id"], "size": size_name, "kind": kind, "learning_rate": rate, "seed": rules["seed"], "minutes": (time.perf_counter() - started) / 60.0})
                    done.write_text(json.dumps(result, indent=2) + "\n")
                    print(f"done {job} acc {result['scaled']['accuracy']:.4f} dev_nll {result['temperature']['dev_nll']:.4f} p50 {result['latency']['p50_ms']:.1f} ms {result['minutes']:.1f} min", flush=True)
                except Exception:
                    print(f"FAILED {job}\n{traceback.format_exc()}", flush=True)
                    torch.cuda.empty_cache()


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    matched, extra = rules["learning_rates"]["matched"], rules["learning_rates"]["extra"]
    small = ["32m", "68m", "17m"]
    if stage in ("all", "encoders"):
        run("encoder", small, [matched])
        run("encoder", small + ["150m"], extra)
    if stage in ("all", "decoders"):
        run("decoder", small, [matched])
        run("decoder", small, [1e-4])
        run("decoder", ["150m"], [1e-4], ["banking77", "massive_en"])
    print("SIZE_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()

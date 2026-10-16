"""MASSIVE English intent scores. Temperature uses the official dev split."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bakeoff.massive import load_massive, load_primary_protocol, protocol_sha256
from bakeoff.metrics import apply_temperature, schema_valid
from bakeoff.protocol import ROOT
from bakeoff.run import (
    _append_jsonl,
    _finalize,
    _git_head,
    _latency,
    _metrics_from_rows,
    _score_items,
    _write_json,
    gpu_snapshot,
)
from bakeoff.systems import build_systems
from bakeoff.temperature import fit_temperature
from bakeoff.train_ettin import (
    EncoderClassifier,
    _batches,
    _encode,
    _metrics,
    decision_loss,
)
from bakeoff.train_ettin_decoder import _apply_loss, _continuation_ids, _score_items as _decoder_score_items
from bakeoff.train_ettin_decoder import _score_texts

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUT = ROOT / "artifacts" / "primary"
PROBE_PATH = OUT / "probe.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(f"expected RTX 5070, found {name}")
    return torch.device("cuda")


def _sha_matches(path: Path, digest: str) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if payload.get("protocol_sha256") != digest:
        raise RuntimeError(f"{path} belongs to a different primary protocol")
    return True


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp.replace(path)


def _aligned(rows: list[dict], items) -> int:
    if len(rows) > len(items):
        raise RuntimeError("decision file is longer than the official split")
    for row, item in zip(rows, items):
        if row["item_id"] != item.item_id or row["gold_index"] != item.gold_index:
            raise RuntimeError("existing decisions do not match the frozen MASSIVE order")
    return len(rows)


def _stamp(payload: dict, protocol: dict) -> dict:
    payload["protocol_id"] = protocol["protocol_id"]
    payload["protocol_sha256"] = protocol_sha256()
    payload["git_head"] = _git_head()
    payload["finished_at_utc"] = _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")
    return payload


def _scores_from_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    missing = [row["item_id"] for row in rows if row.get("scores") is None]
    if missing:
        raise RuntimeError(f"{len(missing)} items have no finite scores, first {missing[0]}")
    scores = np.asarray([row["scores"] for row in rows], dtype=np.float64)
    gold = np.asarray([row["gold_index"] for row in rows], dtype=np.int64)
    return scores, gold


def _pop_curves(metrics: dict) -> dict:
    metrics.pop("risk_coverage", None)
    metrics.pop("ece_bins", None)
    return metrics


def _warmup(system, items, labels: list[str], n: int) -> None:
    for item in items[:n]:
        torch.cuda.synchronize()
        system.score(item.text, labels)
        torch.cuda.synchronize()


def zeroshot(protocol: dict | None = None) -> dict:
    protocol = protocol or load_primary_protocol()
    if protocol["llm_aggrefact"]["scored"]:
        raise RuntimeError("this protocol does not score LLM-AggreFact")
    device_ok = _require_gpu()
    del device_ok
    loaded = load_massive(protocol)
    spec = protocol["datasets"]["massive_en"]
    labels = spec["labels"]
    digest = protocol_sha256()
    warmup_n = protocol["latency"]["warmup"]
    out = OUT / "zeroshot"
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    systems_out = {}
    print(json.dumps({"gpu_before": gpu_snapshot()}), flush=True)
    for system in build_systems(protocol):
        metrics_path = out / system.name / "metrics.json"
        if _sha_matches(metrics_path, digest):
            print(f"skip {system.name}", flush=True)
            systems_out[system.name] = json.loads(metrics_path.read_text())
            continue
        dev_path = out / system.name / "dev.jsonl"
        test_path = out / system.name / "test.jsonl"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"loading {system.name}", flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            system.load()
            device_name = system.device_name()
            dev_rows = _read_jsonl(dev_path)
            done = _aligned(dev_rows, loaded["items"]["dev"])
            if done < spec["dev_n"]:
                print(f"dev {system.name} {done}/{spec['dev_n']}", flush=True)
                fresh = _score_items(system, loaded["items"]["dev"][done:], labels, timed=False)
                for row in fresh:
                    row["dataset"] = "massive_en"
                    row["split"] = "dev"
                    row["source_id"] = str(row["row"])
                    _append_jsonl(dev_path, row)
                dev_rows.extend(fresh)
            if len(dev_rows) != spec["dev_n"]:
                raise RuntimeError(f"dev rows {len(dev_rows)} != {spec['dev_n']}")
            dev_scores, dev_gold = _scores_from_rows(dev_rows)
            fitted = fit_temperature(dev_scores, dev_gold)
            if fitted["n"] != spec["dev_n"]:
                raise RuntimeError(f"temperature n {fitted['n']} != {spec['dev_n']}")
            _finalize(dev_rows, labels, fitted["temperature"])
            _atomic_jsonl(dev_path, dev_rows)
            print(
                f"temperature {system.name} T={fitted['temperature']:.4f} dev_nll={fitted['dev_nll']:.4f}",
                flush=True,
            )
            test_rows = _read_jsonl(test_path)
            done = _aligned(test_rows, loaded["items"]["test"])
            if done < spec["test_n"]:
                print(f"warmup {system.name}", flush=True)
                _warmup(system, loaded["items"]["dev"], labels, warmup_n)
                print(f"test {system.name} {done}/{spec['test_n']}", flush=True)
                fresh = _score_items(system, loaded["items"]["test"][done:], labels, timed=True)
                for row in fresh:
                    row["dataset"] = "massive_en"
                    row["split"] = "test"
                    row["source_id"] = str(row["row"])
                    _append_jsonl(test_path, row)
                test_rows.extend(fresh)
            if len(test_rows) != spec["test_n"]:
                raise RuntimeError(f"test rows {len(test_rows)} != {spec['test_n']}")
            _finalize(test_rows, labels, fitted["temperature"])
            _atomic_jsonl(test_path, test_rows)
            scaled = _pop_curves(_metrics_from_rows(test_rows, scaled=True))
            unscaled = _pop_curves(_metrics_from_rows(test_rows, scaled=False))
            if scaled.get("n") != spec["test_n"]:
                raise RuntimeError(f"scaled n {scaled.get('n')} != {spec['test_n']}")
            result = _stamp(
                {
                    "system": system.name,
                    "dataset": "massive_en",
                    "task": "intent",
                    "slots": "ignored",
                    "test_is_full_split": True,
                    "subsample": None,
                    "dev_n": spec["dev_n"],
                    "test_n": spec["test_n"],
                    "n_labels": len(labels),
                    "temperature": fitted,
                    "scaled": scaled,
                    "unscaled": unscaled,
                    "latency": _latency(test_rows),
                    "schema_validity": sum(1 for row in test_rows if row["schema_valid_scaled"]) / spec["test_n"],
                    "n_truncated": sum(1 for row in test_rows if row["truncated"]),
                    "output_tokens": 0,
                    "device_name": device_name,
                    "dtype": system.dtype_name,
                    "dtype_fallback": system.dtype_fallback,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                },
                protocol,
            )
            _write_json(metrics_path, result)
            systems_out[system.name] = result
            print(
                f"{system.name} acc {scaled['accuracy']:.4f} brier {scaled['brier']:.4f} "
                f"p50 {result['latency']['p50_ms']:.1f} ms",
                flush=True,
            )
        except Exception:
            failures.append({"system": system.name, "traceback": traceback.format_exc()})
            print(failures[-1]["traceback"], flush=True)
        finally:
            system.close()
    summary = _stamp(
        {
            "dataset": "massive_en",
            "counts": loaded["counts"],
            "text_overlap_kept": loaded["text_overlap_kept"],
            "file_sha256": loaded["sha256"],
            "systems": systems_out,
            "failures": failures,
            "llm_aggrefact_scored": False,
        },
        protocol,
    )
    _write_json(out / "summary.json", summary)
    print(f"wrote {out / 'summary.json'}", flush=True)
    return summary


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _projection(step_s: float, eval_s: float, latency_s: float, load_s: float, spec: dict, margin: float) -> dict:
    data = load_primary_protocol()["datasets"]["massive_en"]
    steps = math.ceil(data["train_n"] / spec["batch_size"]) * spec["epochs"]
    eval_batches = math.ceil(data["dev_n"] / spec["eval_batch_size"]) + math.ceil(
        data["test_n"] / spec["eval_batch_size"]
    )
    raw = load_s + steps * step_s + eval_batches * eval_s + data["test_n"] * latency_s
    return {
        "steps": steps,
        "eval_batches": eval_batches,
        "median_step_s": step_s,
        "median_eval_batch_s": eval_s,
        "median_latency_s": latency_s,
        "load_s": load_s,
        "raw_seconds": raw,
        "projected_seconds": raw * margin,
        "margin": margin,
        "batch_size": spec["batch_size"],
        "eval_batch_size": spec["eval_batch_size"],
    }


def probe(protocol: dict | None = None) -> dict:
    """Time training steps on official train rows. This file is not a test metric."""
    protocol = protocol or load_primary_protocol()
    if PROBE_PATH.exists():
        existing = json.loads(PROBE_PATH.read_text())
        if existing.get("protocol_sha256") != protocol_sha256():
            raise RuntimeError("probe.json belongs to a different protocol")
        print(f"probe already written {PROBE_PATH}", flush=True)
        return existing
    device = _require_gpu()
    loaded = load_massive(protocol)
    train_items = loaded["items"]["train"]
    rows = [(item.text, item.gold_index) for item in train_items]
    labels = protocol["datasets"]["massive_en"]["labels"]
    margin = protocol["probe"]["margin"]
    encoder_spec = protocol["ettin_encoder"]
    decoder_spec = protocol["ettin_decoder"]
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(encoder_spec["seed"])
    started = time.perf_counter()
    enc_tokenizer = AutoTokenizer.from_pretrained(encoder_spec["model"], revision=encoder_spec["revision"])
    encoder = AutoModel.from_pretrained(encoder_spec["model"], revision=encoder_spec["revision"], dtype=torch.float32)
    enc_model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
    enc_opt = torch.optim.AdamW(
        enc_model.parameters(), lr=encoder_spec["learning_rate"], weight_decay=encoder_spec["weight_decay"]
    )
    load_s = time.perf_counter() - started
    enc_model.train()
    step_times = []
    taken = 0
    for batch in _batches(rows, encoder_spec["batch_size"], encoder_spec["seed"], 0):
        if taken == 4:
            break
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoded = _encode(enc_tokenizer, [text for text, _ in batch], encoder_spec["max_length"], device)
        gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
        enc_opt.zero_grad(set_to_none=True)
        loss = decision_loss(enc_model(encoded["input_ids"], encoded["attention_mask"]), gold, "log_loss")
        loss.backward()
        enc_opt.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        taken += 1
    enc_model.eval()
    eval_chunk = train_items[: encoder_spec["eval_batch_size"]]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        encoded = _encode(enc_tokenizer, [item.text for item in eval_chunk], encoder_spec["max_length"], device)
        enc_model(encoded["input_ids"], encoded["attention_mask"])
    torch.cuda.synchronize()
    eval_s = time.perf_counter() - t0
    latency = []
    with torch.inference_mode():
        for item in train_items[:4]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            encoded = _encode(enc_tokenizer, [item.text], encoder_spec["max_length"], device)
            enc_model(encoded["input_ids"], encoded["attention_mask"])
            torch.cuda.synchronize()
            latency.append(time.perf_counter() - t0)
    encoder_projection = _projection(step_times[-1], eval_s, _median(latency), load_s, encoder_spec, margin)
    encoder_projection["step_seconds"] = step_times
    del enc_model, encoder, enc_opt
    torch.cuda.empty_cache()

    started = time.perf_counter()
    dec_tokenizer = AutoTokenizer.from_pretrained(decoder_spec["model"], revision=decoder_spec["revision"])
    dec_model = AutoModelForCausalLM.from_pretrained(
        decoder_spec["model"], revision=decoder_spec["revision"], dtype=torch.float32
    )
    dec_model.config.use_cache = False
    dec_model.to(device)
    dec_opt = torch.optim.AdamW(
        dec_model.parameters(), lr=decoder_spec["learning_rate"], weight_decay=decoder_spec["weight_decay"]
    )
    load_s = time.perf_counter() - started
    for label in labels:
        _continuation_ids(dec_tokenizer, "continuation check", label, decoder_spec)
    _apply_loss.drift_logged = False
    dec_model.train()
    step_times = []
    taken = 0
    chunk = decoder_spec["label_chunk_size"]
    for batch in _batches(rows, decoder_spec["batch_size"], decoder_spec["seed"], 0):
        if taken == 2:
            break
        texts = [text for text, _ in batch]
        gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec_opt.zero_grad(set_to_none=True)
        loss = _apply_loss(dec_model, dec_tokenizer, texts, labels, gold, "log_loss", decoder_spec, device, chunk)
        torch.nn.utils.clip_grad_norm_(dec_model.parameters(), decoder_spec["max_grad_norm"])
        dec_opt.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        taken += 1
        print(f"decoder probe step {taken} {step_times[-1]:.3f}s loss {float(loss):.4f}", flush=True)
    eval_items = train_items[: decoder_spec["eval_batch_size"]]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _score_texts(dec_model, dec_tokenizer, [item.text for item in eval_items], labels, decoder_spec, device, chunk)
    torch.cuda.synchronize()
    eval_s = time.perf_counter() - t0
    latency = []
    with torch.inference_mode():
        for item in train_items[:2]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _score_texts(dec_model, dec_tokenizer, [item.text], labels, decoder_spec, device, chunk)
            torch.cuda.synchronize()
            latency.append(time.perf_counter() - t0)
    decoder_projection = _projection(step_times[-1], eval_s, _median(latency), load_s, decoder_spec, margin)
    decoder_projection["step_seconds"] = step_times
    del dec_model, dec_opt
    torch.cuda.empty_cache()
    payload = _stamp(
        {
            "role": "wall-clock gate, not a metric",
            "split_scored": None,
            "encoder": encoder_projection,
            "decoder": decoder_projection,
        },
        protocol,
    )
    _write_json(PROBE_PATH, payload)
    print(json.dumps({"encoder_s": encoder_projection["projected_seconds"], "decoder_s": decoder_projection["projected_seconds"]}), flush=True)
    return payload


def _will_finish(projected_seconds: float, protocol: dict) -> bool:
    end = _utc_now().timestamp() + projected_seconds
    return end <= _parse_utc(protocol["finetune_must_finish_by_utc"]).timestamp()


def _scheduler(optimizer, spec: dict, n_rows: int, batch_size: int):
    steps = max(1, math.ceil(n_rows / batch_size))
    total = steps * spec["epochs"]
    warmup = int(spec["warmup_ratio"] * total)

    def lr_at(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        remain = max(1, total - warmup)
        return max(0.0, (total - step) / remain)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at), total, warmup


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_optimizer(optimizer, state) -> None:
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        for param in group["params"]:
            entry = optimizer.state.get(param)
            if not entry:
                continue
            for key, value in list(entry.items()):
                if torch.is_tensor(value):
                    entry[key] = value.to(param.device)


def _encoder_latency(model, tokenizer, items, spec: dict, warmup: int) -> dict:
    device = next(model.parameters()).device
    model.eval()
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
    return {
        "n": int(array.size),
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
        "mean_ms": float(array.mean()),
    }


def finetune_encoder(protocol: dict, loaded: dict, projected_seconds: float) -> dict:
    from transformers import AutoModel, AutoTokenizer

    spec = protocol["ettin_encoder"]
    if spec["losses"] != ["log_loss"]:
        raise RuntimeError("primary encoder loss is log_loss only")
    out_dir = OUT / "ettin-encoder"
    metrics_path = out_dir / "metrics.json"
    digest = protocol_sha256()
    if _sha_matches(metrics_path, digest):
        print("skip ettin encoder", flush=True)
        return json.loads(metrics_path.read_text())
    if not _will_finish(projected_seconds, protocol):
        skipped = _stamp(
            {"skipped": True, "reason": "projected finetune does not finish by 20:35 UTC", "projected_seconds": projected_seconds},
            protocol,
        )
        _write_json(out_dir / "skipped.json", skipped)
        print("skip encoder launch; projection misses the cutoff", flush=True)
        return skipped
    device = _require_gpu()
    labels = protocol["datasets"]["massive_en"]["labels"]
    items = loaded["items"]
    rows = [(item.text, item.gold_index) for item in items["train"]]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    batch_size = spec["batch_size"]
    checkpoint = out_dir / "checkpoint.pt"
    oom_fallback = False
    blob = None
    if checkpoint.exists():
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob["protocol_sha256"] != digest:
            raise RuntimeError("encoder checkpoint belongs to a different protocol")
        batch_size = blob["batch_size"]
        oom_fallback = blob["oom_fallback"]

    def build(width: int):
        encoder = AutoModel.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
        model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"Ettin encoder parameters are on {devices}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
        scheduler, _, _ = _scheduler(optimizer, spec, len(rows), width)
        return model, optimizer, scheduler

    model, optimizer, scheduler = build(batch_size)
    start_epoch = 0
    losses: list[float] = []
    if blob is not None:
        model.load_state_dict(blob["model"])
        _load_optimizer(optimizer, blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        start_epoch = blob["epoch"]
        losses = blob["losses"]
        print(f"resume encoder at epoch {start_epoch}", flush=True)
    epoch = start_epoch
    while epoch < spec["epochs"]:
        try:
            model.train()
            epoch_losses = []
            for batch in _batches(rows, batch_size, spec["seed"], epoch):
                encoded = _encode(tokenizer, [text for text, _ in batch], spec["max_length"], device)
                gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = decision_loss(model(encoded["input_ids"], encoded["attention_mask"]), gold, "log_loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.extend(epoch_losses)
            epoch += 1
            print(f"encoder epoch {epoch} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
            _save_checkpoint(
                checkpoint,
                {
                    "protocol_sha256": digest,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "losses": losses,
                    "batch_size": batch_size,
                    "oom_fallback": oom_fallback,
                },
            )
        except torch.cuda.OutOfMemoryError:
            if oom_fallback:
                raise
            oom_fallback = True
            batch_size = spec["oom_batch_fallback"]
            print(f"encoder OOM, reloading at batch {batch_size}", flush=True)
            del model, optimizer, scheduler
            if checkpoint.exists():
                checkpoint.unlink()
            torch.cuda.empty_cache()
            model, optimizer, scheduler = build(batch_size)
            losses = []
            epoch = 0
    if epoch != spec["epochs"]:
        raise RuntimeError("refusing to score test from a partial encoder run")
    from bakeoff.train_ettin import _score_items as _encoder_score_items

    dev_scores, dev_gold, _ = _encoder_score_items(
        model, tokenizer, items["dev"], spec, device, spec["eval_batch_size"]
    )
    fitted = fit_temperature(dev_scores, dev_gold)
    test_scores, test_gold, n_truncated = _encoder_score_items(
        model, tokenizer, items["test"], spec, device, spec["eval_batch_size"]
    )
    latency = _encoder_latency(model, tokenizer, items, spec, protocol["latency"]["warmup"])
    if not all(schema_valid(row, len(labels)) for row in apply_temperature(test_scores, fitted["temperature"])):
        raise RuntimeError("encoder head was not a simplex")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"head": model.head.state_dict(), "labels": labels, "loss": "log_loss"}, out_dir / "head.pt")
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    result = _stamp(
        {
            "system": "ettin-encoder-150m",
            "dataset": "massive_en",
            "task": "intent",
            "slots": "ignored",
            "loss": "log_loss",
            "test_is_full_split": True,
            "subsample": None,
            "train_n": len(rows),
            "dev_n": len(items["dev"]),
            "test_n": len(items["test"]),
            "epochs": spec["epochs"],
            "batch_size": batch_size,
            "oom_fallback": oom_fallback,
            "learning_rate": spec["learning_rate"],
            "mean_train_loss": float(np.mean(losses)),
            "temperature": fitted,
            "scaled": _metrics(test_scores, test_gold, fitted["temperature"]),
            "unscaled": _metrics(test_scores, test_gold, 1.0),
            "n_truncated_test": n_truncated,
            "latency": latency,
            "output_tokens": 0,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "device": torch.cuda.get_device_name(0),
        },
        protocol,
    )
    np.savez_compressed(out_dir / "dev_scores.npz", scores=dev_scores, gold=dev_gold)
    np.savez_compressed(out_dir / "test_scores.npz", scores=test_scores, gold=test_gold)
    _write_json(metrics_path, result)
    if checkpoint.exists():
        checkpoint.unlink()
    print(
        f"encoder test acc {result['scaled']['accuracy']:.4f} brier {result['scaled']['brier']:.4f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return result


def _decoder_latency(model, tokenizer, items, labels, spec, device, warmup: int, chunk_size: int) -> dict:
    model.eval()
    timed = []
    with torch.inference_mode():
        for item in items["dev"][:warmup]:
            _score_texts(model, tokenizer, [item.text], labels, spec, device, chunk_size)
            torch.cuda.synchronize()
        for item in items["test"]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            _score_texts(model, tokenizer, [item.text], labels, spec, device, chunk_size)
            torch.cuda.synchronize()
            timed.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(timed, dtype=np.float64)
    return {
        "n": int(array.size),
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
        "mean_ms": float(array.mean()),
    }


def finetune_decoder(protocol: dict, loaded: dict, projected_seconds: float) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = protocol["ettin_decoder"]
    if spec["losses"] != ["log_loss"] or spec["generate"] or spec["rationale"]:
        raise RuntimeError("primary decoder is closed-set log loss")
    out_dir = OUT / "ettin-decoder"
    metrics_path = out_dir / "metrics.json"
    digest = protocol_sha256()
    if _sha_matches(metrics_path, digest):
        print("skip ettin decoder", flush=True)
        return json.loads(metrics_path.read_text())
    if not _will_finish(projected_seconds, protocol):
        skipped = _stamp(
            {"skipped": True, "reason": "projected finetune does not finish by 20:35 UTC", "projected_seconds": projected_seconds},
            protocol,
        )
        _write_json(out_dir / "skipped.json", skipped)
        print("skip decoder launch; projection misses the cutoff", flush=True)
        return skipped
    device = _require_gpu()
    labels = protocol["datasets"]["massive_en"]["labels"]
    items = loaded["items"]
    rows = [(item.text, item.gold_index) for item in items["train"]]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    if tokenizer.bos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("decoder tokenizer is missing bos or pad")
    for label in labels:
        _continuation_ids(tokenizer, "continuation check", label, spec)
    batch_size = spec["batch_size"]
    chunk_size = spec["label_chunk_size"]
    checkpoint = out_dir / "checkpoint.pt"
    oom_fallback = False
    blob = None
    if checkpoint.exists():
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if blob["protocol_sha256"] != digest:
            raise RuntimeError("decoder checkpoint belongs to a different protocol")
        batch_size = blob["batch_size"]
        chunk_size = blob["chunk_size"]
        oom_fallback = blob["oom_fallback"]
    _apply_loss.drift_logged = False

    def build(width: int, chunks: int):
        del chunks
        model = AutoModelForCausalLM.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
        model.config.use_cache = False
        model.to(device)
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"Ettin decoder parameters are on {devices}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
        scheduler, _, _ = _scheduler(optimizer, spec, len(rows), width)
        return model, optimizer, scheduler

    model, optimizer, scheduler = build(batch_size, chunk_size)
    start_epoch = 0
    losses: list[float] = []
    if blob is not None:
        model.load_state_dict(blob["model"])
        _load_optimizer(optimizer, blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        start_epoch = blob["epoch"]
        losses = blob["losses"]
        print(f"resume decoder at epoch {start_epoch}", flush=True)
    epoch = start_epoch
    while epoch < spec["epochs"]:
        try:
            model.train()
            epoch_losses = []
            for batch in _batches(rows, batch_size, spec["seed"], epoch):
                texts = [text for text, _ in batch]
                gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = _apply_loss(model, tokenizer, texts, labels, gold, "log_loss", spec, device, chunk_size)
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.detach().cpu()))
                if len(epoch_losses) % 50 == 0:
                    # Keeps the allocator cache from growing until the Windows driver spills into system memory.
                    torch.cuda.empty_cache()
                if len(epoch_losses) % 100 == 0:
                    print(
                        f"decoder epoch {epoch + 1} step {len(epoch_losses)} loss {epoch_losses[-1]:.4f}",
                        flush=True,
                    )
            losses.extend(epoch_losses)
            epoch += 1
            print(f"decoder epoch {epoch} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
            _save_checkpoint(
                checkpoint,
                {
                    "protocol_sha256": digest,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "losses": losses,
                    "batch_size": batch_size,
                    "chunk_size": chunk_size,
                    "oom_fallback": oom_fallback,
                },
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            cuda_failure = isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA" in str(error)
            if not cuda_failure:
                raise
            if oom_fallback:
                raise
            oom_fallback = True
            batch_size = spec["oom_batch_fallback"]
            chunk_size = spec["oom_label_chunk_fallback"]
            print(f"decoder OOM, reloading at batch {batch_size} chunk {chunk_size}", flush=True)
            del model, optimizer, scheduler
            if checkpoint.exists():
                checkpoint.unlink()
            torch.cuda.empty_cache()
            model, optimizer, scheduler = build(batch_size, chunk_size)
            losses = []
            epoch = 0
    if epoch != spec["epochs"]:
        raise RuntimeError("refusing to score test from a partial decoder run")
    dev_scores, dev_gold, _ = _decoder_score_items(
        model, tokenizer, items["dev"], labels, spec, device, spec["eval_batch_size"], chunk_size
    )
    fitted = fit_temperature(dev_scores, dev_gold)
    test_scores, test_gold, n_truncated = _decoder_score_items(
        model, tokenizer, items["test"], labels, spec, device, spec["eval_batch_size"], chunk_size
    )
    latency = _decoder_latency(
        model, tokenizer, items, labels, spec, device, protocol["latency"]["warmup"], chunk_size
    )
    if not all(schema_valid(row, len(labels)) for row in apply_temperature(test_scores, fitted["temperature"])):
        raise RuntimeError("decoder readout was not a simplex")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    result = _stamp(
        {
            "system": "ettin-decoder-150m",
            "dataset": "massive_en",
            "task": "intent",
            "slots": "ignored",
            "loss": "log_loss",
            "test_is_full_split": True,
            "subsample": None,
            "train_n": len(rows),
            "dev_n": len(items["dev"]),
            "test_n": len(items["test"]),
            "epochs": spec["epochs"],
            "batch_size": batch_size,
            "label_chunk_size": chunk_size,
            "oom_fallback": oom_fallback,
            "learning_rate": spec["learning_rate"],
            "mean_train_loss": float(np.mean(losses)),
            "temperature": fitted,
            "scaled": _metrics(test_scores, test_gold, fitted["temperature"]),
            "unscaled": _metrics(test_scores, test_gold, 1.0),
            "n_truncated_test": n_truncated,
            "latency": latency,
            "output_tokens": 0,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "device": torch.cuda.get_device_name(0),
        },
        protocol,
    )
    np.savez_compressed(out_dir / "dev_scores.npz", scores=dev_scores, gold=dev_gold)
    np.savez_compressed(out_dir / "test_scores.npz", scores=test_scores, gold=test_gold)
    _write_json(metrics_path, result)
    if checkpoint.exists():
        checkpoint.unlink()
    print(
        f"decoder test acc {result['scaled']['accuracy']:.4f} brier {result['scaled']['brier']:.4f} "
        f"p50 {latency['p50_ms']:.1f} ms",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return result


def finetune(protocol: dict | None = None) -> dict:
    protocol = protocol or load_primary_protocol()
    if not PROBE_PATH.exists():
        raise RuntimeError("refusing to finetune before artifacts/primary/probe.json exists")
    probe_payload = json.loads(PROBE_PATH.read_text())
    if probe_payload.get("protocol_sha256") != protocol_sha256():
        raise RuntimeError("probe.json belongs to a different protocol")
    loaded = load_massive(protocol)
    _require_gpu()
    encoder = finetune_encoder(protocol, loaded, probe_payload["encoder"]["projected_seconds"])
    decoder = finetune_decoder(protocol, loaded, probe_payload["decoder"]["projected_seconds"])
    summary = _stamp({"encoder": encoder, "decoder": decoder, "llm_aggrefact_scored": False}, protocol)
    _write_json(OUT / "finetune_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["probe", "zeroshot", "finetune"])
    args = parser.parse_args()
    protocol = load_primary_protocol()
    if args.command == "probe":
        probe(protocol)
    elif args.command == "zeroshot":
        zeroshot(protocol)
    else:
        finetune(protocol)


if __name__ == "__main__":
    main()

"""Jev as a reference Choice system, called over its HTTP API. Frozen rules are in jev_protocol.json."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature, schema_valid, score_probabilities
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

JEV_PROTOCOL_PATH = ROOT / "jev_protocol.json"
SYSTEM = "jev"
FLOOR = 1e-12


def _api_key(spec: dict) -> str:
    value = os.environ.get(spec["key_env"])
    if not value:
        for line in (ROOT / spec["key_file"]).read_text().splitlines():
            name, _, rest = line.partition("=")
            if name.strip() == spec["key_env"]:
                value = rest.strip().strip("\"'")
    if not value:
        raise RuntimeError(f"{spec['key_env']} is not set in the environment or in {spec['key_file']}")
    return value


def _body(spec: dict, text: str, labels: list[str]) -> dict:
    request = spec["request"]
    return {
        "model": spec["model"],
        "state": text,
        "questions": {
            request["question_id"]: {
                "type": request["type"],
                "instructions": request["instructions"],
                "criteria": {label: None for label in labels},
            }
        },
    }


def _call(client: httpx.Client, spec: dict, text: str, labels: list[str]) -> tuple[dict, float, int]:
    """Return (response json, latency of the answering call in ms, retries)."""
    retry = spec["retry"]
    waits = retry["backoff_seconds"]
    body = _body(spec, text, labels)
    for attempt in range(len(waits) + 1):
        try:
            started = time.perf_counter()
            response = client.post(spec["endpoint"], json=body)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except httpx.TransportError:
            if attempt == len(waits):
                raise
            time.sleep(waits[attempt] + random.random())
            continue
        if response.status_code in retry["statuses"] and attempt < len(waits):
            time.sleep(waits[attempt] + random.random())
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json(), latency_ms, attempt
    raise RuntimeError("retries exhausted")


def _row(client: httpx.Client, spec: dict, item, labels: list[str], dataset: str, split: str) -> dict:
    row = {
        "dataset": dataset,
        "split": split,
        "system": SYSTEM,
        "item_id": item.item_id,
        "row": item.row,
        "gold_index": item.gold_index,
        "gold_label": item.gold_label,
        "scores": None,
        "truncated": False,
        "latency_ms": None,
        "output_tokens": 0,
        "error": None,
    }
    try:
        payload, latency_ms, retries = _call(client, spec, item.text, labels)
        answer = payload["answers"][spec["request"]["question_id"]]
        returned = answer["probabilities"]
        missing = [label for label in labels if label not in returned]
        if missing:
            raise RuntimeError(f"{len(missing)} labels missing from probabilities, first {missing[0]!r}")
        probs = [float(returned[label]) for label in labels]
        if not all(math.isfinite(value) for value in probs):
            raise RuntimeError("non-finite probability")
        row["scores"] = [math.log(max(value, FLOOR)) for value in probs]
        row["latency_ms"] = latency_ms
        row["retries"] = retries
        row["vendor_probability_sum"] = sum(probs)
        row["vendor_choice"] = answer.get("choice")
        row["vendor_confidence"] = answer.get("confidence")
        row["vendor_model"] = payload.get("model")
        row["usage"] = payload.get("usage")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _collect(client, spec, items, labels, dataset, split, path: Path) -> list[dict]:
    done = {row["item_id"] for row in _read(path) if row["error"] is None}
    kept = [row for row in _read(path) if row["error"] is None]
    todo = [item for item in items if item.item_id not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
        with ThreadPoolExecutor(max_workers=spec["concurrency"]) as pool:
            for index, row in enumerate(pool.map(lambda item: _row(client, spec, item, labels, dataset, split), todo), start=1):
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                if index % 100 == 0:
                    print(f"  {SYSTEM} {dataset} {split} {index}/{len(todo)}", flush=True)
    order = {item.item_id: position for position, item in enumerate(items)}
    return sorted(_read(path), key=lambda row: order[row["item_id"]])


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    kept = [row for row in rows if row["scores"] is not None]
    return (
        np.asarray([row["scores"] for row in kept], dtype=np.float64),
        np.asarray([row["gold_index"] for row in kept], dtype=np.int64),
    )


def _scored(scores: np.ndarray, gold: np.ndarray, temperature: float, n_labels: int) -> dict:
    probs = apply_temperature(scores, temperature)
    out = score_probabilities(probs, gold)
    out.pop("risk_coverage", None)
    out.pop("ece_bins", None)
    out["schema_validity_of_scored"] = float(np.mean([schema_valid(row, n_labels) for row in probs]))
    return out


def _latency(rows: list[dict]) -> dict:
    values = np.asarray([row["latency_ms"] for row in rows if row["latency_ms"] is not None], dtype=np.float64)
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "mean_ms": float(values.mean()),
    }


def _replay(client, spec, items, labels, dataset, first: list[dict]) -> dict:
    n = spec["replay"]["n"]
    by_id = {row["item_id"]: row for row in first if row["scores"] is not None}
    flips = 0
    compared = 0
    largest = 0.0
    with ThreadPoolExecutor(max_workers=spec["concurrency"]) as pool:
        again = list(pool.map(lambda item: _row(client, spec, item, labels, dataset, "replay"), items[:n]))
    for row in again:
        before = by_id.get(row["item_id"])
        if before is None or row["scores"] is None:
            continue
        a = np.exp(np.asarray(before["scores"]))
        b = np.exp(np.asarray(row["scores"]))
        compared += 1
        flips += int(np.argmax(a) != np.argmax(b))
        largest = max(largest, float(np.abs(a - b).max()))
    return {"requested": n, "compared": compared, "argmax_flips": flips, "max_abs_probability_change": largest}


def smoke(spec: dict) -> None:
    labels = ["negative", "positive"]
    with httpx.Client(headers={"Authorization": f"Bearer {_api_key(spec)}"}, timeout=60.0) as client:
        payload, latency_ms, retries = _call(client, spec, "The soup was cold and the waiter was rude.", labels)
    print(json.dumps(payload, indent=2))
    print(f"latency_ms {latency_ms:.0f} retries {retries}")


def run(spec: dict) -> dict:
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    out_dir = ROOT / "artifacts" / "jev"
    summary = {"protocol_id": spec["protocol_id"], "datasets": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {_api_key(spec)}"}, timeout=60.0) as client:
        for dataset in spec["datasets"]:
            labels = protocol["datasets"][dataset]["labels"]
            items = materialized["items"][dataset]
            print(f"dev {SYSTEM} {dataset} n={len(items['dev'])}", flush=True)
            dev = _collect(client, spec, items["dev"], labels, dataset, "dev", out_dir / f"{dataset}__dev.jsonl")
            dev_scores, dev_gold = _arrays(dev)
            fitted = fit_temperature(dev_scores, dev_gold)
            print(f"test {SYSTEM} {dataset} n={len(items['test'])}", flush=True)
            test = _collect(client, spec, items["test"], labels, dataset, "test", out_dir / f"{dataset}__test.jsonl")
            scores, gold = _arrays(test)
            ok = [row for row in test if row["scores"] is not None]
            argmax = [labels[int(np.argmax(row["scores"]))] for row in ok]
            summary["datasets"][dataset] = {
                "n_labels": len(labels),
                "dev_n": len(dev),
                "dev_errors": sum(row["error"] is not None for row in dev),
                "test_n": len(test),
                "test_errors": sum(row["error"] is not None for row in test),
                "error_examples": sorted({row["error"][:120] for row in dev + test if row["error"]})[:5],
                "temperature": fitted,
                "scaled": _scored(scores, gold, fitted["temperature"], len(labels)),
                "unscaled": _scored(scores, gold, 1.0, len(labels)),
                "vendor_probability_sum": {
                    "min": min(row["vendor_probability_sum"] for row in ok),
                    "max": max(row["vendor_probability_sum"] for row in ok),
                },
                "vendor_choice_matches_argmax": float(np.mean([row["vendor_choice"] == a for row, a in zip(ok, argmax)])),
                "vendor_models": sorted({str(row["vendor_model"]) for row in ok}),
                "input_tokens": int(sum((row["usage"] or {}).get("input_tokens", 0) for row in dev + test if row.get("usage"))),
                "output_tokens_billed": int(sum((row["usage"] or {}).get("output_tokens", 0) for row in dev + test if row.get("usage"))),
                "retried_rows": sum(bool(row.get("retries")) for row in dev + test),
                "latency": _latency(test),
                "replay": _replay(client, spec, items["test"], labels, dataset, test),
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            scaled = summary["datasets"][dataset]["scaled"]
            print(f"{dataset} acc {scaled['accuracy']:.4f} brier {scaled['brier']:.4f} T {fitted['temperature']:.3f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one synthetic sentence, prints the raw response")
    args = parser.parse_args()
    spec = json.loads(JEV_PROTOCOL_PATH.read_text())
    if args.smoke:
        smoke(spec)
    else:
        run(spec)


if __name__ == "__main__":
    main()

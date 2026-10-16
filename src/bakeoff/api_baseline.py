"""Hosted token-probability baselines on the wider suite. Frozen rules are in api_protocol.json."""

from __future__ import annotations

import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff.protocol import ROOT
from bakeoff.suite import load_suite, load_suite_protocol
from bakeoff.suite_run import _report

API_PATH = ROOT / "api_protocol.json"
OUT = ROOT / "artifacts" / "api"
SPEND_PATH = OUT / "spend.json"
FLOOR = 1e-9
ENDPOINT = json.loads(API_PATH.read_text(encoding="utf-8"))["endpoint"]
_lock = threading.Lock()


class Budget:
    def __init__(self, rules: dict, run_cap: float | None = None):
        self.project_cap = rules["budget"]["project_cap_usd"]
        self.run_cap = run_cap if run_cap is not None else rules["budget"]["this_run_cap_usd"]
        if SPEND_PATH.exists():
            self.total = json.loads(SPEND_PATH.read_text())["total_usd"]
        else:
            self.total = rules["budget"]["spent_before_this_protocol_usd"]
        self.run = 0.0

    def allowed(self) -> bool:
        return self.run < self.run_cap and self.total < self.project_cap

    def add(self, cost: float) -> None:
        with _lock:
            self.total += cost
            self.run += cost

    def save(self) -> None:
        with _lock:
            SPEND_PATH.parent.mkdir(parents=True, exist_ok=True)
            SPEND_PATH.write_text(json.dumps({"total_usd": self.total, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")


def _key(rules: dict) -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        name, _, rest = line.partition("=")
        if name.strip() == rules["key_env"]:
            return rest.strip().strip("\"'")
    raise RuntimeError(f"{rules['key_env']} missing from .env")


def codes_for(n: int) -> list[str]:
    width = len(str(n))
    return [str(position).zfill(width) for position in range(1, n + 1)]


def build_prompt(rules: dict, text: str, labels: list[str]) -> str:
    request = rules["request"]
    options = "\n".join(request["option_line"].format(code=code, label=label) for code, label in zip(codes_for(len(labels)), labels))
    return request["prompt"].format(text=text, options=options)


def tree_distribution(content: list[dict] | None, n: int) -> np.ndarray | None:
    """Option probabilities from the generated tokens and their returned alternatives. None when unscorable."""
    if not content:
        return None
    codes = codes_for(n)
    width = len(codes[0])
    index = {code: position for position, code in enumerate(codes)}
    probs = np.zeros(n, dtype=np.float64)
    prefix, mass = "", 1.0
    for step in content:
        token = step["token"].strip()
        if token == "":
            continue
        alternatives: dict[str, float] = {}
        for entry in step.get("top_logprobs") or []:
            name = entry["token"].strip()
            alternatives[name] = alternatives.get(name, 0.0) + math.exp(entry["logprob"])
        alternatives[token] = max(alternatives.get(token, 0.0), math.exp(step["logprob"]))
        covered: set[str] = set()
        used = 0.0
        for name, p in alternatives.items():
            used += p
            if name == token or name == "":
                continue
            matches = [code for code in codes if code.startswith(prefix + name)] if len(prefix + name) <= width else []
            for code in matches:
                probs[index[code]] += mass * p / len(matches)
            covered.update(matches)
        extended = prefix + token
        rest = [code for code in codes if code.startswith(prefix) and code not in covered and not code.startswith(extended)]
        for code in rest:
            probs[index[code]] += mass * max(0.0, 1.0 - used) / len(rest)
        if len(extended) > width or not any(code.startswith(extended) for code in codes):
            mass = 0.0
            break
        mass *= alternatives[token]
        prefix = extended
        if len(prefix) == width:
            probs[index[prefix]] += mass
            mass = 0.0
            break
    if mass > 0.0 and prefix:
        matches = [code for code in codes if code.startswith(prefix)]
        for code in matches:
            probs[index[code]] += mass / len(matches)
    if prefix == "" or probs.sum() <= 0.0:
        return None
    probs = np.maximum(probs / probs.sum(), FLOOR)
    return probs / probs.sum()


def _post(client, body: dict) -> dict:
    waits = [1, 2, 4, 8]
    for attempt in range(len(waits) + 1):
        try:
            response = client.post(ENDPOINT, json=body)
        except Exception as error:
            if attempt == len(waits):
                return {"error": f"{type(error).__name__}: {str(error)[:120]}"}
            time.sleep(waits[attempt] + random.random())
            continue
        if response.status_code in (408, 429) or response.status_code >= 500:
            if attempt == len(waits):
                return {"error": f"HTTP {response.status_code}: {response.text[:160]}"}
            time.sleep(waits[attempt] + random.random())
            continue
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        payload = response.json()
        if "choices" not in payload:
            return {"error": f"no choices: {json.dumps(payload)[:160]}"}
        return payload
    return {"error": "retries exhausted"}


def _row(client, rules: dict, system: dict, item, labels: list[str], budget: Budget) -> dict:
    row = {"item_id": item.item_id, "gold_index": item.gold_index, "error": None}
    if not budget.allowed():
        row["error"] = "budget cap reached"
        return row
    body = {
        "model": system["model"],
        "messages": [{"role": "user", "content": build_prompt(rules, item.text, labels)}],
        "temperature": rules["request"]["temperature"],
        "max_tokens": rules["request"]["max_tokens"],
        "logprobs": True,
        "top_logprobs": system["top_logprobs"],
        "reasoning": system["reasoning"],
        "usage": {"include": True},
        "provider": {"order": [system["provider"]], "allow_fallbacks": False, "require_parameters": True},
    }
    started = time.perf_counter()
    payload = _post(client, body)
    row["latency_ms"] = (time.perf_counter() - started) * 1000.0
    if "error" in payload:
        row["error"] = payload["error"]
        return row
    choice = payload["choices"][0]
    usage = payload.get("usage") or {}
    budget.add(float(usage.get("cost") or 0.0))
    content = (choice.get("logprobs") or {}).get("content")
    row.update(
        {
            "text": choice["message"].get("content") or "",
            "provider": payload.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost": float(usage.get("cost") or 0.0),
            "content": [
                {"token": step["token"], "logprob": step["logprob"], "top_logprobs": [{"token": a["token"], "logprob": a["logprob"]} for a in step.get("top_logprobs") or []]}
                for step in (content or [])
            ],
        }
    )
    return row


def _read(path) -> dict:
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            rows[row["item_id"]] = row
    return rows


def collect(client, rules, system, items, labels, path, budget: Budget) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        done = {key: row for key, row in _read(path).items() if row["error"] is None}
        todo = [item for item in items if item.item_id not in done]
        if not todo or not budget.allowed():
            break
        with path.open("w", encoding="utf-8") as handle:
            for row in done.values():
                handle.write(json.dumps(row) + "\n")
            with ThreadPoolExecutor(max_workers=8) as pool:
                for count, row in enumerate(pool.map(lambda item: _row(client, rules, system, item, labels, budget), todo), start=1):
                    handle.write(json.dumps(row) + "\n")
                    if count % 500 == 0:
                        handle.flush()
                        budget.save()
                        print(f"    {count}/{len(todo)} run ${budget.run:.3f} total ${budget.total:.3f}", flush=True)
        budget.save()
    rows = _read(path)
    return [rows.get(item.item_id, {"item_id": item.item_id, "error": "not requested"}) for item in items]


def scores_from(rows: list[dict], n: int) -> tuple[np.ndarray, dict]:
    uniform = np.full(n, 1.0 / n)
    out = []
    counts = {"n": len(rows), "request_errors": 0, "unscorable": 0}
    for row in rows:
        probs = None if row.get("error") else tree_distribution(row.get("content"), n)
        if row.get("error"):
            counts["request_errors"] += 1
        if probs is None:
            counts["unscorable"] += 1
            probs = uniform
        out.append(np.log(probs))
    return np.stack(out, axis=0), counts


def run(name: str, limit: int | None = None, run_cap: float | None = None) -> None:
    import httpx

    rules = json.loads(API_PATH.read_text(encoding="utf-8"))
    system = rules["systems"][name]
    budget = Budget(rules, run_cap)
    suite = load_suite()
    out = OUT / name
    summary_path = out / ("summary.json" if limit is None else "dev_check.json")
    summary = {"protocol_id": rules["protocol_id"], "system": name, "suite_protocol_id": load_suite_protocol()["protocol_id"], "datasets": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {_key(rules)}"}, timeout=90.0) as client:
        for dataset, block in suite.items():
            labels = block["labels"]
            splits = ["dev"] if limit else [part for part in ("dev", "test", "oos") if part in block]
            scores, failures, latency, providers, tokens = {}, {}, [], set(), 0
            for split in splits:
                items = block[split][:limit] if limit else block[split]
                print(f"  {name} {dataset} {split} n={len(items)}", flush=True)
                rows = collect(client, rules, system, items, labels, out / (f"{dataset}__{split}.jsonl" if limit is None else f"check__{dataset}.jsonl"), budget)
                scores[split], failures[split] = scores_from(rows, len(labels))
                providers.update(str(row.get("provider")) for row in rows if not row.get("error"))
                tokens += sum(row.get("prompt_tokens") or 0 for row in rows)
                if split == "test":
                    latency = [row["latency_ms"] for row in rows if row.get("latency_ms")]
            if limit:
                gold = np.asarray([item.gold_index for item in block["dev"][:limit]], dtype=np.int64)
                accuracy = float((scores["dev"].argmax(axis=1) == gold).mean())
                summary["datasets"][dataset] = {"dev_check_n": int(len(gold)), "argmax_accuracy": accuracy, "failures": failures["dev"]}
                print(f"{name} {dataset}: dev check acc {accuracy:.3f} failures {failures['dev']}", flush=True)
            else:
                if any(f["request_errors"] for f in failures.values()) and not budget.allowed():
                    raise RuntimeError(f"budget cap reached with {dataset} incomplete; run ${budget.run:.2f} total ${budget.total:.2f}")
                extra = {
                    "readout": "top-k tree",
                    "failures": failures,
                    "providers": sorted(providers),
                    "prompt_tokens": int(tokens),
                    "latency_p50_ms": float(np.percentile(latency, 50)) if latency else None,
                    "latency_note": "HTTPS round trip through OpenRouter, not comparable to local GPU",
                }
                summary["datasets"][dataset] = _report(name, dataset, block, scores, extra)
            summary["spend_usd"] = {"this_run": budget.run, "project_total": budget.total}
            out.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"API_DONE {name} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        run(sys.argv[2], limit=15, run_cap=0.25)
    elif len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        raise SystemExit("usage: python -m bakeoff.api_baseline check|run <system>")

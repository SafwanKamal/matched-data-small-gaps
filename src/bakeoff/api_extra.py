"""Stated-probability baselines on a slice, and the hosted models on the context pilot.

Frozen rules are in api_verbalized_protocol.json; shared request code is in api_baseline.py.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff import api_baseline as base
from bakeoff import context_pilot
from bakeoff.protocol import ROOT
from bakeoff.suite import load_suite
from bakeoff.suite_run import _report

RULES_PATH = ROOT / "api_verbalized_protocol.json"


def stated_distribution(text: str, n: int) -> tuple[np.ndarray | None, int, float]:
    """Option probabilities from a JSON reply. Returns (probs or None, options listed, stated sum)."""
    match = re.search(r"\{.*?\}", text or "", re.S)
    if not match:
        return None, 0, 0.0
    try:
        raw = json.loads(match.group())
    except ValueError:
        return None, 0, 0.0
    codes = base.codes_for(n)
    width = len(codes[0])
    index = {code: position for position, code in enumerate(codes)}
    stated: dict[int, float] = {}
    if not isinstance(raw, dict):
        return None, 0, 0.0
    for key, value in raw.items():
        code = str(key).strip().zfill(width)
        if code in index and isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            stated[index[code]] = stated.get(index[code], 0.0) + float(value)
    total = sum(stated.values())
    if not stated or total <= 0:
        return None, len(stated), total
    probs = np.zeros(n, dtype=np.float64)
    for position, value in stated.items():
        probs[position] = value / total if total > 1.0 else value
    unlisted = [position for position in range(n) if position not in stated]
    if total < 1.0 and unlisted:
        probs[unlisted] = (1.0 - total) / len(unlisted)
    probs = np.maximum(probs / probs.sum(), base.FLOOR)
    return probs / probs.sum(), len(stated), total


def _slice(rules: dict, suite: dict) -> dict:
    path = base.OUT / "verbalized_slice.json"
    if path.exists():
        chosen = json.loads(path.read_text())
    else:
        spec = rules["verbalized"]["slice"]
        children = np.random.SeedSequence(spec["seed"]).spawn(2 * len(suite))
        chosen = {}
        for position, (dataset, block) in enumerate(suite.items()):
            picks = {}
            for offset, (split, count) in enumerate((("dev", spec["dev_n"]), ("test", spec["test_n"]))):
                rng = np.random.default_rng(children[2 * position + offset])
                order = rng.permutation(len(block[split]))[:count]
                picks[split] = [block[split][int(i)].item_id for i in sorted(order.tolist())]
            chosen[dataset] = picks
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(chosen) + "\n")
    out = {}
    for dataset, block in suite.items():
        out[dataset] = {"labels": block["labels"]}
        for split in ("dev", "test"):
            wanted = set(chosen[dataset][split])
            out[dataset][split] = [item for item in block[split] if item.item_id in wanted]
    return out


def _verbal_row(client, rules: dict, system: dict, item, labels: list[str], budget: base.Budget) -> dict:
    row = {"item_id": item.item_id, "gold_index": item.gold_index, "error": None}
    if not budget.allowed():
        row["error"] = "budget cap reached"
        return row
    spec = rules["verbalized"]
    codes = base.codes_for(len(labels))
    options = "\n".join(spec["option_line"].format(code=code, label=label) for code, label in zip(codes, labels))
    example = json.dumps({codes[min(2, len(codes) - 1)]: 0.7, codes[0]: 0.3})
    body = {
        "model": system["model"],
        "messages": [{"role": "user", "content": spec["prompt"].format(text=item.text, options=options, example=example)}],
        "temperature": spec["temperature_setting"],
        "max_tokens": system["max_tokens"],
        "reasoning": system["reasoning"],
        "usage": {"include": True},
        "provider": {"order": [system["provider"]], "allow_fallbacks": False},
    }
    started = time.perf_counter()
    payload = base._post(client, body)
    row["latency_ms"] = (time.perf_counter() - started) * 1000.0
    if "error" in payload:
        row["error"] = payload["error"]
        return row
    usage = payload.get("usage") or {}
    budget.add(float(usage.get("cost") or 0.0))
    row.update(
        {
            "text": payload["choices"][0]["message"].get("content") or "",
            "provider": payload.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": float(usage.get("cost") or 0.0),
        }
    )
    return row


def _collect(client, worker, items, path, budget: base.Budget) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        done = {key: row for key, row in base._read(path).items() if row["error"] is None}
        todo = [item for item in items if item.item_id not in done]
        if not todo or not budget.allowed():
            break
        with path.open("w", encoding="utf-8") as handle:
            for row in done.values():
                handle.write(json.dumps(row) + "\n")
            with ThreadPoolExecutor(max_workers=8) as pool:
                for row in pool.map(worker, todo):
                    handle.write(json.dumps(row) + "\n")
        budget.save()
    rows = base._read(path)
    return [rows.get(item.item_id, {"item_id": item.item_id, "error": "not requested"}) for item in items]


def run_verbalized(name: str, limit: int | None = None) -> None:
    import httpx

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    main_rules = json.loads(base.API_PATH.read_text(encoding="utf-8"))
    system = rules["verbalized"]["systems"][name]
    budget = base.Budget(main_rules, 0.3 if limit else rules["budget"]["this_run_cap_usd"])
    sliced = _slice(rules, load_suite())
    if "sub_slice" in system:
        # A seeded subset of the fixed slice, for a model too costly to run on all of it.
        sub = system["sub_slice"]
        children = np.random.SeedSequence(sub["seed"]).spawn(2 * len(sliced))
        for position, block in enumerate(sliced.values()):
            for offset, (split, count) in enumerate((("dev", sub["dev_n"]), ("test", sub["test_n"]))):
                keep = sorted(np.random.default_rng(children[2 * position + offset]).permutation(len(block[split]))[:count].tolist())
                block[split] = [block[split][index] for index in keep]
    out = base.OUT / name
    summary = {"protocol_id": rules["protocol_id"], "system": name, "readout": "stated probabilities", "datasets": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {base._key(main_rules)}"}, timeout=120.0) as client:
        for dataset, block in sliced.items():
            labels = block["labels"]
            scores, info = {}, {}
            for split in (("dev",) if limit else ("dev", "test")):
                items = block[split][:limit] if limit else block[split]
                path = out / (f"check__{dataset}.jsonl" if limit else f"{dataset}__{split}.jsonl")
                rows = _collect(client, lambda item: _verbal_row(client, rules, system, item, labels, budget), items, path, budget)
                uniform = np.full(len(labels), 1.0 / len(labels))
                values, listed, sums, bad = [], [], [], 0
                for row in rows:
                    probs, count, total = (None, 0, 0.0) if row.get("error") else stated_distribution(row.get("text", ""), len(labels))
                    if probs is None:
                        bad += 1
                        probs = uniform
                    else:
                        listed.append(count)
                        sums.append(total)
                    values.append(np.log(probs))
                scores[split] = np.stack(values, axis=0)
                info[split] = {
                    "n": len(rows),
                    "request_errors": sum(bool(row.get("error")) for row in rows),
                    "unscorable": bad,
                    "mean_options_listed": float(np.mean(listed)) if listed else None,
                    "mean_stated_sum": float(np.mean(sums)) if sums else None,
                    "mean_completion_tokens": float(np.mean([row.get("completion_tokens") or 0 for row in rows])),
                    "providers": sorted({str(row.get("provider")) for row in rows if not row.get("error")}),
                }
            if limit:
                gold = np.asarray([item.gold_index for item in block["dev"][:limit]], dtype=np.int64)
                print(f"{name} {dataset}: dev check acc {float((scores['dev'].argmax(axis=1) == gold).mean()):.3f} {info['dev']}", flush=True)
                continue
            if info["test"]["request_errors"] and not budget.allowed():
                raise RuntimeError(f"budget cap reached with {dataset} incomplete")
            summary["datasets"][dataset] = _report(name, dataset, block, scores, {"readout": "stated probabilities", "details": info})
            summary["spend_usd"] = {"this_run": budget.run, "project_total": budget.total}
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"API_DONE {name} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


def run_pilot(name: str) -> None:
    import httpx

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    main_rules = json.loads(base.API_PATH.read_text(encoding="utf-8"))
    system = main_rules["systems"][name]
    budget = base.Budget(main_rules, rules["budget"]["this_run_cap_usd"])
    pilot = json.loads(context_pilot.PILOT_PATH.read_text(encoding="utf-8"))
    data = context_pilot.build(pilot)
    suite_summary = json.loads((base.OUT / name / "summary.json").read_text())["datasets"]
    temperatures = {"banking77": suite_summary["banking77_full"]["temperature"]["temperature"], "clinc150": suite_summary["clinc150"]["temperature"]["temperature"]}
    with httpx.Client(headers={"Authorization": f"Bearer {base._key(main_rules)}"}, timeout=90.0) as client:
        for dataset, block in data.items():
            labels = block["labels"]
            gold = np.asarray([item.gold_index for item in block["conditions"]["clean"]], dtype=np.int64)
            scores, extra = {}, {}
            for condition, items in block["conditions"].items():
                rows = base.collect(client, main_rules, system, items, labels, base.OUT / name / "context-pilot" / f"{dataset}__{condition}.jsonl", budget)
                scores[condition], failures = base.scores_from(rows, len(labels))
                extra[condition] = {"unscorable": failures["unscorable"]}
            if not budget.allowed():
                raise RuntimeError("budget cap reached during the context pilot")
            context_pilot._save(name, dataset, context_pilot._report(name, dataset, scores, temperatures[dataset], gold, extra))
    budget.save()
    print(f"API_DONE pilot {name} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 2 else ""
    if command == "verbalized-check":
        run_verbalized(sys.argv[2], limit=8)
    elif command == "verbalized":
        run_verbalized(sys.argv[2])
    elif command == "pilot":
        run_pilot(sys.argv[2])
    else:
        raise SystemExit("usage: python -m bakeoff.api_extra verbalized-check|verbalized|pilot <system>")

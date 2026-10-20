"""Reasoning on: run D2's BANKING77-OOS retrieval test with reasoning at low effort (reasoning_on_protocol.json).

    python -m bakeoff.reasoning_on check|hosted|analyze [model]

Same items, prompts, hosts and parsing as run D2 (bakeoff.retrieval_mechanism). The reasoning-off reference is run D2's
saved replies, not re-requested.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import retrieval_mechanism as rm
from bakeoff.protocol import ROOT

PROTOCOL = ROOT / "reasoning_on_protocol.json"
OUT = ROOT / "artifacts" / "reasoning-on"
SPLIT = "banking77_oos"
ARMS = ["zero_shot", "retrieval"]
MAX_TOKENS = 4000


def spec() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _path(model: str, arm: str, folder: str):
    return OUT / folder / f"{model}__{SPLIT}__{arm}.jsonl"


def run(stage: str, only: str | None = None) -> None:
    import httpx

    from bakeoff import api_baseline as api
    from bakeoff import workflow_v4 as v4

    s = spec()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    data = rm.load()
    folder = "check" if stage == "check" else "hosted"
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=300.0) as client:
        for name in [only] if only else s["model_order"]:
            cap = s["cost"]["check_cap_usd"] if stage == "check" else s["cost"]["per_model_cap_usd"][name]
            budget = api.Budget(api_rules, run_cap=cap)
            model = {**s["models"][name], "name": name}
            items = rm._check_items(data, SPLIT) if stage == "check" else data[SPLIT]["items"]
            for arm in ARMS:
                js = rm.jobs(data, SPLIT, arm, items, rm.rules()["seed"])
                path = _path(name, arm, folder)
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    rows = v4._collect(client, model, [(a, b, c) for a, b, c, _ in js], MAX_TOKENS, path, budget)
                except RuntimeError as error:
                    print(f"    {error}", flush=True)
                    rows = v4._read(path)
                parsed = [v4.parse(r["text"], len(js[0][3])) for r in rows.values()]
                rt = [r.get("reasoning_tokens") or 0 for r in rows.values()]
                print(f"{stage} {name:14s} {arm:10s} {len(rows)}/{len(items)} parse_failed {sum(p['parse_failed'] for p in parsed)} "
                      f"reasoning_tokens mean {np.mean(rt) if rt else 0:.0f} max {max(rt) if rt else 0} providers {sorted({str(r.get('provider')) for r in rows.values()})} "
                      f"run ${budget.run:.4f} project ${budget.total:.3f}", flush=True)
                budget.save()
                if not budget.allowed():
                    print("cap reached; the rest is not run", flush=True)
                    break
    print(f"R_{stage.upper()}_DONE", flush=True)


def analyze() -> None:
    from bakeoff import posthoc_round3 as p3
    from bakeoff import retrieval_suppression as rs
    from bakeoff import revision_free as rf
    from bakeoff import workflow_v4 as v4

    s = spec()
    seed = s["seed"]
    b = rm.load()[SPLIT]
    items = b["items"]
    n_labels = len(b["labels"])
    in_scope = np.asarray([i["in_scope"] for i in items])
    gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
    clusters = np.asarray([i["cluster"] for i in items])

    def read(path):
        rows = v4._read(path) if path.exists() else {}
        scope, intent, failed = [], [], 0
        for i in items:
            r = rows.get(i["id"])
            p = v4.parse(r["text"], n_labels) if r is not None else {"scope": not i["in_scope"], "intent": -1, "parse_failed": True}
            failed += bool(p.get("parse_failed"))
            scope.append(p["scope"])
            intent.append(p["intent"])
        return np.asarray(scope, bool), np.asarray(intent), failed, rows

    def summary(scope, intent, failed, rows):
        d, c = p3.sdt(~scope, in_scope)
        num = lambda k: [float(r[k]) for r in rows.values() if r.get(k) is not None]
        return {"held_out_rejected": float((~scope)[~in_scope].mean()), "in_scope_kept": float(scope[in_scope].mean()),
                "in_scope_right": float((scope & (intent == gold))[in_scope].mean()), "d_prime": d, "criterion": c,
                "parse_failed": failed, "replies": len(rows),
                "mean_reasoning_tokens": float(np.mean(num("reasoning_tokens"))) if num("reasoning_tokens") else 0.0,
                "mean_completion_tokens": float(np.mean(num("completion_tokens"))) if num("completion_tokens") else 0.0,
                "mean_cost_usd": float(np.mean(num("cost"))) if num("cost") else 0.0,
                "median_latency_ms": float(np.median(num("latency_ms"))) if num("latency_ms") else 0.0}

    out = {"protocol_id": s["protocol_id"], "models": {}, "tests": {}}
    p_r1 = {}
    for name in s.get("models_analyzed", s["model_order"]):  # amendment of 2026-10-20: Gemma only
        if not _path(name, "zero_shot", "hosted").exists():
            continue
        on = {arm: read(_path(name, arm, "hosted")) for arm in ARMS}
        off = {arm: read(rm._path(name, SPLIT, arm)) for arm in ARMS}
        out["models"][name] = {"on": {arm: summary(*on[arm]) for arm in ARMS}, "off": {arm: summary(*off[arm]) for arm in ARMS}}
        rej = {(k, arm): (~d[arm][0])[~in_scope].astype(float) for k, d in (("on", on), ("off", off)) for arm in ARMS}
        right = {(k, arm): (d[arm][0] & (d[arm][1] == gold))[in_scope].astype(float) for k, d in (("on", on), ("off", off)) for arm in ARMS}
        t = rs.flip_test(rej["on", "zero_shot"] - rej["on", "retrieval"], clusters[~in_scope], seed, True)
        out["tests"][f"R1 {name}"] = t
        p_r1[name] = t["p"]
        for arm in ARMS:
            out["tests"][f"S1 {name} {arm}: rejection on minus off"] = rs.flip_test(rej["on", arm] - rej["off", arm], clusters[~in_scope], seed, False)
            out["tests"][f"S3 {name} {arm}: in-scope right on minus off"] = rs.flip_test(right["on", arm] - right["off", arm], gold[in_scope], seed, False)
        drop_on = rej["on", "zero_shot"] - rej["on", "retrieval"]
        drop_off = rej["off", "zero_shot"] - rej["off", "retrieval"]
        out["tests"][f"S2 {name}: drop on minus drop off"] = rs.flip_test(drop_on - drop_off, clusters[~in_scope], seed, False)
    for name, p in rf.holm(p_r1).items():
        out["tests"][f"R1 {name}"]["holm_p"] = p
    sig = [n for n in p_r1 if out["tests"][f"R1 {n}"]["holm_p"] < 0.05]
    out["R1"] = {"models_run": list(p_r1), "significant": sig, "holds": bool(p_r1) and len(sig) == len(p_r1)}  # all analyzed models; Holm over one test is the test itself
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in out["tests"].items():
        print(f"  {k:52s} {v['difference_points']:+7.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] p={v['p']:.4f}" + (f" holm={v['holm_p']:.4f}" if "holm_p" in v else ""), flush=True)
    for name, m in out["models"].items():
        for k in ("off", "on"):
            for arm in ARMS:
                x = m[k][arm]
                print(f"  {name:12s} {k:3s} {arm:10s} rejected {100 * x['held_out_rejected']:5.1f} kept {100 * x['in_scope_kept']:5.1f} right {100 * x['in_scope_right']:5.1f} "
                      f"d' {x['d_prime']:.2f} c {x['criterion']:+.2f} parse_failed {x['parse_failed']} reasoning {x['mean_reasoning_tokens']:.0f} cost ${x['mean_cost_usd']:.6f} p50 {x['median_latency_ms']:.0f} ms", flush=True)
    print("R_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    analyze() if stage == "analyze" else run(stage, sys.argv[2] if len(sys.argv) > 2 else None)

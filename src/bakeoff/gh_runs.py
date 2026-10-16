"""Runs G (order of retrieved examples) and H (contamination probe). Rules: gh_protocol.json.

Stages: 'check', 'G', 'H', 'analyze'.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "gh_protocol.json"
OUT = ROOT / "artifacts" / "gh"
NEWLINE = "\n"
G_SETS = ["clinc150", "massive"]
SEED = 20261019


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _models() -> dict:
    rs = json.loads((ROOT / "retrieval_suppression_protocol.json").read_text(encoding="utf-8"))["models"]
    return {k: {**rs[k], "name": k} for k in ("gemma-4-31b", "qwen3.5-27b")}


# ---------------------------------------------------------------- G


def g_jobs(name: str, order: str, limit: int | None = None) -> list[tuple[str, str, str]]:
    from bakeoff import confirmatory as cf
    from bakeoff.workflow_retrieval import BM25

    d = cf.load()[name]
    turns = cf.turns_of(d)[:limit] if limit else cf.turns_of(d)
    pool = cf._pool(d, False)
    index = BM25([t for t, _ in pool])
    system = cf._system(d)
    jobs = []
    for t in turns:
        picked = index.top(t["text"], 20)  # most similar first
        if order == "G2":
            rng = np.random.default_rng([SEED, int(hashlib.sha256(t["id"].encode()).hexdigest()[:8], 16)])
            picked = [picked[i] for i in rng.permutation(len(picked))]
        lines = NEWLINE.join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
        jobs.append((t["id"], system, "Similar labelled examples (message => code):" + NEWLINE + lines + NEWLINE + NEWLINE + "Current message: " + t["text"]))
    return jobs


# ---------------------------------------------------------------- H


def h_items() -> list[dict]:
    from bakeoff import workflow as v1
    from bakeoff import workflow_clinc as c

    _, b_test, _, _, _ = v1._sources()
    cdata = c._data(json.loads(c.RULES_PATH.read_text(encoding="utf-8")))
    rng = np.random.default_rng(SEED)
    out = []
    for name, items in (("banking77", b_test), ("clinc150", cdata["test_in"])):
        long = [it for it in items if len(it.text.split()) >= 8]
        for k in sorted(rng.choice(len(long), size=200, replace=False).tolist()):
            it = long[k]
            words = it.text.split()
            half = len(words) // 2
            out.append({"id": f"{name}-{k}", "set": name, "intent": f"{name}-{it.gold_index}", "prefix": " ".join(words[:half]), "suffix": " ".join(words[half:])})
    # Siblings: other sampled test utterances of the same intent, from the full split.
    by_intent: dict[str, list[str]] = {}
    for name, items in (("banking77", b_test), ("clinc150", cdata["test_in"])):
        for it in items:
            w = it.text.split()
            by_intent.setdefault(f"{name}-{it.gold_index}", []).append(" ".join(w[len(w) // 2 :]))
    for item in out:
        siblings = [s for s in by_intent[item["intent"]] if s != item["suffix"]]
        item["siblings"] = siblings[:20]
    return out


H_SYSTEM = "You complete text exactly as it appears in its source. Reply with the rest of the text only."


def h_jobs(limit: int | None = None) -> list[tuple[str, str, str]]:
    items = h_items()[:limit] if limit else h_items()
    return [(i["id"], H_SYSTEM, "Complete this customer message: " + i["prefix"]) for i in items]


# ---------------------------------------------------------------- requests


def run(stage: str) -> None:
    import httpx

    from bakeoff import api_baseline as api
    from bakeoff import workflow_v4 as v4

    spec = rules()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    budget = api.Budget(api_rules, run_cap=0.05 if stage == "check" else spec["cost"]["run_cap_usd"])
    models = _models()
    folder = OUT / ("check" if stage == "check" else "replies")
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client:
        if stage in ("check", "G"):
            for name in G_SETS:
                for order in ("G1", "G2"):
                    js = g_jobs(name, order, 5 if stage == "check" else None)
                    rows = v4._collect(client, models["gemma-4-31b"], js, 24, folder / f"G__{name}__{order}.jsonl", budget)
                    print(f"{stage} G {name} {order} {len(rows)}/{len(js)} run ${budget.run:.3f}", flush=True)
                    budget.save()
        if stage in ("check", "H"):
            for model in ("gemma-4-31b", "qwen3.5-27b"):
                js = h_jobs(5 if stage == "check" else None)
                rows = v4._collect(client, models[model], js, 40, folder / f"H__{model}.jsonl", budget)
                print(f"{stage} H {model} {len(rows)}/{len(js)} sample {next(iter(rows.values()))['text'][:60]!r} run ${budget.run:.3f}", flush=True)
                budget.save()
    print(f"GH_{stage}_DONE run ${budget.run:.3f}", flush=True)


# ---------------------------------------------------------------- analysis


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _f1(a: str, b: str) -> float:
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    overlap = sum((ta & tb).values())
    if not overlap:
        return 0.0
    p, r = overlap / sum(ta.values()), overlap / sum(tb.values())
    return 2 * p * r / (p + r)


def analyze() -> None:
    from bakeoff import confirmatory as cf
    from bakeoff import posthoc_round3b as pb
    from bakeoff import retrieval_suppression as rs
    from bakeoff import revision_free as rf
    from bakeoff import workflow_v4 as v4

    result = {"protocol_id": rules()["protocol_id"], "G": {}, "H": {}}
    seen = pb.seen_texts()
    data = cf.load()
    for name in G_SETS:
        d = data[name]
        turns = cf.turns_of(d)
        in_scope = np.asarray([t["in_scope"] for t in turns])
        clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
        fresh = np.asarray([t["text"] not in seen for t in turns])
        tc = lambda sc, it: rf.turn_correct(turns, sc, it).astype(float)
        sp = {s: rf.specialist(cf.features(name, s), 0.2) for s in cf.SEEDS}
        c_spec = np.mean([tc(sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
        orders = {"A": cf._hosted_answers(cf.OUT / "hosted" / f"A__{name}__gemma-4-31b.jsonl", turns, len(d["labels"]))}
        for o in ("G1", "G2"):
            orders[o] = cf._hosted_answers(OUT / "replies" / f"G__{name}__{o}.jsonl", turns, len(d["labels"]))
        block = {"orders": {}}
        for o, a in orders.items():
            c = tc(a["scope"], a["intent"])
            changed = float(np.mean([(a["scope"][k], a["intent"][k]) != (orders["A"]["scope"][k], orders["A"]["intent"][k]) for k in range(len(turns))]))
            block["orders"][o] = {
                "weighted": cf.weighted(c, in_scope),
                "weighted_never_seen": cf.weighted(c[fresh], in_scope[fresh]),
                "in_scope_kept": float(a["scope"][in_scope].mean()),
                "out_of_scope_rejected": float((~a["scope"])[~in_scope].mean()),
                "share_changed_vs_A": changed,
                "failed": a["failed"],
                "specialist_minus_gemma": cf.weighted_test(c_spec, c, in_scope, clusters, SEED),
            }
        w = [v["weighted"] for v in block["orders"].values()]
        block["weighted_range_points"] = 100 * (max(w) - min(w))
        block["weighted_sd_points"] = 100 * float(np.std(w))
        result["G"][name] = block
    items = h_items()
    for model in ("gemma-4-31b", "qwen3.5-27b"):
        rows = v4._read(OUT / "replies" / f"H__{model}.jsonl")
        own, sib, exact, clusters = [], [], [], []
        for it in items:
            r = rows.get(it["id"])
            text = r["text"].strip() if r else ""
            own.append(_f1(text, it["suffix"]))
            sib.append(float(np.mean([_f1(text, s) for s in it["siblings"]])) if it["siblings"] else 0.0)
            exact.append(" ".join(_tokens(text)) == " ".join(_tokens(it["suffix"])))
            clusters.append(it["intent"])
        own, sib, exact, clusters = np.asarray(own), np.asarray(sib), np.asarray(exact), np.asarray(clusters)
        sets = np.asarray([it["set"] for it in items])
        result["H"][model] = {}
        for s in ("banking77", "clinc150"):
            m = sets == s
            t = rs.flip_test(own[m] - sib[m], clusters[m], SEED, False)
            result["H"][model][s] = {"exact_share": float(exact[m].mean()), "f1_own": float(own[m].mean()), "f1_sibling": float(sib[m].mean()), "own_minus_sibling": t, "memorisation_indicated": bool(t["p"] < 0.05 and t["difference_points"] > 0 and exact[m].mean() > 0.05)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8")
    for name, b in result["G"].items():
        print(f"G {name}: range {b['weighted_range_points']:.2f} points, SD {b['weighted_sd_points']:.2f}")
        for o, v in b["orders"].items():
            t = v["specialist_minus_gemma"]
            print(f"   {o}: weighted {100 * v['weighted']:.2f} (never seen {100 * v['weighted_never_seen']:.2f}) kept {100 * v['in_scope_kept']:.1f} rejected {100 * v['out_of_scope_rejected']:.1f} changed {100 * v['share_changed_vs_A']:.1f}% | spec-gemma {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}")
    for model, b in result["H"].items():
        for s, v in b.items():
            t = v["own_minus_sibling"]
            print(f"H {model} {s}: exact {100 * v['exact_share']:.1f}% F1 own {v['f1_own']:.3f} sibling {v['f1_sibling']:.3f} diff {t['difference_points']:+.1f} p={t['p']:.4f} memorisation {v['memorisation_indicated']}")
    print("GH_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    analyze() if stage == "analyze" else run(stage)

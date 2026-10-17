"""POST HOC analyses from the round 4 internal review. Nothing here was pre-registered.

Writes artifacts/posthoc/round4.json and prints a summary.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.stats import norm

from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "posthoc"
SEED = 20261016


def _rates(reject, in_scope):
    held, inside = reject[~in_scope], reject[in_scope]
    h = np.clip(held.mean(), 0.5 / len(held), 1 - 0.5 / len(held))
    f = np.clip(inside.mean(), 0.5 / len(inside), 1 - 0.5 / len(inside))
    return h, f


def d_a(reject, in_scope, slope: float) -> float:
    """Unequal-variance discrimination d_a for a zROC slope s (Macmillan and Creelman 2005)."""
    h, f = _rates(reject, in_scope)
    return float(np.sqrt(2 / (1 + slope**2)) * (norm.ppf(h) - slope * norm.ppf(f)))


def sibling_absorption() -> dict:
    """D2 BANKING77-OOS: where accepted held-out requests go, and the drop by concentration of the retrieved labels."""
    from bakeoff import retrieval_mechanism as rm
    from bakeoff import workflow_v4 as v4
    from bakeoff.workflow_retrieval import BM25

    data = rm.load()["banking77_oos"]
    items = data["items"]
    labels = data["labels"]
    pool = data["pool"]
    index = BM25([t for t, _ in pool])
    neigh = {i["id"]: [pool[j][1] for j in index.top(i["text"], 20)] for i in items}
    out = {"models": {}}
    held = [i for i in items if not i["in_scope"]]
    conc = np.asarray([Counter(neigh[i["id"]]).most_common(1)[0][1] / 20 for i in held])
    median = float(np.median(conc))
    for model in ("gemma-4-31b", "qwen3.5-27b", "llama-3.3-70b", "deepseek-v4-pro"):
        ret = v4._read(rm._path(model, "banking77_oos", "retrieval"))
        zs = v4._read(rm._path(model, "banking77_oos", "zero_shot"))
        p_ret = {k: v4.parse(r["text"], len(labels)) for k, r in ret.items()}
        p_zs = {k: v4.parse(r["text"], len(labels)) for k, r in zs.items()}
        accepted = [i for i in held if p_ret[i["id"]]["scope"]]
        in_neigh = np.mean([p_ret[i["id"]]["intent"] in neigh[i["id"]] for i in accepted])
        top1 = np.mean([p_ret[i["id"]]["intent"] == neigh[i["id"]][0] for i in accepted])
        mapping = Counter((i["cluster"][5:], labels[p_ret[i["id"]]["intent"]] if 0 <= p_ret[i["id"]]["intent"] < len(labels) else "?") for i in accepted)
        rej = lambda p, sel: float(np.mean([not p[i["id"]]["scope"] for i, s in zip(held, sel) if s]))
        hi, lo = conc > median, conc <= median
        per_intent = {}
        for name in sorted({i["cluster"][5:] for i in held}):
            sel = [i["cluster"][5:] == name for i in held]
            per_intent[name] = (rej(p_zs, sel), rej(p_ret, sel))
        out["models"][model] = {
            "accepted_held_out": len(accepted),
            "chosen_label_among_retrieved": float(in_neigh),
            "chosen_label_is_top1_retrieved": float(top1),
            "top_mappings": [(a, b, n) for (a, b), n in mapping.most_common(8)],
            "all_mappings": [(a, b, n) for (a, b), n in sorted(mapping.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1]))],
            "drop_concentrated": (rej(p_zs, hi), rej(p_ret, hi)),
            "drop_diffuse": (rej(p_zs, lo), rej(p_ret, lo)),
            "per_intent_zero_shot_to_retrieval": per_intent,
        }
    out["median_top_label_share"] = median
    return out


def freshness_components() -> dict:
    """Confirmatory specialist - Gemma split into in-scope and out-of-scope parts, on never-seen and already-seen turns."""
    from bakeoff import confirmatory as cf
    from bakeoff import posthoc_round3b as pb
    from bakeoff import revision_free as rf
    from bakeoff import retrieval_suppression as rs

    seen = pb.seen_texts()
    out = {}
    for name in ("banking77", "clinc150"):
        d = cf.load()[name]
        turns = cf.turns_of(d)
        in_scope = np.asarray([t["in_scope"] for t in turns])
        fresh = np.asarray([t["text"] not in seen for t in turns])
        clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
        sp = {s: rf.specialist(cf.features(name, s), 0.2) for s in cf.SEEDS}
        cs = np.mean([rf.turn_correct(turns, sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
        g = cf._hosted_answers(cf.OUT / "hosted" / f"A__{name}__gemma-4-31b.jsonl", turns, len(d["labels"]))
        diff = cs - rf.turn_correct(turns, g["scope"], g["intent"]).astype(float)
        block = {}
        for part, m in (("in-scope, never seen", fresh & in_scope), ("in-scope, seen before", ~fresh & in_scope), ("out-of-scope, never seen", fresh & ~in_scope), ("out-of-scope, seen before", ~fresh & ~in_scope)):
            if m.sum():
                block[part] = {"n": int(m.sum()), **rs.flip_test(diff[m], clusters[m], SEED, False)}
        out[name] = block
    return out


def cascade_break_even() -> dict:
    """Monthly sessions above which the cascade (hosting + 40% of Gemma's API cost) is cheaper than Gemma alone."""
    cost = json.loads((OUT / "round3b.json").read_text(encoding="utf-8"))["cost"]["sets"]
    out = {}
    for name, block in cost.items():
        per_session = block["gemma-4-31b"]["usd_per_1000_sessions_uncached_list"] / 1000
        out[name] = {f"host ${fixed}/month": fixed / (0.6 * per_session) for fixed in (50, 255)}
        out[name]["specialist_alone_host_$50"] = 50 / per_session
    return out


def zroc_sensitivity() -> dict:
    """D2 BANKING77-OOS: change in d_a (retrieval - zero-shot) for zROC slopes 0.6 to 1.25."""
    from bakeoff import retrieval_mechanism as rm
    from bakeoff import workflow_v4 as v4

    data = rm.load()["banking77_oos"]
    items = data["items"]
    in_scope = np.asarray([i["in_scope"] for i in items])
    out = {}
    for model in ("gemma-4-31b", "qwen3.5-27b", "llama-3.3-70b", "deepseek-v4-pro"):
        reject = {}
        for arm in ("zero_shot", "retrieval"):
            rows = v4._read(rm._path(model, "banking77_oos", arm))
            reject[arm] = np.asarray([not v4.parse(rows[i["id"]]["text"], len(data["labels"]))["scope"] if i["id"] in rows else i["in_scope"] for i in items])
        out[model] = {f"slope {s}": d_a(reject["retrieval"], in_scope, s) - d_a(reject["zero_shot"], in_scope, s) for s in (0.6, 0.8, 1.0, 1.25)}
    return out


def massive_rows_rejection() -> dict:
    """MASSIVE draws: rejection under cross-source rows vs no rows, overall and split by whether a row was retrieved."""
    from bakeoff import massive_robustness as mr
    from bakeoff import retrieval_suppression as rs
    from bakeoff import workflow_v4 as v4
    from bakeoff.workflow_retrieval import BM25

    data = mr.load()
    out = {}
    for draw in mr.DRAWS:
        d = data[draw]
        held = [i for i in d["items"] if not i["in_scope"]]
        n_train = len(d["train"])
        index = BM25([t for t, _ in d["train"]] + list(d["rows_cross"]))
        got_row = np.asarray([any(j >= n_train for j in index.top(i["text"], 20)) for i in held])
        clusters = np.asarray([i.get("domain", "") + str(k) for k, i in enumerate(held)])
        block = {"share_retrieving_a_cross_row": float(got_row.mean())}
        for model in ("gemma-4-31b", "qwen3.5-27b"):
            rej = {}
            for arm in ("no_rows", "cross_source_rows"):
                rows = v4._read(mr._path(model, draw, arm))
                rej[arm] = np.asarray([not v4.parse(rows[i["id"]]["text"], len(d["labels"]))["scope"] if i["id"] in rows else True for i in held], float)
            t = rs.flip_test(rej["cross_source_rows"] - rej["no_rows"], clusters, SEED, True)
            block[model] = {
                "rejection_no_rows": float(rej["no_rows"].mean()),
                "rejection_cross_rows": float(rej["cross_source_rows"].mean()),
                "gain_test": t,
                "gain_when_row_retrieved": float((rej["cross_source_rows"] - rej["no_rows"])[got_row].mean()) if got_row.any() else None,
                "gain_when_no_row_retrieved": float((rej["cross_source_rows"] - rej["no_rows"])[~got_row].mean()) if (~got_row).any() else None,
            }
        out[draw] = block
    return out


def massive_source_clustering() -> dict:
    """Confirmatory MASSIVE: specialist - Gemma and the run C rows effect with out-of-scope turns clustered by their source label."""
    from bakeoff import confirmatory as cf
    from bakeoff import revision_free as rf
    from bakeoff.massive import load_massive

    d = cf.load()["massive"]
    turns = cf.turns_of(d)
    ms = load_massive()
    label_of = {it.text: ms["labels"][it.gold_index] for it in ms["items"]["test"]}
    in_scope = np.asarray([t["in_scope"] for t in turns])
    clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else "o-" + label_of.get(t["orig_text"], "?") for t in turns])
    sp = {s: rf.specialist(cf.features("massive", s), 0.2) for s in cf.SEEDS}
    cs = np.mean([rf.turn_correct(turns, sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
    a = cf._hosted_answers(cf.OUT / "hosted" / "A__massive__gemma-4-31b.jsonl", turns, len(d["labels"]))
    c = cf._hosted_answers(cf.OUT / "hosted" / "C__massive__gemma-4-31b.jsonl", turns, len(d["labels"]))
    ca, cc = rf.turn_correct(turns, a["scope"], a["intent"]).astype(float), rf.turn_correct(turns, c["scope"], c["intent"]).astype(float)
    return {
        "out_of_scope_source_labels": int(len(set(clusters[~in_scope]))),
        "specialist_minus_gemma": cf.weighted_test(cs, ca, in_scope, clusters, SEED),
        "rows_effect": cf.weighted_test(cc, ca, in_scope, clusters, SEED),
    }


def jev_vs_gemma() -> dict:
    from bakeoff import confirmatory as cf
    from bakeoff import jev_rerun as jr
    from bakeoff import revision_free as rf

    out = {}
    for name in cf.SETS:
        d = cf.load()[name]
        turns = cf.turns_of(d)
        in_scope = np.asarray([t["in_scope"] for t in turns])
        clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
        g = cf._hosted_answers(cf.OUT / "hosted" / f"A__{name}__gemma-4-31b.jsonl", turns, len(d["labels"]))
        j = jr._decisions(jr.OUT / "J2" / f"{name}__retrieval.jsonl", [t["id"] for t in turns], in_scope)
        out[name] = cf.weighted_test(rf.turn_correct(turns, j["scope"], j["intent"]).astype(float), rf.turn_correct(turns, g["scope"], g["intent"]).astype(float), in_scope, clusters, SEED)
    mrd = json.loads((ROOT / "artifacts" / "massive-robustness" / "analysis.json").read_text(encoding="utf-8"))["draws"]
    out["massive_draws_jev_minus_gemma"] = {dr: 100 * (mrd[dr]["systems"]["jev-1.13.0 no_rows"]["weighted"] - mrd[dr]["systems"]["gemma-4-31b no_rows"]["weighted"]) for dr in ("draw_1", "draw_2", "draw_3")}
    out["massive_draws_qwen_minus_specialist"] = {dr: 100 * (mrd[dr]["systems"]["qwen3.5-27b no_rows"]["weighted"] - mrd[dr]["systems"]["specialist"]["weighted"]) for dr in ("draw_1", "draw_2", "draw_3")}
    return out


def latency_table() -> dict:
    from bakeoff import confirmatory as cf

    out = {}
    for run, model in (("A", "gemma-4-31b"), ("B", "qwen3.5-27b"), ("I", "kimi-k3"), ("I", "claude-opus-5")):
        lat = []
        for name in cf.SETS:
            for line in open(cf.OUT / "hosted" / f"{run}__{name}__{model}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                if r.get("error") is None and r.get("latency_ms"):
                    lat.append(r["latency_ms"])
        lat = np.asarray(lat)
        out[model] = {"calls": int(len(lat)), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99))}
    return out


def rule_after_works() -> dict:
    """D2 BANKING77-OOS: rule_after vs retrieval against run D's 'works' rule, and weighted accuracy (0.8 in-scope right + 0.2 rejection)."""
    from bakeoff import retrieval_mechanism as rm
    from bakeoff import retrieval_suppression as rs
    from bakeoff import workflow_v4 as v4

    data = rm.load()["banking77_oos"]
    items = data["items"]
    in_scope = np.asarray([i["in_scope"] for i in items])
    gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
    clusters = np.asarray([i["cluster"] for i in items])
    out = {}
    for model in ("gemma-4-31b", "qwen3.5-27b", "llama-3.3-70b", "deepseek-v4-pro"):
        arms = {}
        for arm in ("zero_shot", "retrieval", "rule_after_examples"):
            rows = v4._read(rm._path(model, "banking77_oos", arm))
            p = [v4.parse(rows[i["id"]]["text"], len(data["labels"])) if i["id"] in rows else {"scope": not i["in_scope"], "intent": -1} for i in items]
            arms[arm] = (np.asarray([x["scope"] for x in p]), np.asarray([x["intent"] for x in p]))
        rej = {a: (~s)[~in_scope].astype(float) for a, (s, _) in arms.items()}
        right = {a: (s & (it == gold))[in_scope].astype(float) for a, (s, it) in arms.items()}
        lost = rej["zero_shot"].mean() - rej["retrieval"].mean()
        recovered = (rej["rule_after_examples"].mean() - rej["retrieval"].mean()) / lost if lost > 0 else None
        cost = rs.flip_test(right["rule_after_examples"] - right["retrieval"], clusters[in_scope], SEED, False)
        weighted = {a: float(0.8 * right[a].mean() + 0.2 * rej[a].mean()) for a in arms}
        out[model] = {"share_recovered": recovered, "in_scope_right_change": cost, "works": bool(recovered is not None and recovered >= 0.5 and cost["ci95"][0] > -2.0), "weighted": weighted}
    return out


def run_d_overlap() -> dict:
    from bakeoff.confirmatory import normalise

    rd = json.loads((ROOT / "artifacts" / "retrieval-suppression" / "items.json").read_text(encoding="utf-8"))
    conf = json.loads((ROOT / "artifacts" / "confirmatory" / "items.json").read_text(encoding="utf-8"))
    conf_texts = {t["text"] for b in conf.values() for s in b["sessions"] for t in s}
    return {"run_D_banking77_oos_items_in_confirmatory_turns": int(sum(normalise(i["text"]) in conf_texts for i in rd["banking77_oos"]["items"])), "of": len(rd["banking77_oos"]["items"])}


def main() -> None:
    result = {"status": "POST HOC: round 4 review requests; not pre-registered"}
    for key, fn in (("sibling_absorption", sibling_absorption), ("freshness_components", freshness_components), ("cascade_break_even", cascade_break_even), ("zroc_sensitivity", zroc_sensitivity), ("massive_rows_rejection", massive_rows_rejection), ("massive_source_clustering", massive_source_clustering), ("jev_vs_gemma", jev_vs_gemma), ("latency", latency_table), ("rule_after_works", rule_after_works), ("run_d_overlap", run_d_overlap)):
        result[key] = fn()
        print(f"done {key}", flush=True)
    (OUT / "round4.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("sibling_absorption",)}, indent=1, default=lambda o: round(float(o), 4))[:9000])
    sa = result["sibling_absorption"]
    for m, v in sa["models"].items():
        print(m, "accepted", v["accepted_held_out"], "among retrieved", round(v["chosen_label_among_retrieved"], 3), "top1", round(v["chosen_label_is_top1_retrieved"], 3), "conc", [round(x, 3) for x in v["drop_concentrated"]], "diffuse", [round(x, 3) for x in v["drop_diffuse"]], "map", v["top_mappings"][:4])


if __name__ == "__main__":
    main()

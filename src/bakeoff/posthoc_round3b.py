"""POST HOC re-analyses from the round 3 internal review.

None of this was pre-registered. Output: artifacts/posthoc/round3b.json, artifacts/posthoc/b77oos_label_audit.json.

A  BANKING77-OOS label audit against the original Banking77 labels (the published split mislabels rows).
B  Run D and Jev J1 on BANKING77-OOS restricted to leak-free subsets; closeness with a clean pool.
C  Mitigations (veto, floors) on held-out intents that the dev items did not cover, with the pre-registered 'works' rule.
D  Freshness: every confirmatory comparison restricted to turns that appeared in no earlier session (any casing).
E  MASSIVE: label coverage, per-domain out-of-scope rejection, general_quirky sensitivity, minimum detectable differences.
F  Monte Carlo spread of the Banking77 equivalence bound.
G  Cost on one billing basis, hosting break-even, host table, cross-host variance (DeepSeek Parasail vs StreamLake).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "posthoc"
SEED = 20261016
NEWLINE = "\n"


# ---------------------------------------------------------------- A. label audit


def original_labels() -> dict[str, set[str]]:
    from datasets import load_dataset

    ds = load_dataset("legacy-datasets/banking77", revision="f54121560de48f2852f90be299010d1d6dc612ec")  # as protocol.json pins; the only cached snapshot
    names = ds["train"].features["label"].names
    out: dict[str, set[str]] = {}
    for split in ds:
        for row in ds[split]:
            out.setdefault(row["text"].strip().lower(), set()).add(names[row["label"]])
    return out


def label_audit(orig) -> dict:
    from bakeoff import retrieval_suppression as rs

    block = rs.load()["banking77_oos"]
    labels = [label.replace(" ", "_") for label in block["labels"]]
    held = {i["cluster"][5:] for i in block["items"] if not i["in_scope"]}
    true = lambda text: (sorted(orig[text.strip().lower()])[0] if text.strip().lower() in orig else None)
    pool_flags = []
    for k, (text, gold) in enumerate(block["pool"]):
        t = true(text)
        pool_flags.append({"row": k, "matched": t is not None, "mislabelled": bool(t and t != labels[gold]), "leaked_held_out": bool(t in held), "true": t, "given": labels[gold]})
    items = {}
    for i in block["items"]:
        t = true(i["text"])
        if i["in_scope"]:
            items[i["id"]] = {"wrong_gold": bool(t and t != labels[i["intent"]]), "true_is_held_out": bool(t in held)}
        else:
            items[i["id"]] = {"true_is_in_scope": bool(t and t not in held), "true": t}
    leaked = [f for f in pool_flags if f["leaked_held_out"]]
    out = {
        "status": "POST HOC (round 3 review, R2 N1). Text matched to legacy-datasets/banking77 after lower-casing and stripping.",
        "source_identical_to_upstream": "data/banking77-oos/train is byte-identical to github.com/jianguoz/Few-Shot-Intent-Detection commit fa955331 (sha256 of train/seq.in 5ba97261..., train/label b4cfaddc...)",
        "pool_rows": len(pool_flags),
        "pool_matched": sum(f["matched"] for f in pool_flags),
        "pool_mislabelled": sum(f["mislabelled"] for f in pool_flags),
        "pool_leaked_held_out": len(leaked),
        "held_out_intents_leaking": sorted({f["true"] for f in leaked}),
        "in_scope_items_wrong_gold": sum(v.get("wrong_gold", False) for v in items.values()),
        "held_out_items_true_in_scope": sum(v.get("true_is_in_scope", False) for v in items.values()),
        "leaked_rows": [f["row"] for f in leaked],
        "mislabelled_rows": [f["row"] for f in pool_flags if f["mislabelled"]],
        "items": items,
    }
    out["leaked_rows_sha256"] = hashlib.sha256(json.dumps(out["leaked_rows"]).encode()).hexdigest()
    return out


# ---------------------------------------------------------------- B. run D and Jev J1 on leak-free subsets


def leak_reanalysis(audit: dict) -> dict:
    from bakeoff import jev_rerun as jr
    from bakeoff import posthoc_round3 as p3
    from bakeoff import retrieval_suppression as rs
    from bakeoff.confirmatory import _auroc
    from bakeoff.workflow_retrieval import BM25, embed

    spec = rs.rules()
    block = rs.load()["banking77_oos"]
    items = block["items"]
    in_scope = np.asarray([i["in_scope"] for i in items])
    clusters = np.asarray([i["cluster"] for i in items])
    gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
    leaking = set(audit["held_out_intents_leaking"])
    leaked_rows = set(audit["leaked_rows"])
    pool = [tuple(p) for p in block["pool"]]
    index = BM25([t for t, _ in pool])
    retrieved_leak = np.asarray([(not i["in_scope"]) and any(j in leaked_rows for j in index.top(i["text"], 20)) for i in items])
    never_leaks = np.asarray([i["in_scope"] or i["cluster"][5:] not in leaking for i in items])
    clean_items = np.asarray([not (audit["items"][i["id"]].get("wrong_gold") or audit["items"][i["id"]].get("true_is_in_scope")) for i in items])
    subsets = {
        "as run": np.ones(len(items), bool),
        "held-out intents that never leak": never_leaks,
        "never-leaking intents, label-clean items": never_leaks & clean_items,
        "held-out items with no leaked row retrieved": in_scope | ~retrieved_leak,
    }
    result = {"subset_sizes": {k: {"in_scope": int((v & in_scope).sum()), "held_out": int((v & ~in_scope).sum()), "held_out_intents": int(len(set(clusters[v & ~in_scope])))} for k, v in subsets.items()}, "models": {}}
    systems = {name: {arm: rs._answers(rs._path(name, "banking77_oos", arm), items, len(block["labels"])) for arm in rs.ARMS} for name in spec["models"]}
    for name, arms in systems.items():
        rows = {}
        for sub, mask in subsets.items():
            m_in, m_out = mask & in_scope, mask & ~in_scope
            rej = {a: (~arms[a]["scope"])[m_out].astype(float) for a in rs.ARMS}
            t = rs.flip_test(rej["zero_shot"] - rej["retrieval"], clusters[m_out], SEED, True)
            sd = p3.sdt(~arms["zero_shot"]["scope"][mask], in_scope[mask]), p3.sdt(~arms["retrieval"]["scope"][mask], in_scope[mask])
            rows[sub] = {
                "rejected_zero_shot": float(rej["zero_shot"].mean()),
                "rejected_retrieval": float(rej["retrieval"].mean()),
                "rejected_instruction": float(rej["mitigation_instruction"].mean()),
                "drop_test": t,
                "in_scope_kept_zero_shot": float(arms["zero_shot"]["scope"][m_in].mean()),
                "in_scope_kept_retrieval": float(arms["retrieval"]["scope"][m_in].mean()),
                "d_prime_change": sd[1][0] - sd[0][0],
                "criterion_change": sd[1][1] - sd[0][1],
            }
        result["models"][name] = rows
    # Jev J1 on the same subsets, with AUROC of its scope probability.
    jev = {}
    for arm in ("zero_shot", "retrieval", "instruction"):
        jev[arm] = jr._decisions(jr.OUT / "J1" / f"banking77_oos__{arm}.jsonl", [i["id"] for i in items], in_scope)
    rows = {}
    for sub, mask in subsets.items():
        m_out = mask & ~in_scope
        rej = {a: (~jev[a]["scope"])[m_out].astype(float) for a in jev}
        pz, pr = np.nan_to_num(jev["zero_shot"]["prob"], nan=0.5), np.nan_to_num(jev["retrieval"]["prob"], nan=0.5)
        _, boots = jr._auroc_boot(pz[mask], in_scope[mask], clusters[mask], SEED)
        d_auc = [(_auroc(pr[mask][b], in_scope[mask][b]) - _auroc(pz[mask][b], in_scope[mask][b])) for b in boots]
        rows[sub] = {
            "rejected_zero_shot": float(rej["zero_shot"].mean()),
            "rejected_retrieval": float(rej["retrieval"].mean()),
            "drop_test": rs.flip_test(rej["zero_shot"] - rej["retrieval"], clusters[m_out], SEED, True),
            "auroc_zero_shot": _auroc(pz[mask], in_scope[mask]),
            "auroc_retrieval": _auroc(pr[mask], in_scope[mask]),
            "auroc_change_ci95": np.percentile(d_auc, [2.5, 97.5]).tolist(),
            "mean_prob_held_out_change": float(pr[m_out].mean() - pz[m_out].mean()),
        }
    result["models"]["jev-1.13.0"] = rows
    # Closeness with and without the leaked rows in the pool.
    texts = [i["text"] for i in items]
    emb = embed(texts)
    full = embed([t for t, _ in pool])
    keep = np.asarray([k not in leaked_rows for k in range(len(pool))])
    result["closeness_dense_auroc"] = {
        "as run (leaky pool)": _auroc((emb @ full.T).max(axis=1), in_scope),
        "leaked rows removed": _auroc((emb @ full[keep].T).max(axis=1), in_scope),
        "leaked rows removed, never-leaking intents, label-clean items": _auroc((emb[subsets["never-leaking intents, label-clean items"]] @ full[keep].T).max(axis=1), in_scope[subsets["never-leaking intents, label-clean items"]]),
    }
    return result


# ---------------------------------------------------------------- C. mitigations on dev-unseen intents


def mitigations_unseen() -> dict:
    from bakeoff import retrieval_suppression as rs

    r3 = json.loads((OUT / "round3.json").read_text(encoding="utf-8"))
    data = rs.load()
    sd = ROOT / "data" / "clinc-single-domain-oos"
    dev_seen = {
        "banking77_oos": set(open(ROOT / "data" / "banking77-oos" / "id-oos" / "valid" / "label_original", encoding="utf-8").read().splitlines()[:100]),
        "clinc_banking": set(open(sd / "banking" / "id-oos" / "valid" / "label_original", encoding="utf-8").read().splitlines()[:100]),
        "clinc_credit_cards": set(open(sd / "credit_cards" / "id-oos" / "valid" / "label_original", encoding="utf-8").read().splitlines()[:100]),
    }
    out = {"dev_seen_held_out_intents": {k: sorted(v) for k, v in dev_seen.items()}, "note": "round3.json kept only rates, so the veto and floors are recomputed here on the unseen-intent held-out items"}
    from bakeoff import posthoc_round3 as p3
    from bakeoff import revision_free as rf
    from bakeoff.workflow_retrieval import BM25, embed

    groups = {"banking77_oos": ["banking77_oos"], "clinc_single_domain": ["clinc_banking", "clinc_credit_cards"]}
    spec = rs.rules()
    for group, splits in groups.items():
        items = [i for s in splits for i in data[s]["items"]]
        in_scope = np.asarray([i["in_scope"] for i in items])
        gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
        clusters = np.asarray([i["cluster"] for i in items])
        unseen = np.asarray([i["in_scope"] or not any(i["cluster"].endswith("-held-" + name) or i["cluster"] == "held-" + name for s in splits for name in dev_seen[s]) for i in items])
        # Dense floor and specialist veto, as in posthoc_round3.
        top_dense, dev_in, dev_out = [], [], []
        for s in splits:
            b = data[s]
            pool_emb = embed([t for t, _ in b["pool"]])
            top_dense.extend((embed([i["text"] for i in b["items"]]) @ pool_emb.T).max(axis=1).tolist())
            dev_in.extend((embed([t for t, _ in b["dev_in"]]) @ pool_emb.T).max(axis=1).tolist())
            dev_out.extend((embed(list(b["dev_out"])) @ pool_emb.T).max(axis=1).tolist())
        floor = np.asarray(top_dense) >= p3.youden_threshold(np.asarray(dev_in), np.asarray(dev_out))
        spec_accept = np.mean([np.concatenate([rf.specialist(rs.features(s, seed), 0.2)["scope"] for s in splits]) for seed in rs.SEEDS], axis=0) >= 0.5
        g = {"held_out_unseen": int((unseen & ~in_scope).sum()), "held_out_seen": int((~unseen).sum()), "specialist": {"rejected_seen": float((~spec_accept)[~unseen].mean()), "rejected_unseen": float((~spec_accept)[unseen & ~in_scope].mean())}, "models": {}}
        for name in spec["models"]:
            arms = {a: {k: np.concatenate([rs._answers(rs._path(name, s, a), data[s]["items"], len(data[s]["labels"]))[k] for s in splits]) for k in ("scope", "intent")} for a in rs.ARMS}
            ret, zs = arms["retrieval"], arms["zero_shot"]
            rules_ = {"retrieval + specialist veto": (ret["scope"] & spec_accept, ret["intent"]), "retrieval + dense floor": (ret["scope"] & floor, ret["intent"]), "instruction + specialist veto": (arms["mitigation_instruction"]["scope"] & spec_accept, arms["mitigation_instruction"]["intent"])}
            m = unseen
            rej = lambda sc: (~sc)[m & ~in_scope].astype(float)
            right = lambda sc, it: (sc & (it == gold))[in_scope].astype(float)
            row = {"zero_shot_rejected": float(rej(zs["scope"]).mean()), "retrieval_rejected": float(rej(ret["scope"]).mean())}
            for key, (sc, it) in rules_.items():
                gain = rs.flip_test(rej(sc) - rej(ret["scope"]), clusters[m & ~in_scope], SEED, True)
                cost = rs.flip_test(right(sc, it) - right(ret["scope"], ret["intent"]), clusters[in_scope], SEED, False)
                lost = row["zero_shot_rejected"] - row["retrieval_rejected"]
                recovered = (rej(sc).mean() - row["retrieval_rejected"]) / lost if lost > 0 else None
                row[key] = {"rejected": float(rej(sc).mean()), "gain_test": gain, "in_scope_right_change": cost, "share_recovered": recovered, "works_by_run_D_rule": bool(recovered is not None and recovered >= 0.5 and cost["ci95"][0] > -2.0)}
            g["models"][name] = row
        out[group] = g
    return out


# ---------------------------------------------------------------- D. freshness


def seen_texts() -> set[str]:
    """Normalised texts of every turn in any session built before the confirmatory run."""
    from bakeoff import workflow as v1
    from bakeoff import workflow_clinc as c
    from bakeoff import workflow_massive as m
    from bakeoff import workflow_near_oos as wn
    from bakeoff.confirmatory import normalise

    paths = [v1.OUT / "sessions.json", m.OUT / "sessions.json", c.OUT / "sessions.json", wn.OUT / "sessions__id.json", wn.OUT / "sessions__ood.json"]
    seen = set()
    for path in paths:
        for session in json.loads(path.read_text(encoding="utf-8")):
            for turn in session:
                seen.add(normalise(turn["text"]))
    return seen


def freshness() -> dict:
    from bakeoff import confirmatory as cf
    from bakeoff import jev_rerun as jr
    from bakeoff import revision_free as rf

    seen = seen_texts()
    spec = cf.rules()
    data = cf.load()
    gemma, qwen = spec["general_models"]["primary"]["name"], spec["general_models"]["secondary"]["name"]
    out = {"definition": "a turn is 'never seen' when its normalised text appears in no session built before the confirmatory run (workflow v1, MASSIVE, CLINC150, BANKING77-OOS id and ood)", "sets": {}}
    primary = {}
    for name in cf.SETS:
        d = data[name]
        turns = cf.turns_of(d)
        fresh = np.asarray([t["text"] not in seen for t in turns])
        in_scope = np.asarray([t["in_scope"] for t in turns])
        clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
        n = len(turns)
        tc = lambda sc, it: rf.turn_correct(turns, sc, it).astype(float)
        sp = {s: rf.specialist(cf.features(name, s), 0.2) for s in cf.SEEDS}
        c_spec = np.mean([tc(sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
        llm = {run: cf._hosted_answers(cf.OUT / "hosted" / f"{run}__{name}__{model}.jsonl", turns, len(d["labels"])) for run, model in (("A", gemma), ("B", qwen), ("C", gemma))}
        c = {"specialist": c_spec, gemma: tc(llm["A"]["scope"], llm["A"]["intent"]), qwen: tc(llm["B"]["scope"], llm["B"]["intent"]), "gemma with rows": tc(llm["C"]["scope"], llm["C"]["intent"])}
        for run, model in (("A", gemma), ("B", qwen)):
            comb, casc = [], []
            for s in cf.SEEDS:
                esc = np.zeros(n, bool)
                esc[np.argsort(sp[s]["top"], kind="stable")[: int(round(0.4 * n))]] = True
                comb.append(tc(llm[run]["scope"], np.where(sp[s]["top"] >= 0.9, sp[s]["intent"], llm[run]["intent"])))
                casc.append(tc(np.where(esc, llm[run]["scope"], sp[s]["scope"]), np.where(esc, llm[run]["intent"], sp[s]["intent"])))
            c[f"{model}: combination"], c[f"{model}: cascade_0.4"] = np.mean(comb, axis=0), np.mean(casc, axis=0)
        for arm in jr.J2_ARMS:
            a = jr._decisions(jr.OUT / "J2" / f"{name}__{arm}.jsonl", [t["id"] for t in turns], in_scope)
            c[f"jev {arm}"] = tc(a["scope"], a["intent"])
        sub = lambda v: v[fresh]
        block = {"turns": n, "never_seen": int(fresh.sum()), "never_seen_in_scope": int((fresh & in_scope).sum()), "never_seen_out_of_scope": int((fresh & ~in_scope).sum()), "weighted_never_seen": {k: cf.weighted(sub(v), in_scope[fresh]) for k, v in c.items()}, "tests_never_seen": {}}
        pairs = [("specialist", gemma), (f"{gemma}: combination", gemma), ("specialist", qwen), (f"{gemma}: cascade_0.4", gemma), ("gemma with rows", gemma), ("jev retrieval", gemma), ("jev retrieval", "specialist")]
        for a_, b_ in pairs:
            t = cf.weighted_test(sub(c[a_]), sub(c[b_]), in_scope[fresh], clusters[fresh], SEED)
            block["tests_never_seen"][f"{a_} vs {b_}"] = t
            if (a_, b_) in (("specialist", gemma), (f"{gemma}: combination", gemma)):
                primary[f"{name} | {a_} vs {b_}"] = t["p"]
        out["sets"][name] = block
    for label, value in rf.holm(primary).items():
        name, key = label.split(" | ")
        out["sets"][name]["tests_never_seen"][key]["holm_p_same_family"] = value
    return out


# ---------------------------------------------------------------- E. MASSIVE


def massive() -> dict:
    from bakeoff import confirmatory as cf
    from bakeoff import jev_rerun as jr
    from bakeoff import revision_free as rf

    data = cf.load()["massive"]
    turns = cf.turns_of(data)
    labels = data["labels"]
    in_scope = np.asarray([t["in_scope"] for t in turns])
    clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
    items = json.loads((ROOT / "artifacts" / "confirmatory" / "items.json").read_text(encoding="utf-8"))
    from bakeoff.massive import load_massive

    ms = load_massive()
    all_labels = ms["labels"]
    domain_of = {}
    for split in ("test",):
        for it in ms["items"][split]:
            domain_of[it.text] = all_labels[it.gold_index].split(" ")[0]
    out_domains = np.asarray([domain_of.get(t["orig_text"], "?") if not t["in_scope"] else "" for t in turns])
    sp = {s: rf.specialist(cf.features("massive", s), 0.2) for s in cf.SEEDS}
    tc = lambda sc, it: rf.turn_correct(turns, sc, it).astype(float)
    systems = {"specialist (3 seeds, majority)": np.mean([sp[s]["scope"] for s in cf.SEEDS], axis=0) >= 0.5}
    for run, model in (("A", "gemma-4-31b"), ("B", "qwen3.5-27b"), ("C", "gemma-4-31b")):
        systems[f"{model}{' with rows' if run == 'C' else ''}"] = cf._hosted_answers(cf.OUT / "hosted" / f"{run}__massive__{model}.jsonl", turns, len(labels))["scope"]
    for arm in jr.J2_ARMS:
        systems[f"jev {arm}"] = jr._decisions(jr.OUT / "J2" / f"massive__{arm}.jsonl", [t["id"] for t in turns], in_scope)["scope"]
    per_domain = {}
    for dom in sorted(set(out_domains[~in_scope])):
        sel = out_domains == dom
        per_domain[dom] = {"turns": int(sel.sum()), **{k: float((~v)[sel].mean()) for k, v in systems.items()}}
    # Frontier models on their subset.
    k = cf.rules()["runs"]["I"]["sessions_per_set"]
    m = sum(len(s) for s in data["sessions"][:k])
    for fm in ("kimi-k3", "claude-opus-5"):
        f = cf._hosted_answers(cf.OUT / "hosted" / f"I__massive__{fm}.jsonl", turns[:m], len(labels))
        for dom in per_domain:
            sel = out_domains[:m] == dom
            per_domain[dom][f"{fm} (first {k} sessions)"] = float((~f["scope"])[sel].mean()) if sel.any() else None
    covered = Counter(t["intent"] for t in turns if t["in_scope"])
    top5 = covered.most_common(5)
    gq = labels.index("general quirky")
    keep = np.asarray([not (t["in_scope"] and t["intent"] == gq) for t in turns])
    c_spec = np.mean([tc(sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
    gemma = cf._hosted_answers(cf.OUT / "hosted" / "A__massive__gemma-4-31b.jsonl", turns, len(labels))
    c_g = tc(gemma["scope"], gemma["intent"])
    return {
        "in_scope_labels_covered": len(covered),
        "labels_total": len(labels),
        "top5_share": sum(n for _, n in top5) / sum(covered.values()),
        "top5": [(labels[i], n) for i, n in top5],
        "dev_labels_covered": len({r["intent"] for r in data["dev_in"]}),
        "test_labels_missing_from_dev": len(set(covered) - {r["intent"] for r in data["dev_in"]}),
        "out_of_scope_rejected_by_domain": per_domain,
        "specialist_vs_gemma_without_general_quirky": cf.weighted_test(c_spec[keep], c_g[keep], in_scope[keep], clusters[keep], SEED),
    }


def mdd() -> dict:
    """Minimum detectable difference at 80% power, two-sided 0.05, under the pre-registered weighted sign-flip test: 2.8 x the SD of the observed paired differences' sign-flip distribution."""
    from bakeoff import confirmatory as cf
    from bakeoff import revision_free as rf

    out = {}
    data = cf.load()
    for name in cf.SETS:
        turns = cf.turns_of(data[name])
        in_scope = np.asarray([t["in_scope"] for t in turns])
        clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
        sp = {s: rf.specialist(cf.features(name, s), 0.2) for s in cf.SEEDS}
        c_spec = np.mean([rf.turn_correct(turns, sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
        g = cf._hosted_answers(cf.OUT / "hosted" / f"A__{name}__gemma-4-31b.jsonl", turns, len(data[name]["labels"]))
        diff = c_spec - rf.turn_correct(turns, g["scope"], g["intent"]).astype(float)
        weight = np.where(in_scope, 0.8 / in_scope.sum(), 0.2 / (~in_scope).sum())
        keys, inverse = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inverse, weights=weight * diff)
        rng = np.random.default_rng(SEED)
        null = rng.choice([-1.0, 1.0], size=(10000, len(sums))) @ sums
        out[name] = {"null_sd_points": float(100 * null.std()), "mdd_80pct_points": float(2.8 * 100 * null.std())}
    return out


# ---------------------------------------------------------------- F. equivalence bound noise


def tost_noise() -> dict:
    from bakeoff import confirmatory as cf
    from bakeoff import revision_free as rf

    data = cf.load()["banking77"]
    turns = cf.turns_of(data)
    in_scope = np.asarray([t["in_scope"] for t in turns])
    clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
    sp = {s: rf.specialist(cf.features("banking77", s), 0.2) for s in cf.SEEDS}
    c_spec = np.mean([rf.turn_correct(turns, sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS], axis=0)
    g = cf._hosted_answers(cf.OUT / "hosted" / "A__banking77__gemma-4-31b.jsonl", turns, len(data["labels"]))
    c_g = rf.turn_correct(turns, g["scope"], g["intent"]).astype(float)
    lows = [cf.weighted_test(c_spec, c_g, in_scope, clusters, seed)["ci90"][0] for seed in range(20261016, 20261016 + 40)]
    return {"lower_bound_mean": float(np.mean(lows)), "lower_bound_sd": float(np.std(lows)), "share_of_seeds_inside_margin": float(np.mean(np.asarray(lows) > -2.5)), "seeds": 40}


# ---------------------------------------------------------------- G. cost basis, hosts, cross-host variance


HOSTS = {
    "gemma-4-31b": {"host": "Novita", "precision": "bf16", "usd_per_million": (0.14, 0.40)},
    "qwen3.5-27b": {"host": "Alibaba", "precision": "not stated by host", "usd_per_million": (0.195, 1.56)},
    "kimi-k3": {"host": "Moonshot AI", "precision": "mxfp4", "usd_per_million": (3.0, 15.0)},
    "claude-opus-5": {"host": "Anthropic", "precision": "not stated by host", "usd_per_million": (5.0, 25.0)},
    "llama-3.3-70b": {"host": "DeepInfra", "precision": "fp8", "usd_per_million": (0.10, 0.32)},
    "deepseek-v4-pro": {"host": "StreamLake (Parasail fp8 partial, discarded)", "precision": "fp8", "usd_per_million": (0.955, 1.911)},
    "jev-1.13.0": {"host": "TypeSafe", "precision": "not stated", "usd_per_million": None},
}


def cost_basis() -> dict:
    from bakeoff import confirmatory as cf

    r = json.loads((cf.OUT / "analysis.json").read_text(encoding="utf-8"))
    out = {"prices_as_of": "2026-10-15/16, OpenRouter listing for the pinned host", "hosts": HOSTS, "sets": {}}
    data = cf.load()
    for name in cf.SETS:
        per_session = len(cf.turns_of(data[name])) / len(data[name]["sessions"])
        rows_by = {}
        for model, path in (("gemma-4-31b", f"A__{name}__gemma-4-31b"), ("qwen3.5-27b", f"B__{name}__qwen3.5-27b"), ("kimi-k3", f"I__{name}__kimi-k3"), ("claude-opus-5", f"I__{name}__claude-opus-5")):
            rows = [json.loads(line) for line in open(cf.OUT / "hosted" / f"{path}.jsonl", encoding="utf-8")]
            rows = list({row["id"]: row for row in rows if row.get("error") is None}.values())
            pin, pout = HOSTS[model]["usd_per_million"]
            uncached = np.mean([(row["prompt_tokens"] * pin + row["completion_tokens"] * pout) / 1e6 for row in rows])
            billed = np.mean([row["cost"] for row in rows])
            cached_share = sum(row.get("cached_tokens") or 0 for row in rows) / sum(row["prompt_tokens"] for row in rows)
            rows_by[model] = {"usd_per_1000_sessions_billed": 1000 * billed * per_session, "usd_per_1000_sessions_uncached_list": 1000 * uncached * per_session, "cached_share": cached_share}
        g = rows_by["gemma-4-31b"]["usd_per_1000_sessions_uncached_list"]
        for model in rows_by:
            rows_by[model]["multiple_of_gemma_uncached"] = rows_by[model]["usd_per_1000_sessions_uncached_list"] / g
        # Break-even: monthly sessions above which hosting the specialist costs less than calling Gemma on every turn.
        rows_by["break_even_sessions_per_month"] = {f"host ${fixed}/month": fixed / (g / 1000) for fixed in (50, 255)}
        out["sets"][name] = rows_by
    # Cross-host variance: DeepSeek on Parasail vs StreamLake, same 686 prompts.
    from bakeoff import workflow_v4 as v4

    para = v4._read(ROOT / "artifacts" / "retrieval-suppression" / "parasail-partial" / "hosted" / "deepseek-v4-pro__banking77_oos__zero_shot.jsonl")
    stream = v4._read(ROOT / "artifacts" / "retrieval-suppression" / "hosted" / "deepseek-v4-pro__banking77_oos__zero_shot.jsonl")
    common = sorted(set(para) & set(stream))
    pp = {k: v4.parse(para[k]["text"], 50) for k in common}
    ss = {k: v4.parse(stream[k]["text"], 50) for k in common}
    held = [k for k in common if "-out-" in k]
    out["cross_host_deepseek"] = {
        "requests": len(common),
        "different_reply_text": float(np.mean([para[k]["text"].strip() != stream[k]["text"].strip() for k in common])),
        "scope_flips": float(np.mean([pp[k]["scope"] != ss[k]["scope"] for k in common])),
        "held_out_rejected_parasail": float(np.mean([not pp[k]["scope"] for k in held])),
        "held_out_rejected_streamlake": float(np.mean([not ss[k]["scope"] for k in held])),
    }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    orig = original_labels()
    audit = label_audit(orig)
    (OUT / "b77oos_label_audit.json").write_text(json.dumps(audit, indent=1) + NEWLINE, encoding="utf-8")
    print(f"audit: pool {audit['pool_rows']} matched {audit['pool_matched']} mislabelled {audit['pool_mislabelled']} leaked {audit['pool_leaked_held_out']} ({len(audit['held_out_intents_leaking'])} intents); in-scope items wrong gold {audit['in_scope_items_wrong_gold']}; held-out items truly in scope {audit['held_out_items_true_in_scope']}", flush=True)
    result = {"status": "POST HOC: round 3 review requests; not pre-registered", "label_audit_summary": {k: v for k, v in audit.items() if k not in ("items", "leaked_rows", "mislabelled_rows")}}
    result["leak"] = leak_reanalysis(audit)
    print("leak done", flush=True)
    result["mitigations_unseen"] = mitigations_unseen()
    print("mitigations done", flush=True)
    result["freshness"] = freshness()
    print("freshness done", flush=True)
    result["massive"] = massive()
    result["mdd"] = mdd()
    result["tost_noise"] = tost_noise()
    result["cost"] = cost_basis()
    (OUT / "round3b.json").write_text(json.dumps(result, indent=1, default=float) + NEWLINE, encoding="utf-8")
    print("ROUND3B_DONE", flush=True)


if __name__ == "__main__":
    main()

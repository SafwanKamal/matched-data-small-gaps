"""Run D2: replication on fresh label-clean items and mechanism controls. Rules: retrieval_mechanism_protocol.json.

Stages: 'build' (frozen items), 'check' (validation items, API), 'hosted' (general models), 'jev', 'analyze'.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "retrieval_mechanism_protocol.json"
OUT = ROOT / "artifacts" / "retrieval-mechanism"
ITEMS = OUT / "items.json"
NEWLINE = "\n"
B77_ARMS = ["zero_shot", "retrieval", "random_examples", "rule_after_examples"]
SD_ARMS = ["zero_shot_10", "retrieval_10", "zero_shot_20", "retrieval_20"]
DOMAINS = ["banking", "credit_cards"]
SCOPE_SENTENCE = "The desk handles only the request types listed below; anything else, including other banking questions, must be passed on."


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------- build


def _seen() -> set[str]:
    """Normalised texts of every earlier session turn, dev item and experiment item."""
    from bakeoff.confirmatory import normalise
    from bakeoff.posthoc_round3b import seen_texts

    seen = set(seen_texts())
    conf = json.loads((ROOT / "artifacts" / "confirmatory" / "items.json").read_text(encoding="utf-8"))
    for name, block in conf.items():
        for session in block["sessions"]:
            seen.update(t["text"] for t in session)
        seen.update(r["text"] for r in block["dev_in"] + block["dev_out"])
    rsd = json.loads((ROOT / "artifacts" / "retrieval-suppression" / "items.json").read_text(encoding="utf-8"))
    for key, block in rsd.items():
        if key == "mitigation_rows":
            continue
        seen.update(normalise(i["text"]) for i in block["items"])
        seen.update(normalise(t) for t, _ in block["dev_in"])
        seen.update(normalise(t) for t in block["dev_out"])
    return seen


def _clinc_names() -> dict[str, set[str]]:
    from datasets import load_dataset

    from bakeoff import workflow_clinc as c

    raw = load_dataset(c.SOURCE[0], c.SOURCE[1], revision=c.SOURCE[2])
    out: dict[str, set[str]] = {}
    for split in raw:
        names = raw[split].features["intent"].names
        for row in raw[split]:
            out.setdefault(row["text"].strip().lower(), set()).add(names[row["intent"]])
    return out


def build() -> None:
    from bakeoff.confirmatory import normalise
    from bakeoff.posthoc_round3b import original_labels

    spec = rules()
    rng = np.random.default_rng(spec["seed"])
    seen = _seen()
    out = {}
    # BANKING77-OOS: fresh label-clean validation items; pool without any mislabelled row.
    orig = original_labels()
    audit = json.loads((ROOT / "artifacts" / "posthoc" / "b77oos_label_audit.json").read_text(encoding="utf-8"))
    bad = set(audit["mislabelled_rows"])
    data = ROOT / "data" / "banking77-oos"
    train = list(zip(_lines(data / "train" / "seq.in"), _lines(data / "train" / "label")))
    names = sorted({label for _, label in train})
    index = {name: i for i, name in enumerate(names)}
    held = set(_lines(data / "id-oos" / "test" / "label_original"))
    fresh = lambda t: normalise(t) not in seen
    in_ok = [(t, l) for t, l in zip(_lines(data / "valid" / "seq.in"), _lines(data / "valid" / "label")) if fresh(t) and (orig.get(t.strip().lower()) is None or l in orig[t.strip().lower()])]
    out_ok = [(t, l) for t, l in zip(_lines(data / "id-oos" / "valid" / "seq.in"), _lines(data / "id-oos" / "valid" / "label_original")) if fresh(t) and (orig.get(t.strip().lower()) is None or orig[t.strip().lower()] & held)]
    pick = sorted(rng.choice(len(in_ok), size=500, replace=False).tolist())
    out["banking77_oos"] = {
        "labels": [n.replace("_", " ") for n in names],
        "pool": [[t, index[l]] for k, (t, l) in enumerate(train) if k not in bad],
        "items": [{"id": f"b77v-in-{k}", "text": in_ok[k][0], "in_scope": True, "intent": index[in_ok[k][1]], "cluster": f"in-{in_ok[k][1]}"} for k in pick]
        + [{"id": f"b77v-out-{k}", "text": t, "in_scope": False, "intent": None, "cluster": f"held-{l}"} for k, (t, l) in enumerate(out_ok)],
        "pool_rows_removed": len(bad),
    }
    # CLINC single-domain: validation items; held-out after the first 100 lines.
    clinc = _clinc_names()
    sd = ROOT / "data" / "clinc-single-domain-oos"
    combined = sorted({l for d in DOMAINS for l in _lines(sd / d / "train" / "label")})
    cindex = {name: i for i, name in enumerate(combined)}
    for d in DOMAINS:
        own = sorted(set(_lines(sd / d / "train" / "label")))
        oindex = {name: i for i, name in enumerate(own)}
        valid = list(zip(_lines(sd / d / "valid" / "seq.in"), _lines(sd / d / "valid" / "label")))
        held_rows = list(zip(_lines(sd / d / "id-oos" / "valid" / "seq.in"), _lines(sd / d / "id-oos" / "valid" / "label_original")))[100:]
        check = lambda t, l: clinc.get(t.strip().lower()) is None or l in clinc[t.strip().lower()]
        # In-scope validation items served only as run D's specialist development data (never sent to a model); allowed.
        spec_dev = {normalise(t) for t, _ in json.loads((ROOT / "artifacts" / "retrieval-suppression" / "items.json").read_text(encoding="utf-8"))[f"clinc_{d}"]["dev_in"]}
        fresh_in = lambda t: normalise(t) not in (seen - spec_dev)
        items = [{"id": f"{d}-v-in-{k}", "text": t, "in_scope": True, "intent": oindex[l], "intent20": cindex[l], "cluster": f"{d}-in-{l}"} for k, (t, l) in enumerate(valid) if fresh_in(t) and check(t, l)]
        items += [{"id": f"{d}-v-out-{k}", "text": t, "in_scope": False, "intent": None, "intent20": None, "cluster": f"{d}-held-{l}"} for k, (t, l) in enumerate(held_rows) if fresh(t) and check(t, l.replace("oos_", "", 1))]
        out[f"clinc_{d}"] = {"labels": [n.replace("_", " ") for n in own], "pool": [[t, oindex[l]] for t, l in zip(_lines(sd / d / "train" / "seq.in"), _lines(sd / d / "train" / "label"))], "items": items}
    out["clinc_combined"] = {"labels": [n.replace("_", " ") for n in combined], "pool": [[t, cindex[l]] for d in DOMAINS for t, l in zip(_lines(sd / d / "train" / "seq.in"), _lines(sd / d / "train" / "label"))]}
    for key in ("banking77_oos", "clinc_banking", "clinc_credit_cards"):
        it = out[key]["items"]
        print(f"{key}: in-scope {sum(i['in_scope'] for i in it)} held-out {sum(not i['in_scope'] for i in it)} held-out intents {len({i['cluster'] for i in it if not i['in_scope']})} pool {len(out[key]['pool'])}", flush=True)
    if ITEMS.exists():
        raise RuntimeError("items.json is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(json.dumps(out, indent=1) + NEWLINE, encoding="utf-8")
    print(f"items.json sha256 {hashlib.sha256(ITEMS.read_bytes()).hexdigest()}", flush=True)


def load() -> dict:
    return json.loads(ITEMS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- prompts


def _template(kind: str) -> str:
    from bakeoff import workflow_near_oos as wn

    template = json.loads(wn.RULES_PATH.read_text(encoding="utf-8"))["hosted"]["system_prompt"]
    if kind == "clinc_banking":
        return template.replace("a retail bank's customer-support desk", "a bank's customer-support desk").replace("other banking questions", "other banking or credit-card questions")
    if kind == "clinc_credit_cards":
        return template.replace("a retail bank's customer-support desk", "a credit-card support desk").replace("other banking questions", "other banking or credit-card questions")
    if kind == "clinc_combined":
        return template.replace("a retail bank's customer-support desk", "a bank's customer-support desk for banking and credit cards").replace("other banking questions", "other banking or credit-card questions")
    return template


def _rule_sentence(kind: str) -> str:
    return SCOPE_SENTENCE if kind == "banking77_oos" else SCOPE_SENTENCE.replace("other banking questions", "other banking or credit-card questions")


def jobs(data: dict, split: str, arm: str, items: list[dict], seed: int) -> list[tuple[str, str, str, list[str]]]:
    """(id, system, user, labels) for one split and arm. Labels are the option list the answer is parsed against."""
    from bakeoff import api_baseline as api
    from bakeoff.workflow_retrieval import BM25

    kind = "clinc_combined" if arm.endswith("_20") else split
    block = data[kind] if kind == "clinc_combined" else data[split]
    labels = block["labels"]
    codes = api.codes_for(len(labels))
    system = _template(kind).format(options=NEWLINE.join(f"{c}: {l}" for c, l in zip(codes, labels)))
    pool = [(t, codes[g]) for t, g in block["pool"]]
    index = BM25([t for t, _ in pool]) if arm not in ("zero_shot", "zero_shot_10", "zero_shot_20") else None
    out = []
    for item in items:
        user = "Current message: " + item["text"]
        if index is not None:
            if arm == "random_examples":
                rng = np.random.default_rng([seed, int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16)])
                picked = rng.choice(len(pool), size=20, replace=False).tolist()
            else:
                picked = index.top(item["text"], 20)[::-1]
            lines = NEWLINE.join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
            rule = (NEWLINE + NEWLINE + _rule_sentence(kind)) if arm == "rule_after_examples" else ""
            user = "Similar labelled examples (message => code):" + NEWLINE + lines + rule + NEWLINE + NEWLINE + user
        out.append((item["id"], system, user, labels))
    return out


def _splits_arms():
    yield "banking77_oos", B77_ARMS
    for d in DOMAINS:
        yield f"clinc_{d}", SD_ARMS


def _path(model: str, split: str, arm: str, folder: str = "hosted"):
    return OUT / folder / f"{model}__{split}__{arm}.jsonl"


def _check_items(data: dict, split: str) -> list[dict]:
    """Plumbing only: five training rows of the split and five fixed off-topic sentences; no frozen item is sent."""
    block = data[split]
    used = {i["text"] for i in block["items"]}
    ins = [{"id": f"chk-in-{k}", "text": t} for k, (t, _) in enumerate(block["pool"][:200]) if t not in used][:5]
    return ins + [{"id": f"chk-oos-{k}", "text": t} for k, t in enumerate(["what's the weather like today", "book me a table for two", "play some jazz", "how tall is mount everest", "set an alarm for 7am"])]


def run(stage: str, only: str | None = None) -> None:
    """only: run one model (execution note of 2026-10-16: one process per model, per-process cap)."""
    import httpx

    from bakeoff import api_baseline as api
    from bakeoff import workflow_v4 as v4

    spec = rules()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    cap = spec["cost"]["check_cap_usd"] if stage == "check" else (spec["cost"]["per_model_process_cap_usd"] if only else spec["cost"]["run_cap_usd"])
    budget = api.Budget(api_rules, run_cap=cap)
    data = load()
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client:
        for name in [only] if only else spec["model_order"]:
            model = {**spec["models"][name], "name": name}
            for split, arms in _splits_arms():
                items = _check_items(data, split) if stage == "check" else data[split]["items"]
                for arm in arms:
                    js = jobs(data, split, arm, items, spec["seed"])
                    path = _path(name, split, arm, "check" if stage == "check" else "hosted")
                    try:
                        rows = v4._collect(client, model, [(a, b, c) for a, b, c, _ in js], 24, path, budget)
                    except RuntimeError as error:
                        print(f"    {error}", flush=True)
                        rows = v4._read(path)
                    parsed = [v4.parse(r["text"], len(js[0][3])) for r in rows.values()]
                    print(f"{stage} {name:16s} {split:20s} {arm:20s} {len(rows)}/{len(items)} parse_failed {sum(p['parse_failed'] for p in parsed)} providers {sorted({str(r.get('provider')) for r in rows.values()})} run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)
                    budget.save()
                    if not budget.allowed():
                        print("cap reached; the rest is not run", flush=True)
                        return
    print(f"D2_{stage.upper()}_DONE run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)


def run_jev(stage: str) -> None:
    from bakeoff import jev_rerun as jr

    jspec = jr.rules()
    data = load()
    with jr._client() as client:
        for split, arms in _splits_arms():
            items = _check_items(data, split) if stage == "jev_check" else data[split]["items"]
            for arm in arms:
                kind = "clinc_combined" if arm.endswith("_20") else split
                block = data[kind] if kind == "clinc_combined" else data[split]
                labels, pool = block["labels"], [tuple(p) for p in block["pool"]]
                texts = [(i["id"], i["text"]) for i in items]
                if arm in ("zero_shot", "zero_shot_10", "zero_shot_20"):
                    states = texts
                elif arm == "random_examples":
                    states = []
                    for key, text in texts:
                        rng = np.random.default_rng([rules()["seed"], int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)])
                        lines = [f"{pool[i][0]} => {labels[pool[i][1]]}" for i in rng.choice(len(pool), size=20, replace=False)]
                        states.append((key, "Similar labelled examples (message => request type):" + NEWLINE + NEWLINE.join(lines) + NEWLINE + NEWLINE + "Current message: " + text))
                else:
                    states = jr._states(pool, labels, texts, "retrieval")
                    if arm == "rule_after_examples":
                        states = [(k, s.replace(NEWLINE + NEWLINE + "Current message: ", NEWLINE + NEWLINE + _rule_sentence(split) + NEWLINE + NEWLINE + "Current message: ")) for k, s in states]
                path = OUT / ("jev-check" if stage == "jev_check" else "jev") / f"{split}__{arm}.jsonl"
                rows = jr._collect(client, jspec, labels, jr._questions(jspec, labels, "retrieval"), states, path)
                print(f"{stage} {split:20s} {arm:20s} {len(rows)}/{len(texts)} models {sorted({r['model'] for r in rows.values()})} tokens {np.mean([r['input_tokens'] for r in rows.values()]) if rows else 0:.0f}", flush=True)
    print(f"D2_{stage.upper()}_DONE", flush=True)


# ---------------------------------------------------------------- analysis


def analyze() -> None:
    from bakeoff import jev_rerun as jr
    from bakeoff import posthoc_round3 as p3
    from bakeoff import retrieval_suppression as rs
    from bakeoff import revision_free as rf
    from bakeoff import workflow_v4 as v4
    from bakeoff.confirmatory import _auroc

    spec = rules()
    seed = spec["seed"]
    data = load()
    models = list(spec["models"]) + ["jev-1.13.0"]
    result = {"protocol_id": spec["protocol_id"], "banking77_oos": {}, "clinc_single_domain": {}, "tests": {}}

    def decisions(name, split, arm, items, labels):
        if name == "jev-1.13.0":
            d = jr._decisions(OUT / "jev" / f"{split}__{arm}.jsonl", [i["id"] for i in items], np.asarray([i["in_scope"] for i in items]))
            return d["scope"], d["intent"], np.nan_to_num(d["prob"], nan=0.5)
        rows = v4._read(_path(name, split, arm))
        scope, intent = [], []
        for i in items:
            r = rows.get(i["id"])
            if r is None:
                scope.append(not i["in_scope"])
                intent.append(-1)
            else:
                p = v4.parse(r["text"], len(labels))
                scope.append(p["scope"])
                intent.append(p["intent"])
        return np.asarray(scope, bool), np.asarray(intent), None

    def summary(scope, intent, prob, in_scope, gold):
        out = {"held_out_rejected": float((~scope)[~in_scope].mean()), "in_scope_kept": float(scope[in_scope].mean()), "in_scope_right": float((scope & (intent == gold))[in_scope].mean())}
        d, c = p3.sdt(~scope, in_scope)
        out.update({"d_prime": d, "criterion": c})
        if prob is not None:
            out.update({"auroc": _auroc(prob, in_scope), "mean_prob_held_out": float(prob[~in_scope].mean()), "mean_prob_in_scope": float(prob[in_scope].mean())})
        return out

    # BANKING77-OOS: H1, H2, H3.
    b = data["banking77_oos"]
    items = b["items"]
    in_scope = np.asarray([i["in_scope"] for i in items])
    gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
    clusters = np.asarray([i["cluster"] for i in items])
    ps = {"H1": {}, "H2": {}, "H3": {}, "H4": {}}
    for name in models:
        dec = {arm: decisions(name, "banking77_oos", arm, items, b["labels"]) for arm in B77_ARMS}
        result["banking77_oos"][name] = {arm: summary(*dec[arm], in_scope, gold) for arm in B77_ARMS}
        rej = {arm: (~dec[arm][0])[~in_scope].astype(float) for arm in B77_ARMS}
        t1 = rs.flip_test(rej["zero_shot"] - rej["retrieval"], clusters[~in_scope], seed, True)
        t2 = rs.flip_test(rej["random_examples"] - rej["retrieval"], clusters[~in_scope], seed, False)
        t3 = rs.flip_test(rej["rule_after_examples"] - rej["retrieval"], clusters[~in_scope], seed, True)
        for h, t in (("H1", t1), ("H2", t2), ("H3", t3)):
            result["tests"][f"{h} {name}"] = t
            ps[h][name] = t["p"]
    # CLINC single-domain: H4 (difference of differences, paired on items).
    sd_items = [i for d in DOMAINS for i in data[f"clinc_{d}"]["items"]]
    sd_in = np.asarray([i["in_scope"] for i in sd_items])
    sd_clusters = np.asarray([i["cluster"] for i in sd_items])
    for name in models:
        arms = {}
        for arm in SD_ARMS:
            parts = []
            for d in DOMAINS:
                split = f"clinc_{d}"
                its = data[split]["items"]
                labels = data["clinc_combined"]["labels"] if arm.endswith("_20") else data[split]["labels"]
                parts.append(decisions(name, split, arm, its, labels))
            prob = None if parts[0][2] is None else np.concatenate([p[2] for p in parts])
            arms[arm] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), prob)
        g10 = np.asarray([i["intent"] if i["in_scope"] else -1 for i in sd_items])
        g20 = np.asarray([i["intent20"] if i["in_scope"] else -1 for i in sd_items])
        result["clinc_single_domain"][name] = {arm: summary(*arms[arm], sd_in, g20 if arm.endswith("_20") else g10) for arm in SD_ARMS}
        rej = {arm: (~arms[arm][0])[~sd_in].astype(float) for arm in SD_ARMS}
        dd = (rej["retrieval_20"] - rej["zero_shot_20"]) - (rej["retrieval_10"] - rej["zero_shot_10"])
        t4 = rs.flip_test(dd, sd_clusters[~sd_in], seed, False)
        result["tests"][f"H4 {name}"] = t4
        ps["H4"][name] = t4["p"]
    for h, family in ps.items():
        for name, value in rf.holm(family).items():
            result["tests"][f"{h} {name}"]["holm_p"] = value
    h1 = [result["tests"][f"H1 {m}"] for m in models]
    result["H1_replicates"] = {"significant": sum(t["holm_p"] < 0.05 for t in h1), "same_sign": all(t["difference_points"] > 0 for t in h1) or all(t["difference_points"] < 0 for t in h1)}
    result["H1_replicates"]["replicates"] = bool(result["H1_replicates"]["significant"] >= 4 and result["H1_replicates"]["same_sign"])
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8")
    print("BANKING77-OOS (fresh, clean pool): held-out rejected / in-scope kept / d' / c")
    for name, arms in result["banking77_oos"].items():
        print(f"  {name:16s} " + " | ".join(f"{a}: {100 * v['held_out_rejected']:.1f}/{100 * v['in_scope_kept']:.1f}/{v['d_prime']:.2f}/{v['criterion']:+.2f}" + (f" AUC {v['auroc']:.3f}" if 'auroc' in v else "") for a, v in arms.items()))
    print("CLINC single-domain (validation): held-out rejected / in-scope kept")
    for name, arms in result["clinc_single_domain"].items():
        print(f"  {name:16s} " + " | ".join(f"{a}: {100 * v['held_out_rejected']:.1f}/{100 * v['in_scope_kept']:.1f}" + (f" AUC {v['auroc']:.3f}" if 'auroc' in v else "") for a, v in arms.items()))
    for key, t in result["tests"].items():
        print(f"  TEST {key:24s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f} holm={t['holm_p']:.4f}")
    print("H1", result["H1_replicates"])
    print("D2_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "build":
        build()
    elif stage in ("check", "hosted"):
        run(stage, sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage in ("jev", "jev_check"):
        run_jev(stage)
    else:
        analyze()

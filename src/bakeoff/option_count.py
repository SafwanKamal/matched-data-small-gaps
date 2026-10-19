"""Run E: does the retrieval acceptance shift depend on the number or the relatedness of the options? Rules: option_count_protocol.json.

On CLINC150 same-domain splits (6 task domains, 5 held-out intents each), only the option list changes between arms:
the domain's own 10 labels, plus 10 or 40 labels from other domains that are unrelated to the held-out intents, or
plus 10 that are related to them. The desk description, the retrieval pool and the requests are the same in every arm.
Each option list is asked zero-shot and with 20 retrieved in-scope examples.

Stages: 'build' (frozen items, no API), 'estimate' (token-based cost, no API), 'check' (plumbing, API), 'hosted' (API),
'analyze'.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "option_count_protocol.json"
OUT = ROOT / "artifacts" / "option-count"
ITEMS = OUT / "items.json"
NEWLINE = "\n"
CLINC = ("clinc/clinc_oos", "plus", "155b9c710419136e17307b80d0a13e68cd46b4ec")
DOMAINS_FILE = ROOT / "data" / "clinc-domains" / "domains.json"
DOMAINS = ["banking", "credit_cards", "kitchen_and_dining", "home", "auto_and_commute", "travel", "utility", "work"]
# Test domains: every validation and test utterance of banking and credit_cards was already used (run D took CLINC
# single-domain test items, run D2 its validation items), so those two supply added labels only.
ITEM_DOMAINS = ["kitchen_and_dining", "home", "auto_and_commute", "travel", "utility", "work"]
OPTIONS = ["o10", "o20_far", "o20_near", "o50_far"]
ARM_PARTS = {f"{mode}_{opt}": (mode, opt) for opt in OPTIONS for mode in ("zero_shot", "retrieval")}
ARMS = list(ARM_PARTS)
SEED = 20261019
N_HELD, N_IN_PER_INTENT, N_OUT_PER_INTENT, N_NEAR, N_FAR = 5, 6, 15, 10, 40


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- build


def _clinc() -> dict[str, list[tuple[str, str]]]:
    """{split: [(normalised text, intent name)]} for the pinned CLINC150 'plus' revision, out-of-scope rows dropped."""
    from datasets import load_dataset

    from bakeoff.confirmatory import normalise

    raw = load_dataset(CLINC[0], CLINC[1], revision=CLINC[2])
    names = raw["train"].features["intent"].names
    return {split: [(normalise(r["text"]), names[r["intent"]]) for r in raw[split] if names[r["intent"]] != "oos"] for split in ("train", "validation", "test")}


def _seen() -> set[str]:
    """Normalised texts of every earlier session turn, dev item and experiment item (runs A-D2, G-H, MASSIVE draws)."""
    from bakeoff import retrieval_mechanism as rm
    from bakeoff.confirmatory import normalise

    seen = set(rm._seen())
    d2 = rm.load()
    for key in ("banking77_oos", "clinc_banking", "clinc_credit_cards"):
        seen.update(normalise(i["text"]) for i in d2[key]["items"])
    return seen


def _related(train_by_label: dict[str, list[str]], held: list[str], candidates: list[str]) -> tuple[list[str], list[str], dict[str, float]]:
    """Rank other-domain labels by meaning: the highest cosine similarity between a candidate label's mean training-row
    embedding (all-mpnet-base-v2, as the exploratory dense-retrieval arm) and any held-out intent's. Returns the 10 most
    related, the 40 least related, and every candidate's score."""
    from bakeoff.workflow_retrieval import embed

    def centroid(label: str) -> np.ndarray:
        v = embed(train_by_label[label]).mean(axis=0)
        return v / np.linalg.norm(v)

    held_c = np.stack([centroid(h) for h in held])
    score = {lab: float((held_c @ centroid(lab)).max()) for lab in candidates}
    order = sorted(candidates, key=lambda lab: (-score[lab], lab))
    near, far = order[:N_NEAR], order[::-1][:N_FAR]
    assert not set(near) & set(far)
    return near, far, score


def build() -> None:
    domains = json.loads(DOMAINS_FILE.read_text(encoding="utf-8"))
    assert sorted(domains) == sorted(DOMAINS + ["small_talk", "meta"]) and all(len(v) == 15 for v in domains.values())
    data = _clinc()
    seen = _seen()
    rng = np.random.default_rng(SEED)
    train_by_label: dict[str, list[str]] = {}
    for text, lab in data["train"]:
        train_by_label.setdefault(lab, []).append(text)
    fresh_by_label: dict[str, list[str]] = {}
    for split in ("validation", "test"):
        for text, lab in data[split]:
            if text not in seen:
                fresh_by_label.setdefault(lab, []).append(text)
    out = {"source": {"clinc150": list(CLINC), "domains_json_sha256": hashlib.sha256(DOMAINS_FILE.read_bytes()).hexdigest()}, "domains": {}}
    for d in ITEM_DOMAINS:
        intents = sorted(domains[d])
        held = sorted(rng.choice(intents, size=N_HELD, replace=False).tolist())
        kept = [i for i in intents if i not in held]
        items = []
        for lab in kept:
            fresh = sorted(set(fresh_by_label.get(lab, [])))
            pick = sorted(rng.choice(len(fresh), size=N_IN_PER_INTENT, replace=False).tolist())
            items += [{"id": f"{d}-in-{lab}-{k}", "text": fresh[k], "in_scope": True, "intent": lab, "cluster": f"in-{lab}"} for k in pick]
        for lab in held:
            fresh = sorted(set(fresh_by_label.get(lab, [])))
            pick = sorted(rng.choice(len(fresh), size=N_OUT_PER_INTENT, replace=False).tolist())
            items += [{"id": f"{d}-out-{lab}-{k}", "text": fresh[k], "in_scope": False, "intent": None, "cluster": f"held-{lab}"} for k in pick]
        texts = {i["text"] for i in items}
        pool = [[t, lab] for lab in kept for t in train_by_label[lab] if t not in texts]
        candidates = sorted(lab for other in DOMAINS if other != d for lab in domains[other])
        near, far, score = _related(train_by_label, held, candidates)
        options = {"o10": sorted(kept), "o20_far": sorted(kept + far[:10]), "o20_near": sorted(kept + near), "o50_far": sorted(kept + far)}
        out["domains"][d] = {"held_out_intents": held, "in_scope_intents": kept, "pool": pool, "items": items, "options": options, "relatedness": {lab: round(v, 4) for lab, v in score.items()}}
        print(f"{d:20s} held {held}", flush=True)
        print(f"    near {near}", flush=True)
        print(f"    far(10) {far[:10]}", flush=True)
    if ITEMS.exists():
        raise RuntimeError("items.json is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(json.dumps(out, indent=1) + NEWLINE, encoding="utf-8", newline="\n")
    n = sum(len(v["items"]) for v in out["domains"].values())
    print(f"items {n}; items.json sha256 {hashlib.sha256(ITEMS.read_bytes()).hexdigest()}", flush=True)


def load() -> dict:
    return json.loads(ITEMS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- prompts


def system_prompt(labels: list[str]) -> str:
    """Run D2's template with the bank-desk wording made domain-neutral, so added labels from other domains fit."""
    from bakeoff import api_baseline as api
    from bakeoff import workflow_near_oos as wn

    template = json.loads(wn.RULES_PATH.read_text(encoding="utf-8"))["hosted"]["system_prompt"]
    for old, new in (("a retail bank's customer-support desk", "an assistant"), ("The desk handles", "The assistant handles"), ("including other banking questions", "including related requests")):
        assert template.count(old) == 1, old
        template = template.replace(old, new)
    names = [lab.replace("_", " ") for lab in labels]
    return template.format(options=NEWLINE.join(f"{c}: {n}" for c, n in zip(api.codes_for(len(names)), names)))


def jobs(block: dict, arm: str, items: list[dict]) -> list[tuple[str, str, str, list[str]]]:
    """(id, system, user, labels) for one domain and arm. Retrieved examples come from the same pool in every arm."""
    from bakeoff import api_baseline as api
    from bakeoff.workflow_retrieval import BM25

    mode, opt = ARM_PARTS[arm]
    labels = block["options"][opt]
    code = dict(zip(labels, api.codes_for(len(labels))))
    system = system_prompt(labels)
    pool = block["pool"]
    index = BM25([t for t, _ in pool]) if mode == "retrieval" else None
    out = []
    for item in items:
        user = "Current message: " + item["text"]
        if index is not None:
            picked = index.top(item["text"], 20)[::-1]
            user = "Similar labelled examples (message => code):" + NEWLINE + NEWLINE.join(f"{pool[i][0]} => {code[pool[i][1]]}" for i in picked) + NEWLINE + NEWLINE + user
        out.append((item["id"], system, user, labels))
    return out


def _path(model: str, domain: str, arm: str, folder: str = "hosted"):
    return OUT / folder / f"{model}__{domain}__{arm}.jsonl"


def _check_items(block: dict) -> list[dict]:
    """Plumbing only: five pool rows and five fixed off-topic sentences; no frozen item is sent."""
    ins = [{"id": f"chk-in-{k}", "text": t} for k, (t, _) in enumerate(block["pool"][:5])]
    return ins + [{"id": f"chk-oos-{k}", "text": t} for k, t in enumerate(["what's the weather like today", "book me a table for two", "play some jazz", "how tall is mount everest", "set an alarm for 7am"])]


# ---------------------------------------------------------------- cost estimate (no API)


def estimate() -> None:
    """Token-based cost from run D2's measured tokens per prompt character, per model, at the protocol's list prices."""
    from bakeoff import retrieval_mechanism as rm
    from bakeoff import workflow_v4 as v4

    spec = rules()
    data = load()
    d2 = rm.load()
    ref = rm.jobs(d2, "banking77_oos", "retrieval", d2["banking77_oos"]["items"], rm.rules()["seed"])
    ref_chars = {i: len(s) + len(u) for i, s, u, _ in ref}
    chars = {arm: sum(len(s) + len(u) for d in ITEM_DOMAINS for _, s, u, _ in jobs(data["domains"][d], arm, data["domains"][d]["items"])) for arm in ARMS}
    n_items = sum(len(data["domains"][d]["items"]) for d in ITEM_DOMAINS)
    total = 0.0
    report = {"items": n_items, "arms": len(ARMS), "calls_per_model": n_items * len(ARMS), "models": {}}
    for name in spec["model_order"]:
        rows = v4._read(rm._path(name, "banking77_oos", "retrieval"))
        tpc = np.mean([r["prompt_tokens"] / ref_chars[k] for k, r in rows.items() if r.get("prompt_tokens")])
        out_tokens = np.mean([r.get("completion_tokens") or 0 for r in rows.values()])
        price_in, price_out = spec["models"][name]["price_usd_per_million"]
        prompt_tokens = tpc * sum(chars.values())
        usd = (prompt_tokens * price_in + out_tokens * n_items * len(ARMS) * price_out) / 1e6
        report["models"][name] = {"tokens_per_char": float(tpc), "prompt_tokens_total": float(prompt_tokens), "mean_output_tokens": float(out_tokens), "usd_uncached": float(usd)}
        total += usd
    report["usd_uncached_total"] = total
    (OUT / "estimate.json").write_text(json.dumps(report, indent=1) + NEWLINE, encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- run (API)


def run(stage: str, only: str | None = None) -> None:
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
            for d in (ITEM_DOMAINS[:1] if stage == "check" else ITEM_DOMAINS):
                block = data["domains"][d]
                items = _check_items(block) if stage == "check" else block["items"]
                for arm in ARMS:
                    js = jobs(block, arm, items)
                    path = _path(name, d, arm, "check" if stage == "check" else "hosted")
                    try:
                        rows = v4._collect(client, model, [(a, b, c) for a, b, c, _ in js], 24, path, budget)
                    except RuntimeError as error:
                        print(f"    {error}", flush=True)
                        rows = v4._read(path)
                    parsed = [v4.parse(r["text"], len(js[0][3])) for r in rows.values()]
                    print(f"{stage} {name:16s} {d:20s} {arm:20s} {len(rows)}/{len(items)} parse_failed {sum(p['parse_failed'] for p in parsed)} providers {sorted({str(r.get('provider')) for r in rows.values()})} run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)
                    budget.save()
                    if not budget.allowed():
                        print("cap reached; the rest is not run", flush=True)
                        return
    print(f"E_{stage.upper()}_DONE run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)


# ---------------------------------------------------------------- analysis


def analyze() -> None:
    from bakeoff import posthoc_round3 as p3
    from bakeoff import retrieval_suppression as rs
    from bakeoff import revision_free as rf
    from bakeoff import workflow_v4 as v4

    spec = rules()
    data = load()
    items = [i for d in ITEM_DOMAINS for i in data["domains"][d]["items"]]
    in_scope = np.asarray([i["in_scope"] for i in items])
    clusters = np.asarray([i["cluster"] for i in items])
    result = {"protocol_id": spec["protocol_id"], "summary": {}, "tests": {}}
    ps = {"H1": {}, "H2": {}}
    for name in spec["model_order"]:
        scope = {}
        for arm in ARMS:
            parts = []
            for d in ITEM_DOMAINS:
                block = data["domains"][d]
                labels = block["options"][ARM_PARTS[arm][1]]
                rows = v4._read(_path(name, d, arm))
                for i in block["items"]:
                    r = rows.get(i["id"])
                    parts.append((not i["in_scope"]) if r is None else v4.parse(r["text"], len(labels))["scope"])
            scope[arm] = np.asarray(parts, bool)
        result["summary"][name] = {}
        for arm in ARMS:
            d_prime, crit = p3.sdt(~scope[arm], in_scope)
            result["summary"][name][arm] = {"held_out_rejected": float((~scope[arm])[~in_scope].mean()), "in_scope_kept": float(scope[arm][in_scope].mean()), "d_prime": d_prime, "criterion": crit}
        rej = {arm: (~scope[arm])[~in_scope].astype(float) for arm in ARMS}
        drop = {opt: rej[f"zero_shot_{opt}"] - rej[f"retrieval_{opt}"] for opt in OPTIONS}
        held_clusters = clusters[~in_scope]
        tests = {
            "H1": rs.flip_test(drop["o50_far"] - drop["o10"], held_clusters, SEED, True),
            "H2": rs.flip_test(drop["o20_near"] - drop["o20_far"], held_clusters, SEED, True),
            "S1 drop o10": rs.flip_test(drop["o10"], held_clusters, SEED, False),
            "S2 drop o20_far minus o10": rs.flip_test(drop["o20_far"] - drop["o10"], held_clusters, SEED, False),
            "S3 drop o50_far": rs.flip_test(drop["o50_far"], held_clusters, SEED, False),
            "S4 drop o20_near": rs.flip_test(drop["o20_near"], held_clusters, SEED, False),
        }
        for key, t in tests.items():
            result["tests"][f"{key} {name}"] = t
        ps["H1"][name], ps["H2"][name] = tests["H1"]["p"], tests["H2"]["p"]
    for h, family in ps.items():
        for name, value in rf.holm(family).items():
            result["tests"][f"{h} {name}"]["holm_p"] = value
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8", newline="\n")
    for name, arms in result["summary"].items():
        print(f"  {name:16s} " + " | ".join(f"{a}: {100 * v['held_out_rejected']:.1f}/{100 * v['in_scope_kept']:.1f}" for a, v in arms.items()))
    for key, t in result["tests"].items():
        print(f"  TEST {key:36s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}" + (f" holm={t['holm_p']:.4f}" if "holm_p" in t else ""))
    print("E_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "build":
        build()
    elif stage == "estimate":
        estimate()
    elif stage in ("check", "hosted"):
        run(stage, sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown stage {stage}")

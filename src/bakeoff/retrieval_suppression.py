"""Run D: does in-scope retrieval suppress rejection of same-domain unsupported requests? Rules: retrieval_suppression_protocol.json.

Stages: 'build' (frozen items), 'train' (GPU; CLINC single-domain specialists), 'check' (API, validation items),
'hosted' (API), 'analyze' (processor).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "retrieval_suppression_protocol.json"
OUT = ROOT / "artifacts" / "retrieval-suppression"
ITEMS = OUT / "items.json"
SD = ROOT / "data" / "clinc-single-domain-oos"
SPLITS = ["banking77_oos", "clinc_banking", "clinc_credit_cards"]
ARMS = ["zero_shot", "retrieval", "mitigation_instruction", "mitigation_rows"]
SEEDS = [20261007, 20261015, 20261016]
NEWLINE = "\n"
INSTRUCTION = "The labelled examples show only supported request types. A message that resembles them but asks for something none of the options covers is out of scope: answer scope: no."


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def build() -> None:
    from bakeoff import workflow_clinc as c
    from bakeoff import workflow_near_oos as wn

    data = wn._data()
    sessions = wn._sessions(data)["id"]
    used_in = Counter(t["text"] for s in sessions for t in s if t["in_scope"])
    used_out = Counter(t["text"] for s in sessions for t in s if not t["in_scope"])
    held_names = _lines(wn.DATA / "id-oos" / "test" / "label_original")
    out = {}
    b77 = {
        "labels": data["labels"],
        "pool": [[t, g] for t, g in data["train"]],
        "dev_in": [[it.text, it.gold_index] for it in data["dev_in"]],
        "dev_out": data["dev_out"]["id"],
        "items": [],
    }
    for k, it in enumerate(data["test_in"]):
        if it.text not in used_in:
            b77["items"].append({"id": f"b77oos-in-{k}", "text": it.text, "in_scope": True, "intent": it.gold_index, "cluster": f"in-{it.gold_index}"})
    for k, (text, name) in enumerate(zip(data["test_out"]["id"], held_names)):
        if text not in used_out:
            b77["items"].append({"id": f"b77oos-out-{k}", "text": text, "in_scope": False, "intent": None, "cluster": f"held-{name}"})
    out["banking77_oos"] = b77
    for domain in ("banking", "credit_cards"):
        names = sorted(set(_lines(SD / domain / "train" / "label")))
        index = {name: i for i, name in enumerate(names)}
        pairs = lambda split: list(zip(_lines(SD / domain / split / "seq.in"), _lines(SD / domain / split / "label")))
        block = {
            "labels": [name.replace("_", " ") for name in names],
            "pool": [[t, index[l]] for t, l in pairs("train")],
            "dev_in": [[t, index[l]] for t, l in pairs("valid")],
            "dev_out": _lines(SD / domain / "id-oos" / "valid" / "seq.in")[:100],
            "items": [{"id": f"{domain}-in-{k}", "text": t, "in_scope": True, "intent": index[l], "cluster": f"{domain}-in-{l}"} for k, (t, l) in enumerate(pairs("test"))],
        }
        held = zip(_lines(SD / domain / "id-oos" / "test" / "seq.in"), _lines(SD / domain / "id-oos" / "test" / "label_original"))
        block["items"] += [{"id": f"{domain}-out-{k}", "text": t, "in_scope": False, "intent": None, "cluster": f"{domain}-held-{name}"} for k, (t, name) in enumerate(held)]
        out[f"clinc_{domain}"] = block
    crules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
    out["mitigation_rows"] = c._data(crules)["train_out"][:4]
    for name in SPLITS:
        items = out[name]["items"]
        print(f"{name}: labels {len(out[name]['labels'])} pool {len(out[name]['pool'])} in-scope {sum(i['in_scope'] for i in items)} held-out {sum(not i['in_scope'] for i in items)} held-out intents {len({i['cluster'] for i in items if not i['in_scope']})}", flush=True)
    if ITEMS.exists():
        raise RuntimeError("items.json already exists; it is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(json.dumps(out, indent=1) + NEWLINE, encoding="utf-8")
    print(f"items.json sha256 {hashlib.sha256(ITEMS.read_bytes()).hexdigest()}", flush=True)


def load() -> dict:
    return json.loads(ITEMS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- prompts and requests


def system_prompt(split: str, labels: list[str], arm: str) -> str:
    from bakeoff import api_baseline as api
    from bakeoff import workflow_near_oos as wn

    template = json.loads(wn.RULES_PATH.read_text(encoding="utf-8"))["hosted"]["system_prompt"]
    if split == "clinc_banking":
        template = template.replace("a retail bank's customer-support desk", "a bank's customer-support desk").replace("other banking questions", "other banking or credit-card questions")
    elif split == "clinc_credit_cards":
        template = template.replace("a retail bank's customer-support desk", "a credit-card support desk").replace("other banking questions", "other banking or credit-card questions")
    if arm == "mitigation_instruction":
        first, rest = template.split("\n", 1)
        template = first + " " + INSTRUCTION + "\n" + rest
    return template.format(options=NEWLINE.join(f"{code}: {label}" for code, label in zip(api.codes_for(len(labels)), labels)))


def jobs_for(split: str, block: dict, items: list[dict], arm: str, extra_rows: list[str]) -> list[tuple[str, str, str]]:
    from bakeoff import api_baseline as api
    from bakeoff.workflow_retrieval import BM25

    codes = api.codes_for(len(block["labels"]))
    pool = [(t, codes[g]) for t, g in block["pool"]]
    system = system_prompt(split, block["labels"], arm)
    index = BM25([t for t, _ in pool]) if arm != "zero_shot" else None
    jobs = []
    for item in items:
        user = "Current message: " + item["text"]
        if index is not None:
            lines = [f"{pool[i][0]} => {pool[i][1]}" for i in index.top(item["text"], 20)[::-1]]
            if arm == "mitigation_rows":
                lines = [f"{t} => {'0' * len(codes[0])}" for t in extra_rows] + lines
            user = "Similar labelled examples (message => code):" + NEWLINE + NEWLINE.join(lines) + NEWLINE + NEWLINE + user
        jobs.append((item["id"], system, user))
    return jobs


def _path(model: str, split: str, arm: str, folder: str = "hosted"):
    return OUT / folder / f"{model}__{split}__{arm}.jsonl"


def _collect(client, model: dict, jobs, path, budget) -> dict:
    from bakeoff import workflow_v4 as v4

    try:
        return v4._collect(client, model, jobs, 24, path, budget)
    except RuntimeError as error:
        print(f"    {error}", flush=True)
        return v4._read(path)


def _run(stage: str) -> None:
    import httpx

    from bakeoff import api_baseline as api
    from bakeoff import workflow_v4 as v4

    spec = rules()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    run = spec["runs"]["check" if stage == "check" else "D"]
    budget = api.Budget(api_rules, run_cap=run["run_cap_usd"])
    data = load()
    order = sorted(spec["models"], key=lambda m: spec["runs"]["model_order"].index(m))
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client:
        for name in order:
            model = {**spec["models"][name], "name": name}
            for split in SPLITS:
                block = data[split]
                if stage == "check":
                    items = [{"id": f"dev-in-{k}", "text": t} for k, (t, _) in enumerate(block["dev_in"][:5])] + [{"id": f"dev-out-{k}", "text": t} for k, t in enumerate(block["dev_out"][:5])]
                else:
                    items = block["items"]
                for arm in ARMS:
                    path = _path(name, split, arm, "check" if stage == "check" else "hosted")
                    rows = _collect(client, model, jobs_for(split, block, items, arm, data["mitigation_rows"]), path, budget)
                    parsed = [v4.parse(r["text"], len(block["labels"])) for r in rows.values()]
                    print(f"{stage} {name:16s} {split:18s} {arm:22s} {len(rows)}/{len(items)} parse_failed {sum(p['parse_failed'] for p in parsed)} providers {sorted({str(r.get('provider')) for r in rows.values()})} run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)
                    budget.save()
                    if not budget.allowed():
                        print("cap reached; the rest is not run", flush=True)
                        return
    print(f"RS_{stage.upper()}_DONE run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)


# ---------------------------------------------------------------- specialist reference (GPU)


def _sd_folder(split: str, seed: int):
    return OUT / "specialists" / split / f"seed-{seed}"


def train() -> None:
    from bakeoff import revision_free as rf

    data = load()
    for split in ("clinc_banking", "clinc_credit_cards"):
        block = data[split]
        for seed in SEEDS:
            rf._train(_sd_folder(split, seed), block["labels"], [tuple(p) for p in block["pool"]], [tuple(p) for p in block["dev_in"]], 1e-4, 3, seed)
            print(f"trained {split} seed {seed}", flush=True)
    for split in SPLITS:
        for seed in SEEDS:
            features(split, seed)
        print(f"features {split}", flush=True)
    print("RS_TRAIN_DONE", flush=True)


def features(split: str, seed: int) -> dict:
    cache = OUT / "features" / f"{split}__seed-{seed}.npz"
    if cache.exists():
        return dict(np.load(cache))
    from bakeoff import revision_free as rf

    block = load()[split]
    if split == "banking77_oos":
        folder = rf.domain("banking77_oos/id")["folders"][str(seed)]
    else:
        folder = _sd_folder(split, seed)
    parts = {"train": [t for t, _ in block["pool"]], "dev_in": [t for t, _ in block["dev_in"]], "dev_out": list(block["dev_out"]), "turns": [i["text"] for i in block["items"]], "pool": []}
    out = rf.embed_with(folder, parts)
    out["train_gold"] = np.asarray([g for _, g in block["pool"]], dtype=np.int64)
    out["dev_gold"] = np.asarray([g for _, g in block["dev_in"]], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------- analysis


def _answers(path, items, n_labels: int) -> dict:
    from bakeoff import workflow_v4 as v4

    rows = v4._read(path)
    scope, intent, failed, parse_failed = [], [], 0, 0
    for item in items:
        row = rows.get(item["id"])
        if row is None:
            scope.append(not item["in_scope"])
            intent.append(-1)
            failed += 1
            continue
        p = v4.parse(row["text"], n_labels)
        parse_failed += p["parse_failed"]
        scope.append(p["scope"])
        intent.append(p["intent"])
    return {"scope": np.asarray(scope, bool), "intent": np.asarray(intent), "failed": failed, "parse_failed": parse_failed, "rows": [rows[i["id"]] for i in items if i["id"] in rows]}


def flip_test(diff: np.ndarray, clusters: np.ndarray, seed: int, one_sided: bool, draws: int = 10000) -> dict:
    """Mean paired difference; sign flips of cluster sums (one-sided: greater than 0); cluster-bootstrap 95% interval."""
    keys, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    observed = sums.sum()
    flipped = rng.choice([-1.0, 1.0], size=(draws, len(sums))) @ sums
    p = (1 + np.sum(flipped >= observed - 1e-12)) / (draws + 1) if one_sided else (1 + np.sum(np.abs(flipped) >= abs(observed) - 1e-12)) / (draws + 1)
    picks = rng.integers(0, len(sums), size=(2000, len(sums)))
    boot = 100 * sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    return {"difference_points": float(100 * diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "p": float(p), "clusters": int(len(keys))}


def _cost(rows: list[dict]) -> dict:
    if not rows:
        return {}
    latency = np.asarray([r.get("latency_ms") or np.nan for r in rows], float)
    return {
        "calls": len(rows),
        "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens") or 0 for r in rows])),
        "completion_tokens_mean": float(np.mean([r.get("completion_tokens") or 0 for r in rows])),
        "usd_per_call": float(np.mean([r.get("cost") or 0 for r in rows])),
        "usd_total": float(np.sum([r.get("cost") or 0 for r in rows])),
        "latency_ms_p50": float(np.nanpercentile(latency, 50)),
        "latency_ms_p95": float(np.nanpercentile(latency, 95)),
    }


def analyze() -> None:
    from bakeoff import revision_free as rf

    spec = rules()
    seed = spec["seed"]
    data = load()
    groups = {"banking77_oos": ["banking77_oos"], "clinc_single_domain": ["clinc_banking", "clinc_credit_cards"], "clinc_banking": ["clinc_banking"], "clinc_credit_cards": ["clinc_credit_cards"]}
    result = {"protocol_id": spec["protocol_id"], "groups": {}}
    h1, h2 = {}, {}
    for group, splits in groups.items():
        items = [i for s in splits for i in data[s]["items"]]
        in_scope = np.asarray([i["in_scope"] for i in items])
        gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
        clusters = np.asarray([i["cluster"] for i in items])
        block = {"n_in_scope": int(in_scope.sum()), "n_held_out": int((~in_scope).sum()), "held_out_intents": int(len(set(clusters[~in_scope]))), "models": {}, "tests": {}}
        # Specialist reference: mean over three seeds, prior-weighted threshold.
        ref = []
        for s in SEEDS:
            scope, intent = [], []
            for split in splits:
                sp = rf.specialist(features(split, s), 0.2)
                scope.append(sp["scope"])
                intent.append(sp["intent"])
            scope, intent = np.concatenate(scope), np.concatenate(intent)
            ref.append(((~scope)[~in_scope].mean(), (scope & (intent == gold))[in_scope].mean()))
        block["specialist_reference"] = {"held_out_rejected": float(np.mean([r[0] for r in ref])), "in_scope_right": float(np.mean([r[1] for r in ref]))}
        for name in spec["models"]:
            per_arm = {}
            for arm in ARMS:
                parts = [_answers(_path(name, s, arm), data[s]["items"], len(data[s]["labels"])) for s in splits]
                scope = np.concatenate([p["scope"] for p in parts])
                intent = np.concatenate([p["intent"] for p in parts])
                rows = [r for p in parts for r in p["rows"]]
                per_arm[arm] = {
                    "rejected": (~scope)[~in_scope].astype(float),
                    "right": (scope & (intent == gold))[in_scope].astype(float),
                    "summary": {
                        "held_out_rejected": float((~scope)[~in_scope].mean()),
                        "in_scope_right": float((scope & (intent == gold))[in_scope].mean()),
                        "in_scope_kept": float(scope[in_scope].mean()),
                        "intent_on_kept": float((intent == gold)[in_scope & scope].mean()) if (in_scope & scope).any() else None,
                        "failed": int(sum(p["failed"] for p in parts)),
                        "parse_failed": int(sum(p["parse_failed"] for p in parts)),
                        "cost": _cost(rows),
                    },
                }
            block["models"][name] = {arm: v["summary"] for arm, v in per_arm.items()}
            held_clusters, in_clusters = clusters[~in_scope], clusters[in_scope]
            t = flip_test(per_arm["zero_shot"]["rejected"] - per_arm["retrieval"]["rejected"], held_clusters, seed, True)
            block["tests"][f"H1 {name}: zero-shot minus retrieval rejection"] = t
            if group in ("banking77_oos", "clinc_single_domain"):
                h1[f"{group} | H1 {name}: zero-shot minus retrieval rejection"] = t["p"]
            for arm in ("mitigation_instruction", "mitigation_rows"):
                gain = flip_test(per_arm[arm]["rejected"] - per_arm["retrieval"]["rejected"], held_clusters, seed, True)
                cost_in = flip_test(per_arm[arm]["right"] - per_arm["retrieval"]["right"], in_clusters, seed, False)
                zs, rt, mt = (per_arm[a]["summary"]["held_out_rejected"] for a in ("zero_shot", "retrieval", arm))
                recovered = (mt - rt) / (zs - rt) if zs > rt else None
                block["tests"][f"H2 {name}: {arm} minus retrieval rejection"] = {**gain, "share_of_suppression_recovered": recovered, "in_scope_right_change": cost_in, "works": bool(recovered is not None and recovered >= 0.5 and cost_in["ci95"][0] > -2.0)}
                if group in ("banking77_oos", "clinc_single_domain"):
                    h2[f"{group} | H2 {name}: {arm} minus retrieval rejection"] = gain["p"]
        result["groups"][group] = block
    for family in (h1, h2):
        for label, value in rf.holm(family).items():
            group, key = label.split(" | ")
            result["groups"][group]["tests"][key]["holm_p"] = value
    for group in ("banking77_oos", "clinc_single_domain"):
        tests = result["groups"][group]["tests"]
        h1_rows = [v for k, v in tests.items() if k.startswith("H1")]
        significant = sum(v["holm_p"] < 0.05 for v in h1_rows)
        same_sign = all(v["difference_points"] > 0 for v in h1_rows) or all(v["difference_points"] < 0 for v in h1_rows)
        result["groups"][group]["replicates"] = {"significant_models": significant, "same_sign": same_sign, "replicates": bool(significant >= 3 and same_sign)}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8")
    for group, block in result["groups"].items():
        print(f"\n{group}: {block['n_in_scope']} in-scope, {block['n_held_out']} held-out ({block['held_out_intents']} intents) | specialist rejects {100 * block['specialist_reference']['held_out_rejected']:.1f}%, in-scope right {100 * block['specialist_reference']['in_scope_right']:.1f}%" + (f" | REPLICATES {block['replicates']}" if "replicates" in block else ""))
        for name, arms in block["models"].items():
            print(f"  {name:16s} " + " | ".join(f"{arm}: rej {100 * a['held_out_rejected']:.1f} right {100 * a['in_scope_right']:.1f} ${a['cost'].get('usd_total', 0):.3f}" for arm, a in arms.items()))
        for key, t in block["tests"].items():
            extra = f" recovered {t['share_of_suppression_recovered']:.2f} in-scope {t['in_scope_right_change']['difference_points']:+.2f} [{t['in_scope_right_change']['ci95'][0]:+.2f}, {t['in_scope_right_change']['ci95'][1]:+.2f}] works={t['works']}" if "works" in t and t["share_of_suppression_recovered"] is not None else ""
            print(f"  {key:62s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}" + (f" holm={t['holm_p']:.4f}" if "holm_p" in t else "") + extra)
    print("RS_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    {"build": build, "train": train, "check": lambda: _run("check"), "hosted": lambda: _run("hosted"), "analyze": analyze}[stage]()

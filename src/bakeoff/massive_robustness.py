"""MASSIVE robustness: other held-out-domain draws, cross-source rows, instruction. Rules: massive_robustness_protocol.json.

Stages: 'build', 'train' (GPU), 'check', 'hosted <model>', 'jev', 'analyze'. SENSITIVITY ANALYSIS: items reuse MASSIVE test text.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "massive_robustness_protocol.json"
OUT = ROOT / "artifacts" / "massive-robustness"
ITEMS = OUT / "items.json"
NEWLINE = "\n"
DRAWS = ["draw_1", "draw_2", "draw_3"]
ARMS = ["no_rows", "same_source_rows", "cross_source_rows", "instruction"]
SEEDS = [20261007, 20261015, 20261016]
INSTRUCTION = "The labelled examples show only supported skills. A message that resembles them but asks for something none of the skills covers is out of scope: answer scope: no."


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- build


def build() -> None:
    from bakeoff.confirmatory import normalise
    from bakeoff.massive import load_massive

    spec = rules()
    ms = load_massive()
    all_labels = ms["labels"]
    domain = lambda i: all_labels[i].split(" ")[0]
    out = {}
    for k, name in enumerate(DRAWS):
        d = spec["draws"][name]
        T, R = set(d["T"]), set(d["R"])
        kept = [i for i in range(len(all_labels)) if domain(i) not in T | R]
        remap = {old: new for new, old in enumerate(kept)}
        rng = np.random.default_rng(spec["seed"] + 100 * (k + 1))
        test_in = [it for it in ms["items"]["test"] if it.gold_index in remap]
        test_out = [it for it in ms["items"]["test"] if domain(it.gold_index) in T]
        pick_in = sorted(rng.choice(len(test_in), size=1000, replace=False).tolist())
        pick_out = sorted(rng.choice(len(test_out), size=250, replace=False).tolist())
        dev_out_all = [it for it in ms["items"]["dev"] if domain(it.gold_index) in T]
        train_T = [it for it in ms["items"]["train"] if domain(it.gold_index) in T]
        train_R = [it for it in ms["items"]["train"] if domain(it.gold_index) in R]
        out[name] = {
            "T": sorted(T),
            "R": sorted(R),
            "labels": [all_labels[i] for i in kept],
            "items": [{"id": f"{name}-in-{j}", "text": normalise(test_in[j].text), "in_scope": True, "intent": remap[test_in[j].gold_index], "cluster": f"i{remap[test_in[j].gold_index]}"} for j in pick_in]
            + [{"id": f"{name}-out-{j}", "text": normalise(test_out[j].text), "in_scope": False, "intent": None, "domain": domain(test_out[j].gold_index), "cluster": f"o{j}"} for j in pick_out],
            "train": [[normalise(it.text), remap[it.gold_index]] for it in ms["items"]["train"] if it.gold_index in remap],
            "dev_in": [[normalise(it.text), remap[it.gold_index]] for it in ms["items"]["dev"] if it.gold_index in remap][:400],
            "dev_out": [normalise(dev_out_all[j].text) for j in sorted(rng.choice(len(dev_out_all), size=100, replace=False).tolist())],
            "rows_same": [normalise(train_T[j].text) for j in sorted(rng.choice(len(train_T), size=250, replace=False).tolist())],
            "rows_cross": [normalise(train_R[j].text) for j in sorted(rng.choice(len(train_R), size=250, replace=False).tolist())],
        }
        print(f"{name}: T {sorted(T)} R {sorted(R)} labels {len(kept)} in {len(pick_in)} out {len(pick_out)} train {len(out[name]['train'])} dev {len(out[name]['dev_in'])}/{len(out[name]['dev_out'])}", flush=True)
    if ITEMS.exists():
        raise RuntimeError("items.json is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(json.dumps(out, indent=1) + NEWLINE, encoding="utf-8")
    print(f"items.json sha256 {hashlib.sha256(ITEMS.read_bytes()).hexdigest()}", flush=True)


def load() -> dict:
    return json.loads(ITEMS.read_text(encoding="utf-8"))


def original() -> dict:
    """The confirmatory MASSIVE turns, in the same shape as a draw, for the instruction arm."""
    from bakeoff import confirmatory as cf

    d = cf.load()["massive"]
    turns = cf.turns_of(d)
    return {
        "labels": d["labels"],
        "items": [{"id": t["id"], "text": t["text"], "in_scope": t["in_scope"], "intent": t["intent"], "cluster": f"i{t['intent']}" if t["in_scope"] else f"o{k}"} for k, t in enumerate(turns)],
        "train": [[r["text"], r["intent"]] for r in d["train"]],
        "rows_same": [],
        "rows_cross": [],
    }


# ---------------------------------------------------------------- specialists (GPU)


def _folder(draw: str, seed: int):
    return OUT / "specialists" / draw / f"seed-{seed}"


def train() -> None:
    from bakeoff import revision_free as rf

    data = load()
    for draw in DRAWS:
        d = data[draw]
        for seed in SEEDS:
            rf._train(_folder(draw, seed), d["labels"], [tuple(r) for r in d["train"]], [tuple(r) for r in d["dev_in"]], 2e-4, 6, seed)
            features(draw, seed)
            print(f"trained {draw} seed {seed}", flush=True)
    print("MR_TRAIN_DONE", flush=True)


def features(draw: str, seed: int) -> dict:
    cache = OUT / "features" / f"{draw}__seed-{seed}.npz"
    if cache.exists():
        return dict(np.load(cache))
    scores = OUT / "specialist-scores" / cache.name
    if scores.exists():  # per-turn specialist scores, shipped with the release (bakeoff.specialist_scores)
        return dict(np.load(scores))
    from bakeoff import revision_free as rf

    d = load()[draw]
    parts = {"train": [t for t, _ in d["train"]], "dev_in": [t for t, _ in d["dev_in"]], "dev_out": list(d["dev_out"]), "turns": [i["text"] for i in d["items"]], "pool": []}
    out = rf.embed_with(_folder(draw, seed), parts)
    out["train_gold"] = np.asarray([g for _, g in d["train"]], dtype=np.int64)
    out["dev_gold"] = np.asarray([g for _, g in d["dev_in"]], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------- prompts and requests


def _system(labels: list[str], arm: str) -> str:
    from bakeoff import workflow_massive as m
    from bakeoff import workflow_v4 as v4

    rules_m = json.loads(m.RULES_PATH.read_text(encoding="utf-8"))
    system = v4._system_prompt(rules_m, labels, "current_only", False)
    if arm == "instruction":
        first, rest = system.split("\n", 1)
        system = first + " " + INSTRUCTION + "\n" + rest
    return system


def jobs(d: dict, arm: str, items: list[dict]) -> list[tuple[str, str, str]]:
    from bakeoff import api_baseline as api
    from bakeoff.workflow_retrieval import BM25

    codes = api.codes_for(len(d["labels"]))
    out_code = "0" * len(codes[0])
    pool = [(t, codes[g]) for t, g in d["train"]]
    if arm == "same_source_rows":
        pool += [(t, out_code) for t in d["rows_same"]]
    elif arm == "cross_source_rows":
        pool += [(t, out_code) for t in d["rows_cross"]]
    index = BM25([t for t, _ in pool])
    system = _system(d["labels"], arm)
    result = []
    for item in items:
        lines = NEWLINE.join(f"{pool[i][0]} => {pool[i][1]}" for i in index.top(item["text"], 20)[::-1])
        result.append((item["id"], system, "Similar labelled examples (message => code):" + NEWLINE + lines + NEWLINE + NEWLINE + "Current message: " + item["text"]))
    return result


def _cells():
    data = load()
    for draw in DRAWS:
        for arm in ARMS:
            yield draw, arm, data[draw]
    yield "original", "instruction", original()


def _path(model: str, draw: str, arm: str, folder: str = "hosted"):
    return OUT / folder / f"{model}__{draw}__{arm}.jsonl"


def run(stage: str, only: str | None = None) -> None:
    import httpx

    from bakeoff import api_baseline as api
    from bakeoff import workflow_v4 as v4

    spec = rules()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    cap = spec["cost"]["check_cap_usd"] if stage == "check" else (spec["cost"]["per_model_process_cap_usd"] if only else spec["cost"]["run_cap_usd"])
    budget = api.Budget(api_rules, run_cap=cap)
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client:
        for name in [only] if only else list(spec["models"]):
            model = {**spec["models"][name], "name": name}
            for draw, arm, d in _cells():
                items = d["items"][:5] + [i for i in d["items"] if not i["in_scope"]][:5] if stage == "check" else d["items"]
                js = jobs(d, arm, items)
                path = _path(name, draw, arm, "check" if stage == "check" else "hosted")
                try:
                    rows = v4._collect(client, model, js, 24, path, budget)
                except RuntimeError as error:
                    print(f"    {error}", flush=True)
                    rows = v4._read(path)
                parsed = [v4.parse(r["text"], len(d["labels"])) for r in rows.values()]
                print(f"{stage} {name:14s} {draw:9s} {arm:18s} {len(rows)}/{len(items)} parse_failed {sum(p['parse_failed'] for p in parsed)} providers {sorted({str(r.get('provider')) for r in rows.values()})} run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)
                budget.save()
                if not budget.allowed():
                    print("cap reached; the rest is not run", flush=True)
                    return
    print(f"MR_{stage.upper()}_DONE run ${budget.run:.3f}", flush=True)


def run_jev(stage: str) -> None:
    from bakeoff import jev_rerun as jr

    jspec = jr.rules()
    with jr._client() as client:
        for draw, arm, d in _cells():
            items = d["items"][:5] + [i for i in d["items"] if not i["in_scope"]][:5] if stage == "jev_check" else d["items"]
            labels = d["labels"]
            pool = [tuple(r) for r in d["train"]]
            texts = [(i["id"], i["text"]) for i in items]
            states = jr._states(pool, labels, texts, "retrieval")
            if arm in ("same_source_rows", "cross_source_rows"):
                from bakeoff.workflow_retrieval import BM25

                rows = d["rows_same"] if arm == "same_source_rows" else d["rows_cross"]
                full = [(t, labels[g]) for t, g in pool] + [(t, "out of scope") for t in rows]
                index = BM25([t for t, _ in full])
                states = [(k, "Similar labelled examples (message => request type):" + NEWLINE + NEWLINE.join(f"{full[i][0]} => {full[i][1]}" for i in index.top(t, 20)[::-1]) + NEWLINE + NEWLINE + "Current message: " + t) for k, t in texts]
            questions = jr._questions(jspec, labels, "instruction" if arm == "instruction" else "retrieval")
            path = OUT / ("jev-check" if stage == "jev_check" else "jev") / f"{draw}__{arm}.jsonl"
            got = jr._collect(client, jspec, labels, questions, states, path)
            print(f"{stage} {draw:9s} {arm:18s} {len(got)}/{len(texts)} models {sorted({r['model'] for r in got.values()})}", flush=True)
    print(f"MR_{stage.upper()}_DONE", flush=True)


# ---------------------------------------------------------------- analysis


def _one_sided(t: dict) -> float:
    """One-sided p for 'difference > 0' from the two-sided sign-flip p (symmetric null)."""
    return t["p"] / 2 if t["difference_points"] > 0 else 1 - t["p"] / 2


def analyze() -> None:
    from bakeoff import confirmatory as cf
    from bakeoff import jev_rerun as jr
    from bakeoff import posthoc_round3 as p3
    from bakeoff import revision_free as rf
    from bakeoff import workflow_v4 as v4

    spec = rules()
    seed = spec["seed"]
    result = {"protocol_id": spec["protocol_id"], "status": "sensitivity analysis", "draws": {}, "tests": {}}
    families = {"M3": {}, "M4": {}, "M5": {}}
    models = list(spec["models"]) + ["jev-1.13.0"]

    def answers(model, draw, arm, d):
        items = d["items"]
        in_scope = np.asarray([i["in_scope"] for i in items])
        if model == "jev-1.13.0":
            a = jr._decisions(OUT / "jev" / f"{draw}__{arm}.jsonl", [i["id"] for i in items], in_scope)
            return a["scope"], a["intent"]
        rows = v4._read(_path(model, draw, arm))
        sc, it = [], []
        for i in items:
            r = rows.get(i["id"])
            if r is None:
                sc.append(not i["in_scope"])
                it.append(-1)
            else:
                p = v4.parse(r["text"], len(d["labels"]))
                sc.append(p["scope"])
                it.append(p["intent"])
        return np.asarray(sc, bool), np.asarray(it)

    def correct(items, scope, intent):
        return np.asarray([(bool(s) == i["in_scope"]) and (not i["in_scope"] or int(n) == i["intent"]) for i, s, n in zip(items, scope, intent)], float)

    data = load()
    for draw in DRAWS + ["original"]:
        d = data[draw] if draw != "original" else original()
        items = d["items"]
        in_scope = np.asarray([i["in_scope"] for i in items])
        clusters = np.asarray([i["cluster"] for i in items])
        arms_here = ARMS if draw != "original" else ["instruction"]
        block = {"systems": {}, "rejected_by_domain": {}}
        c = {}
        if draw != "original":
            sp = {s: rf.specialist(features(draw, s), 0.2) for s in SEEDS}
            c["specialist"] = np.mean([correct(items, sp[s]["scope"], sp[s]["intent"]) for s in SEEDS], axis=0)
            spec_scope = np.mean([sp[s]["scope"] for s in SEEDS], axis=0) >= 0.5
            block["systems"]["specialist"] = {"weighted": cf.weighted(c["specialist"], in_scope), "held_out_rejected": float((~spec_scope)[~in_scope].mean()), "in_scope_kept": float(spec_scope[in_scope].mean())}
            doms = np.asarray([i.get("domain", "") for i in items])
            block["rejected_by_domain"]["specialist"] = {dom: float((~spec_scope)[doms == dom].mean()) for dom in d["T"]}
        else:
            for run, model in (("A", "gemma-4-31b"), ("B", "qwen3.5-27b")):
                h = cf._hosted_answers(cf.OUT / "hosted" / f"{run}__massive__{model}.jsonl", cf.turns_of(cf.load()["massive"]), len(d["labels"]))
                c[f"{model} no_rows"] = correct(items, h["scope"], h["intent"])
            j = jr._decisions(jr.OUT / "J2" / "massive__retrieval.jsonl", [i["id"] for i in items], in_scope)
            c["jev-1.13.0 no_rows"] = correct(items, j["scope"], j["intent"])
        for model in models:
            for arm in arms_here:
                sc, it = answers(model, draw, arm, d)
                key = f"{model} {arm}"
                c[key] = correct(items, sc, it)
                dp, cc = p3.sdt(~sc, in_scope)
                block["systems"][key] = {"weighted": cf.weighted(c[key], in_scope), "held_out_rejected": float((~sc)[~in_scope].mean()), "in_scope_kept": float(sc[in_scope].mean()), "d_prime": dp, "criterion": cc}
                if draw != "original":
                    doms = np.asarray([i.get("domain", "") for i in items])
                    block["rejected_by_domain"][key] = {dom: float((~sc)[doms == dom].mean()) for dom in d["T"]}
        tests = {}
        if draw != "original":
            for model in models:
                tests[f"M1 specialist vs {model} no_rows"] = cf.weighted_test(c["specialist"], c[f"{model} no_rows"], in_scope, clusters, seed)
        for model in models:
            for arm, fam in (("same_source_rows", "M3"), ("cross_source_rows", "M4"), ("instruction", "M5")):
                if f"{model} {arm}" in c and f"{model} no_rows" in c:
                    t = cf.weighted_test(c[f"{model} {arm}"], c[f"{model} no_rows"], in_scope, clusters, seed)
                    t["p_one_sided"] = _one_sided(t)
                    tests[f"{fam} {model} {arm} vs no_rows"] = t
                    if model != "jev-1.13.0":
                        families[fam][f"{draw} | {fam} {model} {arm} vs no_rows"] = t["p_one_sided"]
        block["tests"] = tests
        result["draws"][draw] = block
    for fam, ps in families.items():
        for label, value in rf.holm(ps).items():
            draw, key = label.split(" | ")
            result["draws"][draw]["tests"][key]["holm_p_one_sided"] = value
    m1 = [result["draws"][dr]["tests"]["M1 specialist vs gemma-4-31b no_rows"]["difference_points"] for dr in DRAWS]
    result["M1_sign_count"] = {"positive": int(sum(v > 0 for v in m1)), "values": m1}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8")
    for draw, block in result["draws"].items():
        print(f"\n{draw}")
        for key, s in block["systems"].items():
            print(f"  {key:36s} weighted {100 * s['weighted']:.1f} rejected {100 * s['held_out_rejected']:.1f} kept {100 * s['in_scope_kept']:.1f}")
        for key, v in block["rejected_by_domain"].items():
            print(f"  by domain {key:30s} " + " ".join(f"{dom} {100 * x:.0f}" for dom, x in v.items()))
        for key, t in block["tests"].items():
            print(f"  TEST {key:44s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}" + (f" holm(one-sided)={t['holm_p_one_sided']:.4f}" if "holm_p_one_sided" in t else ""))
    print("M1", result["M1_sign_count"])
    print("MR_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "build":
        build()
    elif stage == "train":
        train()
    elif stage in ("check", "hosted"):
        run(stage, sys.argv[2] if len(sys.argv) > 2 else None)
    elif stage in ("jev", "jev_check"):
        run_jev(stage)
    else:
        analyze()

"""Confirmatory run on fresh sessions. Rules: confirmatory_protocol.json.

Stages: 'build' (processor; frozen items), 'train' (GPU; specialists), 'features' (GPU), 'check' (API, dev items),
'hosted <run>' (API; run in A, B, C, I), 'analyze' (processor), 'figure' (processor).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "confirmatory_protocol.json"
OUT = ROOT / "artifacts" / "confirmatory"
ITEMS = OUT / "items.json"
SETS = ["banking77", "massive", "clinc150"]
SEEDS = [20261007, 20261015, 20261016]
NEWLINE = "\n"


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']|_", " ", text.lower())).strip()


# ---------------------------------------------------------------- build (frozen items)


def _mix(rng, in_items: list[dict], out_items: list[dict], prefix: str) -> list[list[dict]]:
    """Five-turn sessions using every item once; out-of-scope positions drawn exactly; repeat rule of workflow v1."""
    by_label: dict[int, list[dict]] = {}
    for item in in_items:
        by_label.setdefault(item["intent"], []).append(item)
    for group in by_label.values():
        rng.shuffle(group)
    out_pool = list(out_items)
    rng.shuffle(out_pool)
    total = len(in_items) + len(out_items)
    is_out = np.zeros(total, bool)
    is_out[rng.permutation(total)[: len(out_items)]] = True
    sessions, turns, last = [], [], None
    for position in range(total):
        index = len(turns)
        if is_out[position]:
            item = out_pool.pop()
            turns.append({"id": f"{prefix}-s{len(sessions)}-t{index}", "text": item["text"], "orig_text": item["orig_text"], "source": item["source"], "in_scope": False, "intent": None, "repeat": None})
        else:
            available = [label for label, group in by_label.items() if group]
            if last is not None and by_label.get(last) and rng.random() < 0.3:
                label = last
            else:
                choices = [value for value in available if value != last] or available
                label = int(rng.choice(choices))
            item = by_label[label].pop()
            turns.append({"id": f"{prefix}-s{len(sessions)}-t{index}", "text": item["text"], "orig_text": item["orig_text"], "source": item["source"], "in_scope": True, "intent": label, "repeat": None if last is None else bool(label == last)})
            last = label
        if len(turns) == 5:
            sessions.append(turns)
            turns, last = [], None
    if turns:
        sessions.append(turns)
    return sessions


def _row(text: str, source: str, intent: int | None = None) -> dict:
    return {"text": normalise(text), "orig_text": text, "source": source, "intent": intent}


def build() -> None:
    from bakeoff import workflow as v1
    from bakeoff import workflow_clinc as c
    from bakeoff import workflow_massive as m
    from bakeoff.data import load_items
    from bakeoff.massive import load_massive
    from bakeoff.protocol import load_protocol
    from bakeoff.train_ettin import _train_rows

    spec = rules()
    seed = spec["seed"]
    rng = np.random.default_rng(seed)
    used = lambda sessions, inside: Counter(t["text"] for s in sessions for t in s if t["in_scope"] == inside)

    # Banking77 and the CLINC150 out-of-scope test queries no earlier session used.
    labels_b, b_test, b_dev, oos_test, oos_dev = v1._sources()
    v1_sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    m_sessions = json.loads((m.OUT / "sessions.json").read_text(encoding="utf-8"))
    crules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
    cdata = c._data(crules)
    c_sessions = c._sessions(crules, cdata)
    b_used = used(v1_sessions, True) + used(m_sessions, False)
    oos_used = used(v1_sessions, False) + used(c_sessions, False)
    oos_unused = [t for t in oos_test if t not in oos_used]
    order = rng.permutation(len(oos_unused))
    oos_b = [oos_unused[i] for i in order[:184]]
    oos_c = [oos_unused[i] for i in order[184:]]
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    b_train = _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"]))
    clinc_oos_train = cdata["train_out"]

    data = {}
    data["banking77"] = {
        "labels": labels_b,
        "rules": "workflow_v4_protocol.json",
        "in": [_row(it.text, f"banking77-test-{it.item_id}", it.gold_index) for it in b_test if it.text not in b_used],
        "out": [_row(t, "clinc150-oos-test") for t in oos_b],
        "train": [_row(t, "banking77-train", g) for t, g in b_train],
        "dev_in": [_row(it.text, "banking77-dev", it.gold_index) for it in b_dev],
        "dev_out": [_row(t, "clinc150-oos-validation") for t in oos_dev],
        "rows": [_row(t, "clinc150-oos-train") for t in clinc_oos_train],
    }
    c_used = used(c_sessions, True)
    c_unused = [it for it in cdata["test_in"] if it.text not in c_used]
    pick = sorted(rng.choice(len(c_unused), size=1665, replace=False).tolist())
    data["clinc150"] = {
        "labels": cdata["labels"],
        "rules": "workflow_clinc_protocol.json",
        "in": [_row(c_unused[i].text, f"clinc150-test-{c_unused[i].row}", c_unused[i].gold_index) for i in pick],
        "out": [_row(t, "clinc150-oos-test") for t in oos_c],
        "train": [_row(it.text, "clinc150-train", it.gold_index) for it in cdata["train_in"]],
        "dev_in": [_row(it.text, "clinc150-validation", it.gold_index) for it in cdata["dev_in"]],
        "dev_out": [_row(t, "clinc150-oos-validation") for t in cdata["dev_out"]],
        "rows": [_row(t, "clinc150-oos-train") for t in clinc_oos_train],
    }

    # MASSIVE with held-out whole domains as native out-of-scope.
    ms = load_massive()
    all_labels = ms["labels"]
    domain = lambda i: all_labels[i].split(" ")[0]
    eligible = sorted({domain(i) for i in range(len(all_labels))} - {"general", "qa"})
    held = sorted(np.random.default_rng(seed).choice(eligible, size=3, replace=False).tolist())
    assert held == spec["sets"]["massive"]["held_out_domains"], held
    kept = [i for i in range(len(all_labels)) if domain(i) not in held]
    remap = {old: new for new, old in enumerate(kept)}
    m_used = used(m_sessions, True)
    test = ms["items"]["test"]
    held_test = [it for it in test if it.text not in m_used and domain(it.gold_index) in held]
    held_train = [it for it in ms["items"]["train"] if domain(it.gold_index) in held]
    held_dev = [it for it in ms["items"]["dev"] if domain(it.gold_index) in held]
    out_pick = rng.choice(len(held_test), size=243, replace=False)
    dev_pick = rng.choice(len(held_dev), size=100, replace=False)
    row_pick = rng.choice(len(held_train), size=250, replace=False)
    data["massive"] = {
        "labels": [all_labels[i] for i in kept],
        "held_out_domains": held,
        "rules": "workflow_massive_protocol.json",
        "in": [_row(it.text, f"massive-test-{it.row}", remap[it.gold_index]) for it in test if it.text not in m_used and domain(it.gold_index) not in held],
        "out": [_row(held_test[i].text, f"massive-test-{held_test[i].row}") for i in sorted(out_pick.tolist())],
        "train": [_row(it.text, "massive-train", remap[it.gold_index]) for it in ms["items"]["train"] if domain(it.gold_index) not in held],
        "dev_in": [_row(it.text, "massive-dev", remap[it.gold_index]) for it in ms["items"]["dev"] if domain(it.gold_index) not in held][:400],
        "dev_out": [_row(held_dev[i].text, "massive-dev-held-out") for i in sorted(dev_pick.tolist())],
        "rows": [_row(held_train[i].text, "massive-train-held-out") for i in sorted(row_pick.tolist())],
    }
    for name in SETS:
        d = data[name]
        d["sessions"] = _mix(rng, d["in"], d["out"], name)
        del d["in"], d["out"]
        turns = [t for s in d["sessions"] for t in s]
        print(f"{name}: labels {len(d['labels'])} sessions {len(d['sessions'])} turns {len(turns)} out-of-scope {sum(not t['in_scope'] for t in turns)} train {len(d['train'])} dev {len(d['dev_in'])}/{len(d['dev_out'])} rows {len(d['rows'])}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if ITEMS.exists():
        raise RuntimeError("items.json already exists; it is frozen")
    ITEMS.write_text(json.dumps(data, indent=1) + NEWLINE, encoding="utf-8")
    print(f"items.json sha256 {hashlib.sha256(ITEMS.read_bytes()).hexdigest()}", flush=True)


def load() -> dict:
    return json.loads(ITEMS.read_text(encoding="utf-8"))


def turns_of(d: dict) -> list[dict]:
    return [t for s in d["sessions"] for t in s]


# ---------------------------------------------------------------- specialists (GPU)


def _folder(name: str, seed: int, arm: str = "normalised"):
    return OUT / "specialists" / arm / name / f"seed-{seed}"


def _setting(name: str) -> tuple[float, int]:
    return {"banking77": (1e-4, 6), "clinc150": (5e-5, 3), "massive": (2e-4, 6)}[name]


def train() -> None:
    from bakeoff import revision_free as rf

    data = load()
    for name in SETS:
        d = data[name]
        lr, epochs = _setting(name)
        dev = [(r["text"], r["intent"]) for r in d["dev_in"]]
        for seed in SEEDS:
            rf._train(_folder(name, seed), d["labels"], [(r["text"], r["intent"]) for r in d["train"]], dev, lr, epochs, seed)
            print(f"trained {name} seed {seed}", flush=True)
        # Original-case arm: one seed on the un-normalised text.
        rf._train(_folder(name, SEEDS[0], "original"), d["labels"], [(r["orig_text"], r["intent"]) for r in d["train"]], [(r["orig_text"], r["intent"]) for r in d["dev_in"]], lr, epochs, SEEDS[0])
        print(f"trained {name} original-case", flush=True)
    print("CONFIRMATORY_TRAIN_DONE", flush=True)


def features(name: str, seed: int, arm: str = "normalised") -> dict:
    cache = OUT / "features" / f"{name}__{arm}__seed-{seed}.npz"
    if cache.exists():
        return dict(np.load(cache))
    from bakeoff import revision_free as rf

    d = load()[name]
    key = "text" if arm == "normalised" else "orig_text"
    parts = {
        "train": [r[key] for r in d["train"]],
        "dev_in": [r[key] for r in d["dev_in"]],
        "dev_out": [r[key] for r in d["dev_out"]],
        "turns": [t[key] for t in turns_of(d)],
        "pool": [r[key] for r in d["rows"]],
    }
    out = rf.embed_with(_folder(name, seed, arm), parts)
    out["train_gold"] = np.asarray([r["intent"] for r in d["train"]], dtype=np.int64)
    out["dev_gold"] = np.asarray([r["intent"] for r in d["dev_in"]], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def stage_features() -> None:
    for name in SETS:
        for seed in SEEDS:
            features(name, seed)
        features(name, SEEDS[0], "original")
        print(f"features {name}", flush=True)
    print("CONFIRMATORY_FEATURES_DONE", flush=True)


# ---------------------------------------------------------------- hosted (API)


def _prompt_rules(d: dict) -> dict:
    return json.loads((ROOT / d["rules"]).read_text(encoding="utf-8"))


def _system(d: dict) -> str:
    from bakeoff import workflow_v4 as v4

    return v4._system_prompt(_prompt_rules(d), d["labels"], "current_only", False)


def _pool(d: dict, with_rows: bool) -> list[tuple[str, str]]:
    from bakeoff import api_baseline as api

    codes = api.codes_for(len(d["labels"]))
    pool = [(r["text"], codes[r["intent"]]) for r in d["train"]]
    if with_rows:
        pool += [(r["text"], "0" * len(codes[0])) for r in d["rows"]]
    return pool


def jobs_for(d: dict, texts: list[tuple[str, str]], with_rows: bool) -> list[tuple[str, str, str]]:
    from bakeoff.workflow_retrieval import BM25

    pool = _pool(d, with_rows)
    index = BM25([text for text, _ in pool])
    system = _system(d)
    out = []
    for key, text in texts:
        picked = index.top(text, 20)[::-1]
        lines = NEWLINE.join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
        out.append((key, system, "Similar labelled examples (message => code):" + NEWLINE + lines + NEWLINE + NEWLINE + "Current message: " + text))
    return out


def _models(spec: dict, run: str) -> list[dict]:
    g = spec["general_models"]
    return {"A": [g["primary"]], "B": [g["secondary"]], "C": [g["primary"]], "I": g["frontier_models"]}[run]


def _collect(client, model: dict, jobs, path, budget) -> dict:
    from bakeoff import workflow_v4 as v4

    try:
        return v4._collect(client, model, jobs, 24, path, budget)
    except RuntimeError as error:
        print(f"    {error}", flush=True)
        return v4._read(path)


def hosted(run: str) -> None:
    import httpx

    from bakeoff import api_baseline as api

    spec = rules()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    budget = api.Budget(api_rules, run_cap=spec["runs"][run]["run_cap_usd"])
    data = load()
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client:
        for model, name in ((m, n) for m in _models(spec, run) for n in SETS):
            d = data[name]
            sessions = d["sessions"][: spec["runs"][run].get("sessions_per_set") or len(d["sessions"])]
            turns = [t for s in sessions for t in s]
            jobs = jobs_for(d, [(t["id"], t["text"]) for t in turns], run == "C")
            print(f"  run {run} {name} {model['name']} {len(jobs)} turns", flush=True)
            rows = _collect(client, model, jobs, OUT / "hosted" / f"{run}__{name}__{model['name']}.jsonl", budget)
            print(f"run {run} {name}: {len(rows)}/{len(jobs)} answered, run ${budget.run:.3f}, project ${budget.total:.3f}", flush=True)
            budget.save()
            if not budget.allowed():
                print(f"run {run}: cap reached; remaining sets not run", flush=True)
                break
    budget.save()
    print(f"CONFIRMATORY_HOSTED_{run}_DONE run ${budget.run:.3f} project ${budget.total:.3f}", flush=True)


def check() -> None:
    """Plumbing check on dev items only; not reported as results."""
    import httpx

    from bakeoff import api_baseline as api
    from bakeoff import workflow_v4 as v4

    spec = rules()
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    budget = api.Budget(api_rules, run_cap=spec["runs"]["check"]["run_cap_usd"])
    data = load()
    g = spec["general_models"]
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=120.0) as client:
        for model in [g["primary"], g["secondary"], *g["frontier_models"]]:
            for name in SETS:
                d = data[name]
                texts = [(f"dev-in-{i}", r["text"]) for i, r in enumerate(d["dev_in"][:8])] + [(f"dev-out-{i}", r["text"]) for i, r in enumerate(d["dev_out"][:2])]
                path = OUT / "check" / f"{name}__{model['name']}.jsonl"
                rows = _collect(client, model, jobs_for(d, texts, False), path, budget)
                parsed = [v4.parse(r["text"], len(d["labels"])) for r in rows.values()]
                print(
                    f"check {model['name']:22s} {name:9s} answered {len(rows)}/10 parse_failed {sum(p['parse_failed'] for p in parsed)} "
                    f"providers {sorted({r.get('provider') for r in rows.values()})} completion tokens max {max([r.get('completion_tokens') or 0 for r in rows.values()] or [0])} "
                    f"cost ${sum(r['cost'] for r in rows.values()):.4f} sample {list(rows.values())[0]['text']!r}" if rows else f"check {model['name']} {name}: no answers",
                    flush=True,
                )
    budget.save()
    print(f"CONFIRMATORY_CHECK_DONE ${budget.run:.4f}", flush=True)


# ---------------------------------------------------------------- analysis


def weighted_test(a: np.ndarray, b: np.ndarray, in_scope: np.ndarray, clusters: np.ndarray, seed: int, draws: int = 10000) -> dict:
    """Paired difference in 0.8 x in-scope accuracy + 0.2 x out-of-scope accuracy; sign flips and bootstrap by cluster."""
    n_in, n_out = in_scope.sum(), (~in_scope).sum()
    weight = np.where(in_scope, 0.8 / n_in, 0.2 / n_out)
    diff = (a - b).astype(float)
    keys, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=weight * diff)
    observed = float(sums.sum())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(draws, len(sums)))
    p = (1 + np.sum(np.abs(flips @ sums) >= abs(observed) - 1e-12)) / (draws + 1)
    in_sum = np.bincount(inverse, weights=diff * in_scope)
    in_n = np.bincount(inverse, weights=in_scope.astype(float))
    out_sum = np.bincount(inverse, weights=diff * ~in_scope)
    out_n = np.bincount(inverse, weights=(~in_scope).astype(float))
    picks = rng.integers(0, len(keys), size=(2000, len(keys)))
    boot = 100 * (0.8 * in_sum[picks].sum(1) / np.maximum(in_n[picks].sum(1), 1) + 0.2 * out_sum[picks].sum(1) / np.maximum(out_n[picks].sum(1), 1))
    return {
        "difference_points": 100 * observed,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
        "p": float(p),
        "clusters": int(len(keys)),
    }


def weighted(correct: np.ndarray, in_scope: np.ndarray) -> float:
    return float(0.8 * correct[in_scope].mean() + 0.2 * correct[~in_scope].mean())


def _hosted_answers(path, turns, n_labels: int) -> dict:
    """Parsed answers; a turn without a successful reply is scored wrong (scope opposite to gold, no intent)."""
    from bakeoff import workflow_v4 as v4

    rows = v4._read(path)
    scope, intent, failed, parse_failed, kept = [], [], 0, 0, []
    for t in turns:
        row = rows.get(t["id"])
        if row is None:
            scope.append(not t["in_scope"])
            intent.append(-1)
            failed += 1
            continue
        p = v4.parse(row["text"], n_labels)
        parse_failed += p["parse_failed"]
        scope.append(p["scope"])
        intent.append(p["intent"])
        kept.append(row)
    return {"scope": np.asarray(scope, bool), "intent": np.asarray(intent), "failed": failed, "parse_failed": parse_failed, "rows": kept, "by_id": rows}


def cost_block(rows: list[dict], turns_per_session: float, n_turns: int, session_success: float | None, turn_share: float = 1.0) -> dict:
    """Tokens, USD and latency of the hosted calls; turn_share scales per-turn cost when only some turns call the model."""
    if not rows:
        return {}
    cost = np.asarray([r.get("cost") or 0.0 for r in rows])
    latency = np.asarray([r.get("latency_ms") or np.nan for r in rows], float)
    prompt = np.asarray([r.get("prompt_tokens") or 0 for r in rows], float)
    usd_turn = float(cost.mean()) * turn_share
    usd_1000 = 1000.0 * usd_turn * turns_per_session
    return {
        "calls": len(rows),
        "prompt_tokens_mean": float(prompt.mean()),
        "completion_tokens_mean": float(np.mean([r.get("completion_tokens") or 0 for r in rows])),
        "cached_share": float(sum(r.get("cached_tokens") or 0 for r in rows) / max(1.0, prompt.sum())),
        "usd_per_turn": usd_turn,
        "usd_per_1000_sessions": usd_1000,
        "usd_per_correct_session": (usd_1000 / 1000.0 / session_success) if session_success else None,
        "latency_ms_p50": float(np.nanpercentile(latency, 50)),
        "latency_ms_p95": float(np.nanpercentile(latency, 95)),
        "latency_ms_p99": float(np.nanpercentile(latency, 99)),
        "providers": dict(Counter(r.get("provider") for r in rows)),
    }


def _summary(turns, scope, intent, correct, in_scope) -> dict:
    gold = np.asarray([t["intent"] if t["in_scope"] else -1 for t in turns])
    return {
        "weighted": weighted(correct, in_scope),
        "turn_correct": float(correct.mean()),
        "scope_accuracy": float((scope == in_scope).mean()),
        "in_scope_kept": float(scope[in_scope].mean()),
        "out_of_scope_caught": float((~scope)[~in_scope].mean()),
        "intent_on_in_scope": float((intent[in_scope] == gold[in_scope]).mean()),
    }


def _auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def analyze_set(name: str, d: dict, spec: dict) -> dict:
    from bakeoff import revision_free as rf

    seed = spec["seed"]
    sessions = d["sessions"]
    turns = turns_of(d)
    n = len(turns)
    in_scope = np.asarray([t["in_scope"] for t in turns])
    clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
    per_session = n / len(sessions)
    labels = d["labels"]
    tc = lambda scope, intent: rf.turn_correct(turns, scope, intent).astype(float)
    sess = lambda scope, intent: float(rf._sessions_ok(sessions, scope, intent).mean())
    out = {"n_turns": n, "n_in": int(in_scope.sum()), "n_out": int((~in_scope).sum()), "n_sessions": len(sessions), "out_of_scope_share": float((~in_scope).mean()), "systems": {}, "tests": {}, "costs": {}}
    words = np.asarray([len(t["text"].split()) for t in turns], float)
    out["length_auroc"] = max(_auroc(words, ~in_scope), 1 - _auroc(words, ~in_scope))

    feats = {s: features(name, s) for s in SEEDS}
    hosted = OUT / "hosted"
    g = spec["general_models"]
    llm = {}
    for run, model in (("A", g["primary"]), ("B", g["secondary"]), ("C", g["primary"])):
        path = hosted / f"{run}__{name}__{model['name']}.jsonl"
        if path.exists():
            llm[run] = _hosted_answers(path, turns, len(labels))

    for rule_name, rate in (("prior_0.2", 0.2), ("prior_0.1", 0.1), ("prior_0.3", 0.3)):
        spec_by_seed = {s: rf.specialist(feats[s], rate) for s in SEEDS}
        block = {"systems": {}, "correct": {}}
        c_spec = np.mean([tc(spec_by_seed[s]["scope"], spec_by_seed[s]["intent"]) for s in SEEDS], axis=0)
        block["correct"]["specialist"] = c_spec
        block["systems"]["specialist"] = {
            **{k: float(np.mean([_summary(turns, spec_by_seed[s]["scope"], spec_by_seed[s]["intent"], tc(spec_by_seed[s]["scope"], spec_by_seed[s]["intent"]), in_scope)[k] for s in SEEDS])) for k in ("weighted", "turn_correct", "scope_accuracy", "in_scope_kept", "out_of_scope_caught", "intent_on_in_scope")},
            "session_success": float(np.mean([sess(spec_by_seed[s]["scope"], spec_by_seed[s]["intent"]) for s in SEEDS])),
            "detectors": [spec_by_seed[s]["detector"] for s in SEEDS],
        }
        for run in ("A", "B"):
            if run not in llm:
                continue
            L = llm[run]
            model_name = (g["primary"] if run == "A" else g["secondary"])["name"]
            c_llm = tc(L["scope"], L["intent"])
            block["correct"][model_name] = c_llm
            block["systems"][model_name] = {**_summary(turns, L["scope"], L["intent"], c_llm, in_scope), "session_success": sess(L["scope"], L["intent"]), "failed_requests": L["failed"], "parse_failed": L["parse_failed"]}
            variants = {"combination": [], "cascade_0.4": [], "model scope + specialist intent": []}
            for s in SEEDS:
                sp = spec_by_seed[s]
                escalate = np.zeros(n, bool)
                escalate[np.argsort(sp["top"], kind="stable")[: int(round(0.4 * n))]] = True
                for key, (sc, it) in {
                    "combination": (L["scope"], np.where(sp["top"] >= 0.9, sp["intent"], L["intent"])),
                    "cascade_0.4": (np.where(escalate, L["scope"], sp["scope"]), np.where(escalate, L["intent"], sp["intent"])),
                    "model scope + specialist intent": (L["scope"], sp["intent"]),
                }.items():
                    variants[key].append((tc(sc, it), sess(sc, it), _summary(turns, sc, it, tc(sc, it), in_scope)))
            for key, rows_ in variants.items():
                c = np.mean([r[0] for r in rows_], axis=0)
                block["correct"][f"{model_name}: {key}"] = c
                block["systems"][f"{model_name}: {key}"] = {**{k: float(np.mean([r[2][k] for r in rows_])) for k in rows_[0][2]}, "session_success": float(np.mean([r[1] for r in rows_]))}
        out["systems"][rule_name] = block["systems"]
        if rule_name != "prior_0.2":
            for key in [k for k in block["correct"] if k != "specialist"][:1]:
                out["tests"][f"{rule_name} | specialist vs {key}"] = weighted_test(block["correct"]["specialist"], block["correct"][key], in_scope, clusters, seed)
            continue
        C = block["correct"]
        for model_name in [m["name"] for m in (g["primary"], g["secondary"]) if m["name"] in C]:
            out["tests"][f"specialist vs {model_name}"] = weighted_test(C["specialist"], C[model_name], in_scope, clusters, seed)
            out["tests"][f"combination vs {model_name}"] = weighted_test(C[f"{model_name}: combination"], C[model_name], in_scope, clusters, seed)
            out["tests"][f"cascade_0.4 vs specialist ({model_name})"] = weighted_test(C[f"{model_name}: cascade_0.4"], C["specialist"], in_scope, clusters, seed)
            out["tests"][f"cascade_0.4 vs {model_name}"] = weighted_test(C[f"{model_name}: cascade_0.4"], C[model_name], in_scope, clusters, seed)
            # Pre-registered decomposition of specialist minus model: scope part and intent part.
            mixed = C[f"{model_name}: model scope + specialist intent"]
            out["tests"][f"decomposition vs {model_name}"] = {"scope_part_points": 100 * (weighted(C["specialist"], in_scope) - weighted(mixed, in_scope)), "intent_part_points": 100 * (weighted(mixed, in_scope) - weighted(C[model_name], in_scope))}
        # Costs of every hosted system on this set (prior 0.2 sessions).
        for run in ("A", "B", "C"):
            if run in llm:
                model_name = (g["primary"] if run in "AC" else g["secondary"])["name"] + (" with rows" if run == "C" else "")
                ss = sess(llm[run]["scope"], llm[run]["intent"])
                out["costs"][model_name] = cost_block(llm[run]["rows"], per_session, n, ss)
                if run in "AB":
                    out["costs"][f"{model_name}: combination"] = cost_block(llm[run]["rows"], per_session, n, block["systems"][f"{model_name}: combination"]["session_success"])
                    out["costs"][f"{model_name}: cascade_0.4"] = cost_block(llm[run]["rows"], per_session, n, block["systems"][f"{model_name}: cascade_0.4"]["session_success"], turn_share=0.4)
        spec_02 = spec_by_seed

    # Run C: the paired effect of the out-of-scope rows, for the general model and for the specialist (kNN vote rule).
    if "C" in llm and "A" in llm:
        cA, cC = tc(llm["A"]["scope"], llm["A"]["intent"]), tc(llm["C"]["scope"], llm["C"]["intent"])
        out["systems"]["prior_0.2"][f"{g['primary']['name']} with rows"] = {**_summary(turns, llm["C"]["scope"], llm["C"]["intent"], cC, in_scope), "session_success": sess(llm["C"]["scope"], llm["C"]["intent"]), "failed_requests": llm["C"]["failed"]}
        out["tests"]["rows effect: gemma with rows vs without"] = weighted_test(cC, cA, in_scope, clusters, seed)
        sym = []
        for s in SEEDS:
            f, sp = feats[s], spec_02[s]
            vote = {p: rf.knn_vote(f, p) for p in ("dev_in", "dev_out", "turns")}
            options = {"detector": {p: sp["scores"][p][sp["detector"]] >= sp["rules"][sp["detector"]][0] for p in vote}, "knn_vote": vote}
            options["either"] = {p: options["detector"][p] & vote[p] for p in vote}
            objective = {k: 0.8 * v["dev_in"].mean() + 0.2 * (~v["dev_out"]).mean() for k, v in options.items()}
            chosen = max(objective, key=objective.get)
            sym.append((tc(options[chosen]["turns"], sp["intent"]), chosen))
        c_sym = np.mean([x[0] for x in sym], axis=0)
        out["systems"]["prior_0.2"]["specialist with rows"] = {"weighted": weighted(c_sym, in_scope), "turn_correct": float(c_sym.mean()), "rules": [x[1] for x in sym]}
        out["tests"]["rows effect: specialist with rows vs without"] = weighted_test(c_sym, np.mean([tc(spec_02[s]["scope"], spec_02[s]["intent"]) for s in SEEDS], axis=0), in_scope, clusters, seed)
        out["tests"]["with rows: specialist vs gemma"] = weighted_test(c_sym, cC, in_scope, clusters, seed)

    # Run I: each frontier model on its session subset, paired with gemma and the specialist on the same turns.
    k = spec["runs"]["I"]["sessions_per_set"]
    sub_turns = [t for s in sessions[:k] for t in s]
    m = len(sub_turns)
    sub_in = in_scope[:m]
    sub_ok = lambda scope, intent: float(rf._sessions_ok(sessions[:k], scope[:m], intent[:m]).mean())
    for frontier in g["frontier_models"]:
        fpath = hosted / f"I__{name}__{frontier['name']}.jsonl"
        if not fpath.exists():
            continue
        block = out.setdefault("frontier", {"n_turns": m, "n_sessions": k, "tests": {}})
        F = _hosted_answers(fpath, sub_turns, len(labels))
        cF = rf.turn_correct(sub_turns, F["scope"], F["intent"]).astype(float)
        ss = sub_ok(F["scope"], F["intent"])
        block[frontier["name"]] = {**_summary(sub_turns, F["scope"], F["intent"], cF, sub_in), "session_success": ss, "failed_requests": F["failed"], "parse_failed": F["parse_failed"]}
        out["costs"][frontier["name"]] = cost_block(F["rows"], per_session, m, ss)
        cs = np.mean([rf.turn_correct(sub_turns, spec_02[s]["scope"][:m], spec_02[s]["intent"][:m]).astype(float) for s in SEEDS], axis=0)
        block["specialist_same_turns"] = {"weighted": weighted(cs, sub_in), "turn_correct": float(cs.mean()), "session_success": float(np.mean([sub_ok(spec_02[s]["scope"], spec_02[s]["intent"]) for s in SEEDS]))}
        block["tests"][f"{frontier['name']} vs specialist"] = weighted_test(cF, cs, sub_in, clusters[:m], seed)
        if "A" in llm:
            cG = tc(llm["A"]["scope"], llm["A"]["intent"])[:m]
            block["gemma_same_turns"] = {"weighted": weighted(cG, sub_in), "turn_correct": float(cG.mean()), "session_success": sub_ok(llm["A"]["scope"], llm["A"]["intent"])}
            block["tests"][f"{frontier['name']} vs gemma"] = weighted_test(cF, cG, sub_in, clusters[:m], seed)
            gem_sub = [llm["A"]["by_id"][t["id"]] for t in sub_turns if t["id"] in llm["A"]["by_id"]]
            out["costs"]["gemma-4-31b (frontier subset)"] = cost_block(gem_sub, per_session, m, block["gemma_same_turns"]["session_success"])

    # Original-case arm: one specialist trained and scored on un-normalised text.
    orig = rf.specialist(features(name, SEEDS[0], "original"), 0.2)
    norm = spec_02[SEEDS[0]]
    co, cn = tc(orig["scope"], orig["intent"]), tc(norm["scope"], norm["intent"])
    out["systems"]["prior_0.2"]["specialist original case (seed 1)"] = _summary(turns, orig["scope"], orig["intent"], co, in_scope)
    out["tests"]["original case vs normalised (seed 1)"] = weighted_test(co, cn, in_scope, clusters, seed)
    return out


def analyze() -> None:
    from bakeoff import revision_free as rf

    spec = rules()
    data = load()
    result = {"protocol_id": spec["protocol_id"], "sets": {}}
    for name in SETS:
        result["sets"][name] = analyze_set(name, data[name], spec)
        print(f"analyzed {name}", flush=True)
    primary = {}
    gemma = spec["general_models"]["primary"]["name"]
    for name in SETS:
        for key in (f"specialist vs {gemma}", f"combination vs {gemma}"):
            if key in result["sets"][name]["tests"]:
                primary[f"{name} | {key}"] = result["sets"][name]["tests"][key]["p"]
    for label, value in rf.holm(primary).items():
        name, key = label.split(" | ")
        result["sets"][name]["tests"][key]["holm_p"] = value
    eq = result["sets"]["banking77"]["tests"].get(f"specialist vs {gemma}")
    if eq:
        eq["equivalent_within_2.5"] = bool(-2.5 < eq["ci90"][0] and eq["ci90"][1] < 2.5)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8")
    for name, r in result["sets"].items():
        print(f"\n{name}: {r['n_turns']} turns ({r['n_out']} out of scope), {r['n_sessions']} sessions, length AUROC {r['length_auroc']:.3f}")
        for key, row in r["systems"]["prior_0.2"].items():
            print(f"  {key:52s} weighted {100 * row['weighted']:.2f} turn {100 * row['turn_correct']:.2f}" + (f" kept {100 * row['in_scope_kept']:.1f} caught {100 * row['out_of_scope_caught']:.1f} intent {100 * row['intent_on_in_scope']:.1f}" if "in_scope_kept" in row else "") + (f" session {100 * row['session_success']:.1f}" if "session_success" in row else ""))
        for key, t in r["tests"].items():
            if "p" in t:
                print(f"  TEST {key:50s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}" + (f" holm={t['holm_p']:.4f}" if "holm_p" in t else "") + (f" equivalent={t['equivalent_within_2.5']} (90% [{t['ci90'][0]:+.2f}, {t['ci90'][1]:+.2f}])" if "equivalent_within_2.5" in t else ""))
            else:
                print(f"  {key:55s} " + " ".join(f"{k} {v:+.2f}" for k, v in t.items()))
        for key, c in r["costs"].items():
            if c:
                print(f"  COST {key:45s} in {c['prompt_tokens_mean']:.0f} out {c['completion_tokens_mean']:.1f} ${c['usd_per_1000_sessions']:.3f}/1000 sessions p50 {c['latency_ms_p50']:.0f}ms p95 {c['latency_ms_p95']:.0f}ms")
        if "frontier" in r:
            fr = r["frontier"]
            print(f"  FRONTIER subset {fr['n_turns']} turns: " + json.dumps({k: (round(100 * v["weighted"], 2) if isinstance(v, dict) and "weighted" in v else None) for k, v in fr.items() if isinstance(v, dict) and "weighted" in v}))
            for key, t in fr["tests"].items():
                print(f"  TEST {key:50s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}")
    print("CONFIRMATORY_ANALYZE_DONE", flush=True)


def figure() -> None:
    """Weighted accuracy against USD per 1,000 sessions (log axis), one panel per set, with the cost-accuracy frontier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bakeoff.paper_figures import GEMMA, GRID, INK, MUTED, QWEN, SPECIALIST, SURFACE, _style

    result = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    floor = 0.01  # where the specialist (0 API dollars) is drawn on the log axis
    # Identity: colour = model family, marker shape = system design (never colour alone; legend below).
    style = {
        "specialist": ("Specialist (Ettin 150M, $0 API)", SPECIALIST, "o"),
        "gemma-4-31b": ("Gemma 4 31B, retrieval", GEMMA, "D"),
        "gemma-4-31b: combination": ("Gemma + specialist intent (combination)", GEMMA, "P"),
        "gemma-4-31b: cascade_0.4": ("Gemma cascade, 40% of turns", GEMMA, "v"),
        "gemma-4-31b with rows": ("Gemma, pool with out-of-scope rows", GEMMA, "X"),
        "qwen3.5-27b": ("Qwen3.5 27B, retrieval", QWEN, "^"),
        "qwen3.5-27b: combination": ("Qwen + specialist intent (combination)", QWEN, "P"),
        "qwen3.5-27b: cascade_0.4": ("Qwen cascade, 40% of turns", QWEN, "v"),
        "kimi-k3": ("Kimi K3 (2.8T), retrieval", INK, "s"),
        "claude-opus-5": ("Claude Opus 5, retrieval", INK, "*"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), facecolor=SURFACE)
    titles = {"banking77": "Banking77", "massive": "MASSIVE (42 skills)", "clinc150": "CLINC150"}
    seen = {}

    def draw(ax, points) -> None:
        ordered = sorted((p for p in points if p[1] > floor), key=lambda p: (p[1], -p[2]))  # the $0 specialist is a placeholder, not on the frontier
        best, frontier = -1.0, []
        for p in ordered:
            if p[2] > best:
                frontier.append(p)
                best = p[2]
        ax.plot([p[1] for p in frontier], [100 * p[2] for p in frontier], color=MUTED, linewidth=1.2, linestyle=(0, (3, 2)), zorder=1)
        for key, x, y in points:
            label, colour, marker = style[key]
            seen[key] = ax.scatter([x], [100 * y], s=150 if marker == "*" else 58, marker=marker, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
        ax.set_xscale("log")
        ax.set_xlim(floor / 2, max(p[1] for p in points) * 3)
        ax.axvspan(floor / 2, floor * 1.8, color=GRID, alpha=0.5, zorder=0)
        ax.annotate("$0 API\n(off scale)", (floor, ax.get_ylim()[0]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
        _style(ax)
        ax.grid(color=GRID, linewidth=0.7)

    for col, name in enumerate(SETS):
        r = result["sets"][name]
        systems, costs = r["systems"]["prior_0.2"], r["costs"]
        top = [("specialist", floor, systems["specialist"]["weighted"])]
        top += [(key, costs[key]["usd_per_1000_sessions"], systems[key]["weighted"]) for key in style if key != "specialist" and key in systems and costs.get(key)]
        draw(axes[0, col], top)
        axes[0, col].set_title(f"{titles[name]}: all systems, {r['n_sessions']} sessions", color=INK, fontsize=9.5, loc="left")
        fr = r.get("frontier", {})
        sub = [("specialist", floor, fr["specialist_same_turns"]["weighted"]), ("gemma-4-31b", costs["gemma-4-31b (frontier subset)"]["usd_per_1000_sessions"], fr["gemma_same_turns"]["weighted"])]
        uncached = json.loads((ROOT / "artifacts" / "posthoc" / "round3b.json").read_text(encoding="utf-8"))["cost"]["sets"][name]
        sub += [(key, uncached[key]["usd_per_1000_sessions_uncached_list"], fr[key]["weighted"]) for key in ("kimi-k3", "claude-opus-5") if key in fr]
        draw(axes[1, col], sub)
        axes[1, col].set_title(f"{titles[name]}: frontier models, first {fr['n_sessions']} sessions", color=INK, fontsize=9.5, loc="left")
        axes[1, col].set_xlabel("API USD per 1,000 sessions, uncached list price (log)", color=MUTED, fontsize=8)
    for row in range(2):
        axes[row, 0].set_ylabel("weighted turn accuracy (%)\n0.8 in-scope + 0.2 out-of-scope", color=MUTED, fontsize=8.5)
    fig.suptitle("Accuracy against API cost on the confirmatory sessions (dashed line: frontier of systems with an API cost)", color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.legend([seen[k] for k in style if k in seen], [style[k][0] for k in style if k in seen], loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=False, fontsize=8, labelcolor=INK, handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "docs" / "figures" / f"f5_cost_frontier.{suffix}", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.05)
    print("FIGURE_DONE docs/figures/f5_cost_frontier.png", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "hosted":
        for run_name in sys.argv[2].split(","):
            hosted(run_name)
    else:
        {"build": build, "train": train, "features": stage_features, "check": check, "analyze": analyze, "figure": figure}[stage]()

"""Jev re-run under the paper's two pre-registered designs. Rules: jev_rerun_protocol.json.

Stages: 'check' (dev items), 'J1' (retrieval suppression items), 'J2' (confirmatory sessions), 'analyze' (processor).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "jev_rerun_protocol.json"
OUT = ROOT / "artifacts" / "jev-rerun"
NEWLINE = "\n"
J1_SPLITS = ["banking77_oos", "clinc_banking", "clinc_credit_cards"]
J1_ARMS = ["zero_shot", "retrieval", "instruction"]
J2_SETS = ["banking77", "massive", "clinc150"]
J2_ARMS = ["zero_shot", "retrieval"]


def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _questions(spec: dict, labels: list[str], arm: str) -> dict:
    q = spec["request"]["questions"]
    scope = q["scope"]["instructions"].format(labels=", ".join(labels))
    if arm == "instruction":
        scope += " " + spec["request"]["instruction_arm"].split("appended to the scope question's instructions: ")[1].strip("'")
    return {
        "scope": {"type": "noul", "instructions": scope},
        "intent": {"type": "choice", "instructions": q["intent"]["instructions"], "criteria": {label: None for label in labels}},
    }


def _states(pool: list[tuple[str, int]], labels: list[str], texts: list[tuple[str, str]], arm: str) -> list[tuple[str, str]]:
    from bakeoff.workflow_retrieval import BM25

    if arm == "zero_shot":
        return list(texts)
    index = BM25([t for t, _ in pool])
    out = []
    for key, text in texts:
        lines = [f"{pool[i][0]} => {labels[pool[i][1]]}" for i in index.top(text, 20)[::-1]]
        out.append((key, "Similar labelled examples (message => request type):" + NEWLINE + NEWLINE.join(lines) + NEWLINE + NEWLINE + "Current message: " + text))
    return out


def _collect(client, spec: dict, labels: list[str], questions: dict, states: list[tuple[str, str]], path) -> dict:
    from bakeoff import workflow as v1

    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    for _ in range(3):
        done = _read(path)
        todo = [s for s in states if s[0] not in done]
        if not todo:
            break

        def work(item):
            key, state = item
            try:
                payload = v1._jev_post(client, {"model": spec["model"]["model"], "state": state, "questions": questions})
                answers = payload["answers"]
                row = {
                    "id": key,
                    "error": None,
                    "scope": float(answers["scope"]["noul"]),
                    "intent_probs": [float(answers["intent"]["probabilities"][label]) for label in labels],
                    "model": payload.get("model"),
                    "input_tokens": (payload.get("usage") or {}).get("input_tokens", 0),
                    "usage": payload.get("usage"),
                }
                if row["model"] != spec["model"]["model"]:
                    row["error"] = f"model {row['model']} is not the pinned version"
            except Exception as error:  # noqa: BLE001 - every failure is recorded and retried on the next pass
                row = {"id": key, "error": str(error)[:300]}
            return row

        with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=spec["request"]["concurrency"]) as pool:
            for count, row in enumerate(pool.map(work, todo), start=1):
                handle.write(json.dumps(row) + NEWLINE)
                if count % 500 == 0:
                    handle.flush()
                    print(f"    {path.stem} {count}/{len(todo)}", flush=True)
    done = _read(path)
    return done


def _read(path) -> dict:
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("error") is None:
                rows[row["id"]] = row
    return rows


def _client():
    import httpx

    from bakeoff import jev

    key = jev._api_key(json.loads(jev.JEV_PROTOCOL_PATH.read_text()))
    return httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=90.0)


def _j1_blocks() -> dict:
    from bakeoff import retrieval_suppression as rs

    return rs.load()


def _j2_blocks() -> dict:
    from bakeoff import confirmatory as cf

    data = cf.load()
    return {name: {"labels": data[name]["labels"], "pool": [(r["text"], r["intent"]) for r in data[name]["train"]], "turns": cf.turns_of(data[name]), "dev_in": data[name]["dev_in"], "dev_out": data[name]["dev_out"]} for name in J2_SETS}


def run(stage: str) -> None:
    spec = rules()
    with _client() as client:
        if stage in ("check", "J1"):
            data = _j1_blocks()
            for split in J1_SPLITS:
                block = data[split]
                labels, pool = block["labels"], [tuple(p) for p in block["pool"]]
                if stage == "check":
                    texts = [(f"dev-in-{k}", t) for k, (t, _) in enumerate(block["dev_in"][:5])] + [(f"dev-out-{k}", t) for k, t in enumerate(block["dev_out"][:5])]
                else:
                    texts = [(i["id"], i["text"]) for i in block["items"]]
                for arm in J1_ARMS:
                    path = OUT / ("check" if stage == "check" else "J1") / f"{split}__{arm}.jsonl"
                    rows = _collect(client, spec, labels, _questions(spec, labels, arm), _states(pool, labels, texts, arm), path)
                    tokens = [r["input_tokens"] for r in rows.values()]
                    print(f"{stage} J1 {split:18s} {arm:12s} {len(rows)}/{len(texts)} models {sorted({r['model'] for r in rows.values()})} input tokens mean {np.mean(tokens) if tokens else 0:.0f}", flush=True)
        if stage in ("check", "J2"):
            data = _j2_blocks()
            for name in J2_SETS:
                block = data[name]
                if stage == "check":
                    texts = [(f"dev-in-{k}", r["text"]) for k, r in enumerate(block["dev_in"][:5])] + [(f"dev-out-{k}", r["text"]) for k, r in enumerate(block["dev_out"][:5])]
                else:
                    texts = [(t["id"], t["text"]) for t in block["turns"]]
                for arm in J2_ARMS:
                    path = OUT / ("check" if stage == "check" else "J2") / f"{name}__{arm}.jsonl"
                    rows = _collect(client, spec, block["labels"], _questions(spec, block["labels"], arm), _states(block["pool"], block["labels"], texts, arm), path)
                    tokens = [r["input_tokens"] for r in rows.values()]
                    print(f"{stage} J2 {name:18s} {arm:12s} {len(rows)}/{len(texts)} models {sorted({r['model'] for r in rows.values()})} input tokens mean {np.mean(tokens) if tokens else 0:.0f}", flush=True)
    print(f"JEV_{stage}_DONE", flush=True)


# ---------------------------------------------------------------- analysis


def _decisions(path, keys: list[str], in_scope: np.ndarray) -> dict:
    rows = _read(path)
    prob = np.asarray([rows[k]["scope"] if k in rows else np.nan for k in keys])
    failed = np.isnan(prob)
    scope = np.where(failed, ~in_scope, np.nan_to_num(prob) >= 0.5)
    intent = np.asarray([int(np.argmax(rows[k]["intent_probs"])) if k in rows else -1 for k in keys])
    return {"prob": prob, "scope": scope.astype(bool), "intent": intent, "failed": int(failed.sum()), "rows": [rows[k] for k in keys if k in rows]}


def _auroc_boot(prob: np.ndarray, in_scope: np.ndarray, clusters: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, np.ndarray]:
    from bakeoff.confirmatory import _auroc

    rng = np.random.default_rng(seed)
    members = {k: np.where(clusters == k)[0] for k in np.unique(clusters)}
    keys_in, keys_out = np.unique(clusters[in_scope]), np.unique(clusters[~in_scope])
    boots = []
    for _ in range(draws):
        idx = np.concatenate([members[k] for k in rng.choice(keys_in, len(keys_in))] + [members[k] for k in rng.choice(keys_out, len(keys_out))])
        boots.append(idx)
    return _auroc(prob, in_scope), boots


def analyze() -> None:
    from bakeoff import confirmatory as cf
    from bakeoff import retrieval_suppression as rs
    from bakeoff import revision_free as rf
    from bakeoff.confirmatory import _auroc

    spec = rules()
    seed = 20261016
    result = {"protocol_id": spec["protocol_id"], "J1": {}, "J2": {}}
    data = _j1_blocks()
    groups = {"banking77_oos": ["banking77_oos"], "clinc_single_domain": ["clinc_banking", "clinc_credit_cards"]}
    ps = {}
    for group, splits in groups.items():
        items = [i for s in splits for i in data[s]["items"]]
        keys = [i["id"] for i in items]
        in_scope = np.asarray([i["in_scope"] for i in items])
        gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
        clusters = np.asarray([i["cluster"] for i in items])
        arms = {}
        for arm in J1_ARMS:
            parts = [_decisions(OUT / "J1" / f"{s}__{arm}.jsonl", [i["id"] for i in data[s]["items"]], np.asarray([i["in_scope"] for i in data[s]["items"]])) for s in splits]
            arms[arm] = {k: np.concatenate([p[k] for p in parts]) for k in ("prob", "scope", "intent")}
            arms[arm]["failed"] = sum(p["failed"] for p in parts)
            arms[arm]["rows"] = [r for p in parts for r in p["rows"]]
        block = {"arms": {}, "tests": {}}
        for arm, a in arms.items():
            prob = np.nan_to_num(a["prob"], nan=0.5)
            block["arms"][arm] = {
                "held_out_rejected": float((~a["scope"])[~in_scope].mean()),
                "in_scope_right": float((a["scope"] & (a["intent"] == gold))[in_scope].mean()),
                "in_scope_kept": float(a["scope"][in_scope].mean()),
                "auroc_scope_prob": _auroc(prob, in_scope),
                "mean_scope_prob_in_scope": float(prob[in_scope].mean()),
                "mean_scope_prob_held_out": float(prob[~in_scope].mean()),
                "failed": int(a["failed"]),
                "input_tokens_mean": float(np.mean([r["input_tokens"] for r in a["rows"]])) if a["rows"] else None,
            }
        rej = {arm: (~arms[arm]["scope"])[~in_scope].astype(float) for arm in J1_ARMS}
        t = rs.flip_test(rej["zero_shot"] - rej["retrieval"], clusters[~in_scope], seed, True)
        block["tests"]["HJ1 zero-shot minus retrieval rejection"] = t
        ps[group] = t["p"]
        # HJ2: AUROC change and mean-probability change with cluster bootstrap.
        pz, pr = np.nan_to_num(arms["zero_shot"]["prob"], nan=0.5), np.nan_to_num(arms["retrieval"]["prob"], nan=0.5)
        _, boots = _auroc_boot(pz, in_scope, clusters, seed)
        d_auc = np.asarray([_auroc(pr[i], in_scope[i]) - _auroc(pz[i], in_scope[i]) for i in boots])
        d_held = np.asarray([pr[i][~in_scope[i]].mean() - pz[i][~in_scope[i]].mean() for i in boots])
        d_in = np.asarray([pr[i][in_scope[i]].mean() - pz[i][in_scope[i]].mean() for i in boots])
        block["tests"]["HJ2 retrieval minus zero-shot"] = {
            "auroc_change": _auroc(pr, in_scope) - _auroc(pz, in_scope),
            "auroc_change_ci95": np.percentile(d_auc, [2.5, 97.5]).tolist(),
            "mean_prob_change_held_out": float(pr[~in_scope].mean() - pz[~in_scope].mean()),
            "mean_prob_change_held_out_ci95": np.percentile(d_held, [2.5, 97.5]).tolist(),
            "mean_prob_change_in_scope": float(pr[in_scope].mean() - pz[in_scope].mean()),
            "mean_prob_change_in_scope_ci95": np.percentile(d_in, [2.5, 97.5]).tolist(),
        }
        hj2 = block["tests"]["HJ2 retrieval minus zero-shot"]
        hj2["readout"] = "operating-point shift" if (hj2["mean_prob_change_held_out"] > 0 and hj2["auroc_change_ci95"][0] <= 0 <= hj2["auroc_change_ci95"][1]) else ("discrimination change" if not (hj2["auroc_change_ci95"][0] <= 0 <= hj2["auroc_change_ci95"][1]) else "neither")
        right = {arm: (arms[arm]["scope"] & (arms[arm]["intent"] == gold))[in_scope].astype(float) for arm in J1_ARMS}
        block["tests"]["secondary: instruction minus retrieval rejection"] = rs.flip_test(rej["instruction"] - rej["retrieval"], clusters[~in_scope], seed, True)
        block["tests"]["secondary: instruction minus retrieval in-scope right"] = rs.flip_test(right["instruction"] - right["retrieval"], clusters[in_scope], seed, False)
        result["J1"][group] = block
    for group, value in rf.holm(ps).items():
        result["J1"][group]["tests"]["HJ1 zero-shot minus retrieval rejection"]["holm_p"] = value

    # J2: confirmatory sessions.
    conf = cf.load()
    gemma = cf.rules()["general_models"]["primary"]["name"]
    for name in J2_SETS:
        d = conf[name]
        turns = cf.turns_of(d)
        keys = [t["id"] for t in turns]
        in_scope = np.asarray([t["in_scope"] for t in turns])
        clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
        tc = lambda scope, intent: rf.turn_correct(turns, scope, intent).astype(float)
        spec_by_seed = {s: rf.specialist(cf.features(name, s), 0.2) for s in cf.SEEDS}
        c_spec = np.mean([tc(spec_by_seed[s]["scope"], spec_by_seed[s]["intent"]) for s in cf.SEEDS], axis=0)
        llm = cf._hosted_answers(cf.OUT / "hosted" / f"A__{name}__{gemma}.jsonl", turns, len(d["labels"]))
        c_gemma = tc(llm["scope"], llm["intent"])
        block = {"systems": {"specialist": {"weighted": cf.weighted(c_spec, in_scope)}, gemma: {"weighted": cf.weighted(c_gemma, in_scope)}}, "tests": {}}
        for arm in J2_ARMS:
            a = _decisions(OUT / "J2" / f"{name}__{arm}.jsonl", keys, in_scope)
            c_alone = tc(a["scope"], a["intent"])
            c_gate = np.mean([tc(a["scope"], spec_by_seed[s]["intent"]) for s in cf.SEEDS], axis=0)
            prob = np.nan_to_num(a["prob"], nan=0.5)
            block["systems"][f"jev {arm}"] = {**cf._summary(turns, a["scope"], a["intent"], c_alone, in_scope), "auroc_scope_prob": _auroc(prob, in_scope), "failed": a["failed"], "input_tokens_mean": float(np.mean([r["input_tokens"] for r in a["rows"]])) if a["rows"] else None, "latency_note": "Jev latency is not recorded by this client"}
            block["systems"][f"jev gate {arm} + specialist intent"] = {"weighted": cf.weighted(c_gate, in_scope)}
            if arm == "retrieval":
                block["tests"]["secondary: jev alone (retrieval) vs specialist"] = cf.weighted_test(c_alone, c_spec, in_scope, clusters, seed)
                block["tests"][f"secondary: jev alone (retrieval) vs {gemma}"] = cf.weighted_test(c_alone, c_gemma, in_scope, clusters, seed)
                block["tests"]["secondary: jev gate (retrieval) vs specialist"] = cf.weighted_test(c_gate, c_spec, in_scope, clusters, seed)
        result["J2"][name] = block
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + NEWLINE, encoding="utf-8")
    for group, block in result["J1"].items():
        print(f"\nJ1 {group}")
        for arm, a in block["arms"].items():
            print(f"  {arm:12s} rejected {100 * a['held_out_rejected']:.1f} in-scope right {100 * a['in_scope_right']:.1f} kept {100 * a['in_scope_kept']:.1f} AUROC {a['auroc_scope_prob']:.3f} mean p(in) in-scope {a['mean_scope_prob_in_scope']:.3f} held-out {a['mean_scope_prob_held_out']:.3f} failed {a['failed']} tokens {a['input_tokens_mean']}")
        for key, t in block["tests"].items():
            print(f"  {key}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}))
    for name, block in result["J2"].items():
        print(f"\nJ2 {name}")
        for key, s in block["systems"].items():
            print(f"  {key:40s} " + " ".join(f"{k} {100 * v:.1f}" for k, v in s.items() if isinstance(v, float) and k != "input_tokens_mean"))
        for key, t in block["tests"].items():
            print(f"  {key:52s} {t['difference_points']:+.2f} [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}] p={t['p']:.4f}")
    print("JEV_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    analyze() if stage == "analyze" else run(stage)

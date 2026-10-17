"""POST HOC analyses from the round 5 internal review. Nothing here was pre-registered.

CPU only, from saved replies, cached specialist features and frozen item lists. Writes artifacts/posthoc/round5.json:
- massive_inscope_only_threshold: the confirmatory specialist with each detector's threshold at the 5th or 3rd percentile of
  development in-scope scores (no out-of-scope data), against the pre-registered rule; mean of three seeds.
- run_d_specialist: the run D specialist reference on BANKING77-OOS, as run and on the leak-free subset, per seed.
- jev_matched_point: Jev's D2 rejection with each arm's threshold set so that a target share of in-scope items is kept.
- sibling_baseline: for D2 BANKING77-OOS, the share of accepted held-out requests whose chosen label is among the 20 BM25
  labels, in the zero-shot, random and retrieval arms.
- run_d_overlap: run D items whose normalised text was a confirmatory turn.
"""

import json
import re

import numpy as np

from bakeoff import confirmatory as cf
from bakeoff import retrieval_mechanism as rm
from bakeoff import retrieval_suppression as rs
from bakeoff import revision_free as rf
from bakeoff import workflow_v4 as v4
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.robustness_seeds import _detector_scores
from bakeoff.temperature import fit_temperature
from bakeoff.workflow_retrieval import BM25

OUT = ROOT / "artifacts" / "posthoc" / "round5.json"


def inscope_only_threshold() -> dict:
    out = {}
    for name in ("banking77", "massive", "clinc150"):
        d = cf.load()[name]
        turns = cf.turns_of(d)
        y = np.array([t["in_scope"] for t in turns])
        res = {}
        for s in cf.SEEDS:
            f = cf.features(name, s)
            T = fit_temperature(f["dev_in_logits"], f["dev_gold"])["temperature"]
            sc = _detector_scores(f, T)
            intent = apply_temperature(f["turns_logits"], T).argmax(1)

            def summ(scope):
                c = rf.turn_correct(turns, scope, intent).astype(float)
                return [cf.weighted(c, y) * 100, scope[y].mean() * 100, (~scope[~y]).mean() * 100]

            res.setdefault("pre-registered rule", []).append(summ(rf.specialist(f, 0.2)["scope"]))
            for det in sc["turns"]:
                for q in (0.05, 0.03):
                    t = np.quantile(sc["dev_in"][det], q)
                    res.setdefault(f"{det}, in-scope-only q={q}", []).append(summ(sc["turns"][det] >= t))
        out[name] = {k: dict(zip(("weighted", "kept", "rejected"), np.array(v).mean(0).tolist())) for k, v in res.items()}
    return out


def run_d_specialist() -> dict:
    audit = json.loads((ROOT / "artifacts/posthoc/b77oos_label_audit.json").read_text(encoding="utf-8"))
    items = rs.load()["banking77_oos"]["items"]
    y = np.array([i["in_scope"] for i in items])
    gold = np.array([i["intent"] if i["in_scope"] else -1 for i in items])
    leaking = set(audit["held_out_intents_leaking"])
    never = np.array([i["in_scope"] or i["cluster"][5:] not in leaking for i in items])
    clean = np.array([not (audit["items"][i["id"]].get("wrong_gold") or audit["items"][i["id"]].get("true_is_in_scope")) for i in items])
    sub = never & clean
    seeds = {}
    for s in rs.SEEDS:
        sp = rf.specialist(rs.features("banking77_oos", s), 0.2)
        sc, right = sp["scope"], sp["scope"] & (sp["intent"] == gold)
        seeds[str(s)] = {"detector": sp["detector"],
                         "as_run_rejected": float((~sc[~y]).mean() * 100), "as_run_inscope_right": float(right[y].mean() * 100),
                         "leak_free_rejected": float((~sc[~y & sub]).mean() * 100), "leak_free_inscope_right": float(right[y & sub].mean() * 100)}
    mean = {k: float(np.mean([v[k] for v in seeds.values()])) for k in next(iter(seeds.values())) if k != "detector"}
    return {"seeds": seeds, "mean": mean}


def jev_matched_point() -> dict:
    d = rm.load()
    out = {}
    for split, arms in [("banking77_oos", ["zero_shot", "retrieval", "random_examples", "rule_after_examples"]),
                        ("clinc_banking", ["zero_shot_10", "retrieval_10", "zero_shot_20", "retrieval_20"]),
                        ("clinc_credit_cards", ["zero_shot_10", "retrieval_10", "zero_shot_20", "retrieval_20"])]:
        items = d[split]["items"]
        ids = [i["id"] for i in items]
        y = np.array([i["in_scope"] for i in items])
        res = {}
        for a in arms:
            rows = {}
            for line in (ROOT / f"artifacts/retrieval-mechanism/jev/{split}__{a}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                rows[r["id"]] = r
            p = np.array([rows[i]["scope"] for i in ids])
            r = {"rejected_at_0.5": float(np.mean(p[~y] < 0.5) * 100), "kept_at_0.5": float(np.mean(p[y] >= 0.5) * 100)}
            for target in (0.90, 0.95):
                t = next(t for t in sorted(np.unique(p), reverse=True) if np.mean(p[y] >= t) >= target)
                r[f"kept_{int(target * 100)}"] = {"threshold": float(t), "kept": float(np.mean(p[y] >= t) * 100),
                                                    "rejected": float(np.mean(p[~y] < t) * 100)}
            res[a] = r
        out[split] = res
    return out


def sibling_baseline() -> dict:
    data = rm.load()["banking77_oos"]
    items, labels, pool = data["items"], data["labels"], data["pool"]
    index = BM25([t for t, _ in pool])
    neigh = {i["id"]: [pool[j][1] for j in index.top(i["text"], 20)] for i in items}
    held = [i for i in items if not i["in_scope"]]
    out = {}
    for model in ("gemma-4-31b", "qwen3.5-27b", "llama-3.3-70b", "deepseek-v4-pro"):
        r = {}
        for arm in ("zero_shot", "random_examples", "retrieval"):
            rows = v4._read(rm._path(model, "banking77_oos", arm))
            p = {k: v4.parse(x["text"], len(labels)) for k, x in rows.items()}
            acc = [i for i in held if p[i["id"]]["scope"]]
            r[arm] = {"accepted": len(acc),
                      "share_in_retrieved_labels": float(np.mean([p[i["id"]]["intent"] in neigh[i["id"]] for i in acc]) * 100)}
        out[model] = r
    return out


def run_d_overlap() -> dict:
    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t.lower())).strip()

    seen = set()

    def collect(o):
        if isinstance(o, dict):
            for k in ("text", "original_text", "normalised"):
                if isinstance(o.get(k), str):
                    seen.add(norm(o[k]))
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)

    conf = json.loads((ROOT / "artifacts/confirmatory/items.json").read_text(encoding="utf-8"))
    for block in conf.values():
        collect(block.get("items", block.get("sessions", [])) if isinstance(block, dict) else block)
    run_d = rs.load()
    out = {}
    for split in ("banking77_oos", "clinc_banking", "clinc_credit_cards"):
        items = run_d[split]["items"]
        out[split] = {"items": len(items), "were_confirmatory_turns": sum(norm(i["text"]) in seen for i in items)}
    return out


def main() -> None:
    result = {"label": "POST HOC (round 5 internal review); not pre-registered",
              "massive_inscope_only_threshold": inscope_only_threshold(),
              "run_d_specialist": run_d_specialist(),
              "jev_matched_point": jev_matched_point(),
              "sibling_baseline": sibling_baseline(),
              "run_d_overlap": run_d_overlap()}
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

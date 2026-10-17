"""POST HOC: analyses requested by two external reviews of draft 3 (2026-10-17). None was pre-registered; no API calls.

    python -m bakeoff.posthoc_round6

On the confirmatory sessions, with the saved replies and the cached specialist features:
- per_seed: each specialist seed's weighted accuracy and its paired difference from Gemma (prior-0.2 threshold rule);
- endpoint_prior: weighted accuracy A(pi) = (1 - pi) C + pi R for out-of-scope weights pi other than the pre-registered 0.2
  (C: share of in-scope turns kept with the right intent; R: share of out-of-scope turns rejected), with the specialist - Gemma
  difference and its sign-flip p under each weight (same clusters as the primary test, unadjusted);
- cascade_cost: the cascade's actual API cost, summing Gemma's billed cost over the turns each seed's cascade escalates,
  against the 0.4 x mean-cost figure printed in draft 3;
- clusters: the number of in-scope intent clusters and out-of-scope turns per set.
The pre-registered threshold-rule sensitivity (out-of-scope rates 0.1 and 0.3) is already in artifacts/confirmatory/analysis.json
and is only read here. Writes artifacts/posthoc/round6.json.
"""

from __future__ import annotations

import json

import numpy as np

from bakeoff import confirmatory as cf
from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "posthoc"
PRIORS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def weighted_test_at(a, b, in_scope, clusters, seed: int, pi: float, draws: int = 10000) -> dict:
    """cf.weighted_test with out-of-scope weight pi instead of 0.2 (sign flips of cluster sums)."""
    n_in, n_out = in_scope.sum(), (~in_scope).sum()
    weight = np.where(in_scope, (1 - pi) / n_in, pi / n_out)
    diff = (a - b).astype(float)
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=weight * diff)
    observed = float(sums.sum())
    flips = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, len(sums)))
    p = (1 + np.sum(np.abs(flips @ sums) >= abs(observed) - 1e-12)) / (draws + 1)
    return {"difference_points": 100 * observed, "p": float(p)}


def analyze_set(name: str, d: dict, spec: dict) -> dict:
    from bakeoff import revision_free as rf

    seed = spec["seed"]
    turns = cf.turns_of(d)
    n = len(turns)
    in_scope = np.asarray([t["in_scope"] for t in turns])
    clusters = np.asarray([f"i{t['intent']}" if t["in_scope"] else f"o{k}" for k, t in enumerate(turns)])
    tc = lambda scope, intent: rf.turn_correct(turns, scope, intent).astype(float)
    per_session = n / len(d["sessions"])
    g = spec["general_models"]
    gemma = cf._hosted_answers(cf.OUT / "hosted" / f"A__{name}__{g['primary']['name']}.jsonl", turns, len(d["labels"]))
    qwen = cf._hosted_answers(cf.OUT / "hosted" / f"B__{name}__{g['secondary']['name']}.jsonl", turns, len(d["labels"]))
    c_gemma, c_qwen = tc(gemma["scope"], gemma["intent"]), tc(qwen["scope"], qwen["intent"])
    sp = {s: rf.specialist(cf.features(name, s), 0.2) for s in cf.SEEDS}
    c_seed = {s: tc(sp[s]["scope"], sp[s]["intent"]) for s in cf.SEEDS}
    c_spec = np.mean([c_seed[s] for s in cf.SEEDS], axis=0)
    out = {"clusters": {"in_scope_intent_clusters": int(len({t["intent"] for t in turns if t["in_scope"]})), "out_of_scope_turns": int((~in_scope).sum())}}

    out["per_seed"] = {
        str(s): {"specialist_weighted": 100 * cf.weighted(c_seed[s], in_scope), "minus_gemma": cf.weighted_test(c_seed[s], c_gemma, in_scope, clusters, seed)}
        for s in cf.SEEDS
    }
    out["per_seed"]["mean_of_seeds"] = {"specialist_weighted": 100 * cf.weighted(c_spec, in_scope), "minus_gemma": cf.weighted_test(c_spec, c_gemma, in_scope, clusters, seed)}

    rows = {}
    for label, c in (("specialist", c_spec), ("gemma-4-31b", c_gemma), ("qwen3.5-27b", c_qwen)):
        rows[label] = {"C": 100 * float(c[in_scope].mean()), "R": 100 * float(c[~in_scope].mean())}
    out["endpoint_prior"] = {
        "components": rows,
        "by_pi": {
            str(pi): {
                **{k: (1 - pi) * v["C"] + pi * v["R"] for k, v in rows.items()},
                "specialist_minus_gemma": weighted_test_at(c_spec, c_gemma, in_scope, clusters, seed, pi),
            }
            for pi in PRIORS
        },
    }

    cost = {t["id"]: (gemma["by_id"][t["id"]].get("cost") or 0.0) if t["id"] in gemma["by_id"] else 0.0 for t in turns}
    full = np.asarray([cost[t["id"]] for t in turns])
    routed = []
    for s in cf.SEEDS:
        escalate = np.zeros(n, bool)
        escalate[np.argsort(sp[s]["top"], kind="stable")[: int(round(0.4 * n))]] = True
        routed.append(float(full[escalate].sum()))
    out["cascade_cost"] = {
        "turn_share_escalated": int(round(0.4 * n)) / n,
        "gemma_usd_per_1000_sessions": 1000 * float(full.mean()) * per_session,
        "cascade_usd_per_1000_sessions_actual": 1000 * float(np.mean(routed)) / n * per_session,
        "cascade_usd_per_1000_sessions_draft3": 1000 * float(full.mean()) * 0.4 * per_session,
        "actual_share_of_gemma_cost": float(np.mean(routed) / full.sum()),
        "per_seed_share": [r / float(full.sum()) for r in routed],
    }
    return out


def main() -> None:
    spec = cf.rules()
    data = cf.load()
    analysis = json.loads((cf.OUT / "analysis.json").read_text(encoding="utf-8"))
    result = {"status": "POST HOC: requested by external reviews of draft 3 (2026-10-17); not pre-registered", "sets": {}}
    for name in cf.SETS:
        block = analyze_set(name, data[name], spec)
        tests = analysis["sets"][name]["tests"]
        systems = analysis["sets"][name]["systems"]
        block["threshold_rule_sensitivity_preregistered"] = {
            rule: {
                "specialist_weighted": 100 * systems[rule]["specialist"]["weighted"],
                "gemma_weighted": 100 * systems[rule]["gemma-4-31b"]["weighted"],
                "minus_gemma": tests["specialist vs gemma-4-31b"] if rule == "prior_0.2" else tests[f"{rule} | specialist vs gemma-4-31b"],
            }
            for rule in ("prior_0.1", "prior_0.2", "prior_0.3")
        }
        result["sets"][name] = block
        print(f"done {name}", flush=True)
    (OUT / "round6.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    for name, b in result["sets"].items():
        print(f"\n{name} clusters {b['clusters']}")
        for s, v in b["per_seed"].items():
            print(f"  seed {s}: specialist {v['specialist_weighted']:.2f}  minus gemma {v['minus_gemma']['difference_points']:+.2f} p={v['minus_gemma']['p']:.4f}")
        for pi, v in b["endpoint_prior"]["by_pi"].items():
            print(f"  pi={pi}: spec {v['specialist']:.1f} gemma {v['gemma-4-31b']:.1f} qwen {v['qwen3.5-27b']:.1f} diff {v['specialist_minus_gemma']['difference_points']:+.2f} p={v['specialist_minus_gemma']['p']:.4f}")
        c = b["cascade_cost"]
        print(f"  cascade actual share {c['actual_share_of_gemma_cost']:.3f} ({[round(x, 3) for x in c['per_seed_share']]}), ${c['cascade_usd_per_1000_sessions_actual']:.3f} vs draft ${c['cascade_usd_per_1000_sessions_draft3']:.3f}")
        for rule, v in b["threshold_rule_sensitivity_preregistered"].items():
            print(f"  {rule}: spec {v['specialist_weighted']:.2f} gemma {v['gemma_weighted']:.2f} diff {v['minus_gemma']['difference_points']:+.2f} [{v['minus_gemma']['ci95'][0]:+.2f}, {v['minus_gemma']['ci95'][1]:+.2f}] p={v['minus_gemma']['p']:.4f}")


if __name__ == "__main__":
    main()

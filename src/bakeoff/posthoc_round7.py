"""POST HOC: are CLINC150's reused out-of-scope turns different from its never-seen ones? Not pre-registered; no API calls.

    python -m bakeoff.posthoc_round7

On the confirmatory CLINC150 sessions, 78 of the 185 out-of-scope turns had appeared in an earlier (exploratory) session,
because those sessions drew CLINC out-of-scope queries through BANKING77-OOS (posthoc_round3b.seen_texts). Specialist - Gemma
on out-of-scope turns is +4.0 on never-seen turns and -7.3 on seen ones (posthoc_round4.freshness_components). This module
asks whether that contrast is larger than random splits of the same turns give, and whether the two subsets differ:
- interaction: label permutation of 'seen' among the out-of-scope turns (each is its own cluster in the primary test), for
  the per-turn specialist - Gemma rejection difference (specialist: mean of its three seeds, prior-0.2 threshold rule);
- rejection: each system's rejection rate on seen and never-seen out-of-scope turns, with the same permutation test;
- content: word count, BM25 top-1 score against the in-scope training rows (the pool the general models retrieve from), and
  the specialist's detector score (seed 1), with the same permutation test.
All p-values are two-sided and unadjusted. Writes artifacts/posthoc/round7.json.
"""

from __future__ import annotations

import json

import numpy as np

from bakeoff import confirmatory as cf
from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "posthoc"
SEED = 20261018
DRAWS = 20000
HOSTED = {"gemma-4-31b (run A)": "A__clinc150__gemma-4-31b", "qwen3.5-27b (run B)": "B__clinc150__qwen3.5-27b", "gemma-4-31b with rows (run C)": "C__clinc150__gemma-4-31b"}


def permutation(x: np.ndarray, seen: np.ndarray, seed: int) -> dict:
    """Mean of x on seen minus never-seen items; two-sided p from random relabelling of which items are seen."""
    observed = x[seen].mean() - x[~seen].mean()
    rng = np.random.default_rng(seed)
    null = np.empty(DRAWS)
    for k in range(DRAWS):
        p = rng.permutation(seen)
        null[k] = x[p].mean() - x[~p].mean()
    return {"seen": float(x[seen].mean()), "never_seen": float(x[~seen].mean()), "seen_minus_never_seen": float(observed), "p": float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (DRAWS + 1))}


def main() -> None:
    from bakeoff import posthoc_round3b as pb
    from bakeoff import revision_free as rf
    from bakeoff.workflow_retrieval import BM25

    d = cf.load()["clinc150"]
    turns = cf.turns_of(d)
    oos = np.asarray([not t["in_scope"] for t in turns])
    texts = [t["text"] for t, o in zip(turns, oos) if o]
    seen_all = pb.seen_texts()
    seen = np.asarray([t in seen_all for t in texts])

    sp = {s: rf.specialist(cf.features("clinc150", s), 0.2) for s in cf.SEEDS}
    rejected = {f"specialist seed {s}": (~sp[s]["scope"][oos]).astype(float) for s in cf.SEEDS}
    rejected["specialist (mean of seeds)"] = np.mean([rejected[f"specialist seed {s}"] for s in cf.SEEDS], axis=0)
    for name, stem in HOSTED.items():
        answers = cf._hosted_answers(cf.OUT / "hosted" / f"{stem}.jsonl", turns, len(d["labels"]))
        rejected[name] = (~np.asarray(answers["scope"], bool)[oos]).astype(float)

    index = BM25([r["text"] for r in d["train"]])

    def top1(text: str) -> float:
        q = index.vectorizer.transform([text])
        q.data[:] = 1.0
        return float((index.matrix @ q.T).max())

    first = sp[cf.SEEDS[0]]
    content = {
        "words": np.asarray([len(t.split()) for t in texts], float),
        "bm25_top1_vs_in_scope_training_rows": np.asarray([top1(t) for t in texts]),
        f"specialist_detector_score_seed_{cf.SEEDS[0]} ({first['detector']})": np.asarray(first["scores"]["turns"][first["detector"]], float)[oos],
    }

    diff = rejected["specialist (mean of seeds)"] - rejected["gemma-4-31b (run A)"]
    result = {
        "status": "POST HOC: requested after the 2026-10-18 reading of the freshness paragraph; not pre-registered; p unadjusted",
        "n_out_of_scope": int(oos.sum()),
        "n_seen": int(seen.sum()),
        "n_never_seen": int((~seen).sum()),
        "draws": DRAWS,
        "interaction_specialist_minus_gemma_rejection": {k: 100 * v if k != "p" else v for k, v in permutation(diff, seen, SEED).items()},
        "rejection_percent": {name: {k: 100 * v if k != "p" else v for k, v in permutation(x, seen, SEED + 1 + i).items()} for i, (name, x) in enumerate(rejected.items())},
        "content": {name: permutation(x, seen, SEED + 100 + i) for i, (name, x) in enumerate(content.items())},
    }
    result["min_p_single_system_or_content"] = min([v["p"] for v in result["rejection_percent"].values()] + [v["p"] for v in result["content"].values()])
    (OUT / "round7.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")

    i = result["interaction_specialist_minus_gemma_rejection"]
    print(f"CLINC150 out-of-scope: {result['n_seen']} seen, {result['n_never_seen']} never seen")
    print(f"specialist - Gemma rejection: seen {i['seen']:+.1f}, never seen {i['never_seen']:+.1f}, contrast {i['seen_minus_never_seen']:+.1f}, p={i['p']:.4f}")
    for name, v in result["rejection_percent"].items():
        print(f"  {name:32s} seen {v['seen']:5.1f}  never seen {v['never_seen']:5.1f}  p={v['p']:.3f}")
    for name, v in result["content"].items():
        print(f"  {name:48s} seen {v['seen']:9.3f}  never seen {v['never_seen']:9.3f}  p={v['p']:.3f}")
    print(f"smallest p among single systems and content: {result['min_p_single_system_or_content']:.3f}")


if __name__ == "__main__":
    main()

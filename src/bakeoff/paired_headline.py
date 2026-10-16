"""Paired tests for the main comparison (outline table T3): best-equipped specialist against each hosted model with retrieval.

Post hoc: written after the numbers in audit sections 3.29 to 3.36 were seen, to put paired intervals and tests on them.
The specialist is the first-seed model with its dev-selected scope detector. Sessions (conversations) are the unit.
"""

from __future__ import annotations

import json
from math import comb

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

OUT = ROOT / "artifacts" / "robustness"


def session_flags(sessions, decisions) -> np.ndarray:
    flags = []
    for turns, rows in zip(sessions, decisions):
        ok = True
        for turn, row in zip(turns, rows):
            ok = ok and row["scope"] == turn["in_scope"]
            if turn["in_scope"]:
                ok = ok and row["intent"] == turn["intent"] and (turn["repeat"] is None or row["repeat"] == turn["repeat"])
        flags.append(ok)
    return np.asarray(flags)


def _mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = min(1.0, 2.0 * sum(comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2.0**n) if n else 1.0
    return {"only_specialist": only_a, "only_model": only_b, "exact_p": p}


def _compare(a: np.ndarray, b: np.ndarray, seed: int = 20261015) -> dict:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(a), size=(5000, len(a)))
    diff = (a[picks].mean(axis=1) - b[picks].mean(axis=1)) * 100
    return {"specialist": float(a.mean()), "model": float(b.mean()), "difference_points": float((a.mean() - b.mean()) * 100), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "mcnemar": _mcnemar(a, b), "n": int(len(a))}


def _specialist(features: dict, dev_gold: np.ndarray, turns) -> dict:
    """Dev-selected detector among softmax, energy, Mahalanobis and nearest neighbour, as in section 3.30."""
    from bakeoff.robustness_seeds import _detector_scores

    temperature = fit_temperature(features["dev_in_logits"], dev_gold)["temperature"]
    scores = _detector_scores(features, temperature)
    rules = {d: v1._scope_threshold(scores["dev_in"][d], scores["dev_out"][d]) for d in scores["turns"]}
    best = max(rules, key=lambda d: rules[d]["dev_balanced_accuracy"])
    scope = scores["turns"][best] >= rules[best]["tau"]
    intent = features["turns_logits"].argmax(axis=1)
    return {"detector": best, "answers": {turn["id"]: {"scope": bool(s), "intent": int(i)} for turn, s, i in zip(turns, scope, intent)}}


def _table(path, n_labels) -> dict:
    return {key: v4.parse(row["text"], n_labels) for key, row in v4._read(path).items()}


def main() -> None:
    from bakeoff import workflow_near_oos as wn
    from bakeoff.oos_detectors import _domain

    results = {}
    synthetic = {name: ROOT / "artifacts" / "oos-detectors" / f"{name}__features.npz" for name in ("banking77", "massive", "clinc150")}
    for name, cache in synthetic.items():
        domain = _domain(name)
        sessions, labels = domain["sessions"], domain["labels"]
        turns = [turn for session in sessions for turn in session]
        dev_gold = np.asarray([g for _, g in domain["dev_in"]], dtype=np.int64)
        spec = _specialist(dict(np.load(cache)), dev_gold, turns)
        a = session_flags(sessions, v4.decide(sessions, lambda turn: spec["answers"][turn["id"]]))
        for model, path in (("gemma-4-31b", ROOT / "artifacts" / "workflow-retrieval" / f"{name}__gemma-4-31b__full_train.jsonl"), ("qwen3.5-27b", OUT / f"{name}__qwen3.5-27b__full_train.jsonl")):
            table = _table(path, len(labels))
            b = session_flags(sessions, v4.decide(sessions, lambda turn: table[turn["id"]]))
            results[f"{name} | specialist ({spec['detector']}) vs {model} retrieval"] = _compare(a, b)

    data = wn._data()
    near = wn._sessions(data)
    dev_gold = np.asarray([item.gold_index for item in data["dev_in"]], dtype=np.int64)
    for kind, sessions in near.items():
        turns = [turn for session in sessions for turn in session]
        spec = _specialist(dict(np.load(ROOT / "artifacts" / "oos-detectors" / f"b77oos-{kind}__features.npz")), dev_gold, turns)
        a = session_flags(sessions, v4.decide(sessions, lambda turn: spec["answers"][turn["id"]]))
        for model, path in (("gemma-4-31b", ROOT / "artifacts" / "workflow-near-oos" / f"gemma__{kind}__retrieval.jsonl"), ("qwen3.5-27b", OUT / f"b77oos__qwen3.5-27b__{kind}__retrieval.jsonl")):
            table = _table(path, len(data["labels"]))
            b = session_flags(sessions, v4.decide(sessions, lambda turn: table[turn["id"]]))
            results[f"banking77_oos {kind} | specialist ({spec['detector']}) vs {model} retrieval"] = _compare(a, b)

    from bakeoff import workflow_multidogo as wm

    dev, test = wm.load("dev"), wm.load("test")
    test_turns = [turn for conversation in test for turn in conversation]
    scores = dict(np.load(ROOT / "artifacts" / "workflow-multidogo" / "encoder__scores.npz"))
    temperature = fit_temperature(scores["dev"], np.asarray([t["label"] for c in dev for t in c], dtype=np.int64))["temperature"]
    predicted = apply_temperature(scores["test"], temperature).argmax(axis=1)
    encoder = {turn["id"]: {"label": int(p), "repeat": None} for turn, p in zip(test_turns, predicted)}

    def conversation_flags(table) -> np.ndarray:
        flags = []
        for conversation in test:
            gold_repeat = wm._repeat_gold(conversation)
            last, ok = None, True
            for turn, wanted in zip(conversation, gold_repeat):
                answer = table[turn["id"]]
                right = answer["label"] == turn["label"]
                typed = last is not None and answer["label"] == last
                if 0 <= answer["label"] < len(wm.TASKS):
                    last = answer["label"]
                if wanted is not None:
                    right = right and typed == wanted
                ok = ok and right
            flags.append(ok)
        return np.asarray(flags)

    a = conversation_flags(encoder)
    for model, path in (("gemma-4-31b", ROOT / "artifacts" / "workflow-multidogo" / "gemma_retrieval_current.jsonl"), ("qwen3.5-27b", OUT / "multidogo__qwen3.5-27b__retrieval_current.jsonl")):
        table = {key: wm._parse(row["text"]) for key, row in v4._read(path).items()}
        results[f"multidogo | specialist vs {model} retrieval"] = _compare(a, conversation_flags(table))

    (OUT / "paired_headline.json").write_text(json.dumps(results, indent=2) + "\n")
    for key, row in results.items():
        print(
            f"{key:70s} specialist {100 * row['specialist']:5.1f} model {100 * row['model']:5.1f} diff {row['difference_points']:+5.1f} [{row['ci'][0]:+.1f}, {row['ci'][1]:+.1f}] "
            f"McNemar {row['mcnemar']['only_specialist']}/{row['mcnemar']['only_model']} p={row['mcnemar']['exact_p']:.3g} n={row['n']}"
        )


if __name__ == "__main__":
    main()

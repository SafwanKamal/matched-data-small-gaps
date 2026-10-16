"""POST HOC analyses of run D (retrieval suppression), written after its results were seen. Nothing here was pre-registered.

Q1  operating point or specific effect: signal-detection discrimination (d') and criterion (c) of the scope decision per
    model, split and arm, with cluster-bootstrap intervals; plus the old off-topic BANKING77-OOS sessions as a contrast.
Q2  why the splits disagree: zero-shot over-rejection, how close held-out items are to the retrieval pool, label copying,
    and suppression per held-out intent against that closeness.
Q3  zero-cost mitigations on saved answers (reviewer 4, round 2): a retrieval-similarity floor chosen on dev, a
    neighbour-label rule, and a specialist veto.
Output: artifacts/posthoc/round3.json and a printed summary.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm, spearmanr

from bakeoff import retrieval_suppression as rs
from bakeoff.protocol import ROOT

OUT = ROOT / "artifacts" / "posthoc"
GROUPS = {"banking77_oos": ["banking77_oos"], "clinc_single_domain": ["clinc_banking", "clinc_credit_cards"]}
SEED = 20261016


def bm25_scores(index, query: str) -> np.ndarray:
    q = index.vectorizer.transform([query])
    q.data[:] = 1.0
    return np.asarray((index.matrix @ q.T).todense()).ravel()


def sdt(reject: np.ndarray, in_scope: np.ndarray) -> tuple[float, float]:
    """d' and criterion c for 'reject' as the response to a held-out item; rates clipped by 0.5/n."""
    held, inside = reject[~in_scope], reject[in_scope]
    h = np.clip(held.mean(), 0.5 / len(held), 1 - 0.5 / len(held))
    f = np.clip(inside.mean(), 0.5 / len(inside), 1 - 0.5 / len(inside))
    return float(norm.ppf(h) - norm.ppf(f)), float(-(norm.ppf(h) + norm.ppf(f)) / 2)


def boot_sdt(arms: dict, in_scope: np.ndarray, clusters: np.ndarray, draws: int = 2000) -> dict:
    """Cluster bootstrap (held-out and in-scope intents resampled separately) of d', c and their retrieval-minus-zero-shot changes."""
    rng = np.random.default_rng(SEED)
    keys_in, keys_out = np.unique(clusters[in_scope]), np.unique(clusters[~in_scope])
    members = {k: np.where(clusters == k)[0] for k in np.unique(clusters)}
    stats = {name: [] for name in arms}
    for _ in range(draws):
        idx = np.concatenate([members[k] for k in rng.choice(keys_in, len(keys_in))] + [members[k] for k in rng.choice(keys_out, len(keys_out))])
        for name, reject in arms.items():
            stats[name].append(sdt(reject[idx], in_scope[idx]))
    out = {}
    for name, reject in arms.items():
        d, c = sdt(reject, in_scope)
        s = np.asarray(stats[name])
        out[name] = {"d_prime": d, "criterion": c, "d_prime_ci": np.percentile(s[:, 0], [2.5, 97.5]).tolist(), "criterion_ci": np.percentile(s[:, 1], [2.5, 97.5]).tolist()}
    if "zero_shot" in arms and "retrieval" in arms:
        diff = np.asarray(stats["retrieval"]) - np.asarray(stats["zero_shot"])
        out["retrieval_minus_zero_shot"] = {
            "d_prime": out["retrieval"]["d_prime"] - out["zero_shot"]["d_prime"],
            "d_prime_ci": np.percentile(diff[:, 0], [2.5, 97.5]).tolist(),
            "criterion": out["retrieval"]["criterion"] - out["zero_shot"]["criterion"],
            "criterion_ci": np.percentile(diff[:, 1], [2.5, 97.5]).tolist(),
        }
    return out


def youden_threshold(dev_in: np.ndarray, dev_out: np.ndarray) -> float:
    """Similarity floor: in scope when the score is at least tau; tau maximises balanced accuracy on dev."""
    best = None
    for tau in np.unique(np.concatenate([dev_in, dev_out])):
        value = 0.5 * ((dev_in >= tau).mean() + (dev_out < tau).mean())
        if best is None or value > best[1]:
            best = (float(tau), float(value))
    return best[0]


def run() -> dict:
    from bakeoff import api_baseline as api
    from bakeoff import revision_free as rf
    from bakeoff.workflow_retrieval import BM25, embed

    spec = rs.rules()
    data = rs.load()
    result = {"status": "POST HOC: written after run D's results were seen; not pre-registered", "groups": {}}
    for group, splits in GROUPS.items():
        items = [i for s in splits for i in data[s]["items"]]
        in_scope = np.asarray([i["in_scope"] for i in items])
        gold = np.asarray([i["intent"] if i["in_scope"] else -1 for i in items])
        clusters = np.asarray([i["cluster"] for i in items])
        # Retrieval neighbours and similarity, recomputed exactly as the prompts were built.
        top_bm25, top_dense, neighbour_labels, dev_scores = [], [], [], {"bm25": ([], []), "dense": ([], [])}
        for s in splits:
            block = data[s]
            pool_texts = [t for t, _ in block["pool"]]
            index = BM25(pool_texts)
            pool_emb = embed(pool_texts)
            texts = [i["text"] for i in block["items"]]
            emb = embed(texts)
            for text, e in zip(texts, emb):
                sc = bm25_scores(index, text)
                order = np.argsort(-sc, kind="stable")[:20]
                top_bm25.append(float(sc[order[0]]))
                neighbour_labels.append([block["pool"][j][1] for j in order])
            top_dense.extend((emb @ pool_emb.T).max(axis=1).tolist())
            for part, texts_dev in (("in", [t for t, _ in block["dev_in"]]), ("out", list(block["dev_out"]))):
                dev_scores["bm25"][0 if part == "in" else 1].extend(float(bm25_scores(index, t).max()) for t in texts_dev)
                dev_scores["dense"][0 if part == "in" else 1].extend((embed(texts_dev) @ pool_emb.T).max(axis=1).tolist())
        top_bm25, top_dense = np.asarray(top_bm25), np.asarray(top_dense)
        g = {"n_in": int(in_scope.sum()), "n_held_out": int((~in_scope).sum()), "models": {}}
        # Q2: how close the held-out items are to the pool (AUROC of top-1 similarity for in-scope versus held-out).
        from bakeoff.confirmatory import _auroc

        g["closeness"] = {
            "bm25_top1_mean": {"in_scope": float(top_bm25[in_scope].mean()), "held_out": float(top_bm25[~in_scope].mean())},
            "dense_top1_mean": {"in_scope": float(top_dense[in_scope].mean()), "held_out": float(top_dense[~in_scope].mean())},
            "auroc_in_vs_held_bm25": _auroc(top_bm25, in_scope),
            "auroc_in_vs_held_dense": _auroc(top_dense, in_scope),
        }
        # Similarity floors chosen on dev (dev_out are held-out-intent validation items, as for the specialist).
        floors = {}
        for kind, scores in (("bm25", top_bm25), ("dense", top_dense)):
            tau = youden_threshold(np.asarray(dev_scores[kind][0]), np.asarray(dev_scores[kind][1]))
            floors[kind] = {"tau": tau, "accept": scores >= tau}
        g["floors_alone"] = {k: {"held_out_rejected": float((~v["accept"])[~in_scope].mean()), "in_scope_kept": float(v["accept"][in_scope].mean())} for k, v in floors.items()}
        # Specialist veto: mean of three seeds' scope decisions (majority).
        spec_scope = []
        for seed in rs.SEEDS:
            spec_scope.append(np.concatenate([rf.specialist(rs.features(s, seed), 0.2)["scope"] for s in splits]))
        spec_accept = np.mean(spec_scope, axis=0) >= 0.5
        for name in spec["models"]:
            arms = {}
            for arm in rs.ARMS:
                parts = [rs._answers(rs._path(name, s, arm), data[s]["items"], len(data[s]["labels"])) for s in splits]
                arms[arm] = {"scope": np.concatenate([p["scope"] for p in parts]), "intent": np.concatenate([p["intent"] for p in parts])}
            m = {"sdt": boot_sdt({a: ~arms[a]["scope"] for a in ("zero_shot", "retrieval", "mitigation_instruction")}, in_scope, clusters)}
            # Label copying under retrieval: accepted items whose predicted intent is the most frequent retrieved label.
            ret = arms["retrieval"]
            codes_pred = ret["intent"]
            majority = np.asarray([max(set(labels), key=labels.count) for labels in neighbour_labels])
            in_list = np.asarray([p in labels for p, labels in zip(codes_pred, neighbour_labels)])
            accepted = ret["scope"]
            m["label_copying"] = {
                "held_out_accepted_predicts_majority_neighbour": float((codes_pred == majority)[accepted & ~in_scope].mean()) if (accepted & ~in_scope).any() else None,
                "in_scope_accepted_predicts_majority_neighbour": float((codes_pred == majority)[accepted & in_scope].mean()),
                "held_out_accepted_predicts_a_neighbour_label": float(in_list[accepted & ~in_scope].mean()) if (accepted & ~in_scope).any() else None,
            }
            # Q2: suppression per held-out intent against closeness.
            per = []
            for k in np.unique(clusters[~in_scope]):
                sel = clusters == k
                per.append(((~arms["zero_shot"]["scope"])[sel].mean() - (~ret["scope"])[sel].mean(), top_dense[sel].mean()))
            per = np.asarray(per)
            m["suppression_vs_closeness_spearman"] = float(spearmanr(per[:, 0], per[:, 1]).correlation) if len(per) > 3 else None
            # Q3: zero-cost mitigations on the retrieval answers.
            mit = {}
            rules_ = {
                "retrieval (as run)": ret["scope"],
                "zero-shot (as run)": arms["zero_shot"]["scope"],
                "instruction (as run)": arms["mitigation_instruction"]["scope"],
                "retrieval + neighbour-label rule": ret["scope"] & in_list,
                "retrieval + BM25 floor": ret["scope"] & floors["bm25"]["accept"],
                "retrieval + dense floor": ret["scope"] & floors["dense"]["accept"],
                "retrieval + specialist veto": ret["scope"] & spec_accept,
                "instruction + specialist veto": arms["mitigation_instruction"]["scope"] & spec_accept,
                "specialist alone": spec_accept,
            }
            for key, scope in rules_.items():
                intent = arms["zero_shot"]["intent"] if key.startswith("zero") else (arms["mitigation_instruction"]["intent"] if key.startswith("instruction") else ret["intent"])
                if key == "specialist alone":
                    continue
                mit[key] = {
                    "held_out_rejected": float((~scope)[~in_scope].mean()),
                    "in_scope_kept": float(scope[in_scope].mean()),
                    "in_scope_right": float((scope & (intent == gold))[in_scope].mean()),
                    "balanced": float(0.5 * ((~scope)[~in_scope].mean() + scope[in_scope].mean())),
                }
            m["mitigations"] = mit
            g["models"][name] = m
        g["specialist_majority"] = {"held_out_rejected": float((~spec_accept)[~in_scope].mean()), "in_scope_kept": float(spec_accept[in_scope].mean())}
        result["groups"][group] = g
    # Contrast: the old off-topic BANKING77-OOS sessions (exploratory data), zero-shot against retrieval.
    from bakeoff import workflow_near_oos as wn
    from bakeoff import workflow_v4 as v4

    contrast = {}
    sessions = wn._sessions(wn._data())
    for kind in ("id", "ood"):
        turns = [t for s in sessions[kind] for t in s]
        inside = np.asarray([t["in_scope"] for t in turns])
        for model, pattern in (("gemma-4-31b", wn.OUT / "gemma__{k}__{a}.jsonl"), ("qwen3.5-27b", ROOT / "artifacts" / "robustness" / "b77oos__qwen3.5-27b__{k}__{a}.jsonl")):
            row = {}
            for arm in ("zero_shot", "retrieval"):
                rows = v4._read(pattern.with_name(pattern.name.format(k=kind, a=arm)))
                scope = np.asarray([v4.parse(rows[t["id"]]["text"], 50)["scope"] if t["id"] in rows else (not t["in_scope"]) for t in turns])
                row[arm] = {"out_rejected": float((~scope)[~inside].mean()), "in_scope_kept": float(scope[inside].mean()), "d_prime_c": sdt(~scope, inside)}
            contrast[f"{model} | {kind}"] = row
    result["old_sessions_contrast"] = {"note": "exploratory: the 400 old BANKING77-OOS sessions, id = held-out intents, ood = off-topic", "rows": contrast}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "round3.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def report(result: dict) -> None:
    for group, g in result["groups"].items():
        c = g["closeness"]
        print(f"\n{group}: closeness dense top-1 in {c['dense_top1_mean']['in_scope']:.3f} held {c['dense_top1_mean']['held_out']:.3f} | AUROC in-vs-held bm25 {c['auroc_in_vs_held_bm25']:.3f} dense {c['auroc_in_vs_held_dense']:.3f}")
        print(f"  floors alone: " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in g["floors_alone"].items()}) + f" | specialist majority {g['specialist_majority']}")
        for name, m in g["models"].items():
            s = m["sdt"]
            print(f"  {name:16s} d' zs {s['zero_shot']['d_prime']:.2f} ret {s['retrieval']['d_prime']:.2f} instr {s['mitigation_instruction']['d_prime']:.2f} | c zs {s['zero_shot']['criterion']:+.2f} ret {s['retrieval']['criterion']:+.2f} instr {s['mitigation_instruction']['criterion']:+.2f} | "
                  f"ret-zs d' {s['retrieval_minus_zero_shot']['d_prime']:+.2f} [{s['retrieval_minus_zero_shot']['d_prime_ci'][0]:+.2f}, {s['retrieval_minus_zero_shot']['d_prime_ci'][1]:+.2f}] c {s['retrieval_minus_zero_shot']['criterion']:+.2f} [{s['retrieval_minus_zero_shot']['criterion_ci'][0]:+.2f}, {s['retrieval_minus_zero_shot']['criterion_ci'][1]:+.2f}]")
            lc = m["label_copying"]
            print(f"     copying: held-out accepted -> majority neighbour {lc['held_out_accepted_predicts_majority_neighbour']}, in-scope {lc['in_scope_accepted_predicts_majority_neighbour']:.3f}; suppression~closeness rho {m['suppression_vs_closeness_spearman']}")
            for key, v in m["mitigations"].items():
                print(f"     {key:34s} rejected {100 * v['held_out_rejected']:5.1f} kept {100 * v['in_scope_kept']:5.1f} right {100 * v['in_scope_right']:5.1f} balanced {100 * v['balanced']:5.1f}")
    print("\nold sessions contrast (exploratory):")
    for key, row in result["old_sessions_contrast"]["rows"].items():
        print(f"  {key:22s} " + " | ".join(f"{a}: out rejected {100 * v['out_rejected']:.1f} kept {100 * v['in_scope_kept']:.1f} d' {v['d_prime_c'][0]:.2f} c {v['d_prime_c'][1]:+.2f}" for a, v in row.items()))


if __name__ == "__main__":
    report(run())

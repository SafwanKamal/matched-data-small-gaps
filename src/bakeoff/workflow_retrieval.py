"""Retrieved demonstrations for the hosted models on all three workflows. Rules: workflow_retrieval_protocol.json."""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "workflow_retrieval_protocol.json"
OUT = ROOT / "artifacts" / "workflow-retrieval"
NEWLINE = "\n"


def _domain(name: str) -> dict:
    """Everything a domain needs: prompt rules, labels, sessions, training rows, out-of-scope pool, specialist, saved runs."""
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    if name == "banking77":
        from datasets import load_dataset

        from bakeoff.data import load_items
        from bakeoff.data_fraction import _block
        from bakeoff.protocol import load_protocol
        from bakeoff.suite import load_suite_protocol
        from bakeoff.train_ettin import _train_rows

        protocol = load_protocol()
        materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
        labels, _, dev, _, _ = v1._sources()
        sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
        turns = [turn for session in sessions for turn in session]
        clinc = load_suite_protocol()["datasets"]["clinc150"]
        raw = load_dataset(clinc["hf_id"], clinc["config"], revision=clinc["revision"])["train"]
        oos = raw.features["intent"].names.index("oos")
        return {
            "rules": rules_v4,
            "labels": labels,
            "sessions": sessions,
            "train": _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"])),
            "out_pool": [example["text"] for example in raw if example["intent"] == oos],
            "specialist": _block(dict(np.load(ROOT / "artifacts" / "data-fraction" / "scores" / "encoder-150m@1.npz")), dev, turns),
            "zero_shot": ROOT / "artifacts" / "workflow-v4",
            "fixed": {"gemma-4-31b": ROOT / "artifacts" / "workflow-v5" / "d__gemma-4-31b__3shot.jsonl", "qwen3.5-9b": ROOT / "artifacts" / "workflow-v5" / "d__qwen3.5-9b__3shot.jsonl"},
        }
    if name == "massive":
        from bakeoff import workflow_massive as m

        rules = json.loads(m.RULES_PATH.read_text(encoding="utf-8"))
        rules["models"] = rules_v4["models"]
        loaded, sessions, _ = m._sources(rules)
        _, oos_dev_all = m._banking_out_of_scope()
        return {
            "rules": rules,
            "labels": loaded["labels"],
            "sessions": sessions,
            "train": [(item.text, item.gold_index) for item in loaded["items"]["train"]],
            "out_pool": oos_dev_all[104:],
            "specialist": m.specialists(rules, loaded, sessions, oos_dev_all)["encoder"]["answers"],
            "zero_shot": m.OUT,
            "fixed": {"gemma-4-31b": m.OUT / "d__gemma-4-31b__3shot.jsonl", "qwen3.5-9b": m.OUT / "d__qwen3.5-9b__3shot.jsonl"},
        }
    from bakeoff import workflow_clinc as c

    rules = json.loads(c.RULES_PATH.read_text(encoding="utf-8"))
    rules["models"] = rules_v4["models"]
    data = c._data(rules)
    sessions = c._sessions(rules, data)
    return {
        "rules": rules,
        "labels": data["labels"],
        "sessions": sessions,
        "train": [(item.text, item.gold_index) for item in data["train_in"]],
        "out_pool": data["train_out"],
        "specialist": c.specialists(rules, data, sessions)["encoder"]["answers"],
        "zero_shot": c.OUT,
        "fixed": {"gemma-4-31b": c.OUT / "d__gemma-4-31b__3shot.jsonl"},
    }


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        tf = self.vectorizer.fit_transform(texts).astype(np.float64).tocsr()
        n = tf.shape[0]
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        norm = k1 * (1.0 - b + b * lengths / max(lengths.mean(), 1e-9))
        coo = tf.tocoo()
        weights = idf[coo.col] * coo.data * (k1 + 1.0) / (coo.data + norm[coo.row])
        self.matrix = sparse.csr_matrix((weights, (coo.row, coo.col)), shape=tf.shape)

    def top(self, query: str, k: int) -> list[int]:
        q = self.vectorizer.transform([query])
        q.data[:] = 1.0
        scores = np.asarray((self.matrix @ q.T).todense()).ravel()
        return np.argsort(-scores, kind="stable")[:k].tolist()


class DenseIndex:
    """all-mpnet-base-v2 sentence embeddings, mean-pooled and unit-normalised; cosine nearest neighbours."""

    def __init__(self, texts: list[str]):
        self.matrix = embed(texts)

    def top(self, query: str, k: int) -> list[int]:
        return self.top_many([query], k)[0]

    def top_many(self, queries: list[str], k: int) -> list[list[int]]:
        scores = embed(queries) @ self.matrix.T
        return [np.argsort(-row, kind="stable")[:k].tolist() for row in scores]


_EMBEDDER = {}
EMBEDDER_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDER_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not _EMBEDDER:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDER["tokenizer"] = AutoTokenizer.from_pretrained(EMBEDDER_NAME, revision=EMBEDDER_REVISION)
        _EMBEDDER["model"] = AutoModel.from_pretrained(EMBEDDER_NAME, revision=EMBEDDER_REVISION).to(device).eval()
        _EMBEDDER["device"] = device
    tokenizer, model, device = _EMBEDDER["tokenizer"], _EMBEDDER["model"], _EMBEDDER["device"]
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), 128):
            batch = tokenizer(texts[start : start + 128], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            out.append(torch.nn.functional.normalize(pooled, dim=1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def dense() -> None:
    """Amendment of 2026-10-15: dense retrieval for Gemma 4 31B on the three synthetic domains."""
    import httpx

    from bakeoff.paired_headline import _compare, session_flags

    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    part = spec["amendment_2026-10-15_dense"]
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    v4.OUT = OUT
    budget = api.Budget(api_rules, run_cap=part["run_cap_usd"])
    path = OUT / "summary_dense.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": spec["protocol_id"], "domains": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        for name in ("banking77", "massive", "clinc150"):
            domain = _domain(name)
            rules, labels, sessions = domain["rules"], domain["labels"], domain["sessions"]
            turns = [turn for session in sessions for turn in session]
            system = v4._system_prompt(rules, labels, "current_only", False)
            pool = _pools(domain, spec["seed"])["full_train"]
            codes = api.codes_for(len(labels))
            out_code = "0" * len(codes[0])
            gold_code = [codes[turn["intent"]] if turn["in_scope"] else out_code for turn in turns]
            k = spec["retrieval"]["k"]
            dense_picks = DenseIndex([text for text, _ in pool]).top_many([turn["text"] for turn in turns], k)
            bm25 = BM25([text for text, _ in pool])
            bm25_picks = [bm25.top(turn["text"], k) for turn in turns]

            def hit(picks) -> float:
                return float(np.mean([gold in {pool[i][1] for i in chosen} for gold, chosen in zip(gold_code, picks)]))

            jobs = []
            for turn, picked in zip(turns, dense_picks):
                lines = NEWLINE.join(f"{pool[i][0]} => {pool[i][1]}" for i in picked[::-1])
                jobs.append((turn["id"], system, "Similar labelled examples (message => code):" + NEWLINE + lines + NEWLINE + NEWLINE + "Current message: " + turn["text"]))
            print(f"  {name} dense", flush=True)
            rows = v4._collect(client, rules["models"]["gemma-4-31b"], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"{name}__gemma-4-31b__full_train__dense.jsonl", budget)
            table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
            old = {key: v4.parse(row["text"], len(labels)) for key, row in v4._read(OUT / f"{name}__gemma-4-31b__full_train.jsonl").items()}
            new_flags = session_flags(sessions, v4.decide(sessions, lambda turn: table[turn["id"]]))
            old_flags = session_flags(sessions, v4.decide(sessions, lambda turn: old[turn["id"]]))
            result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: table[turn["id"]]), None)
            block = {
                "dense_alone": {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")},
                "dense_minus_bm25": _compare(new_flags, old_flags),
                "gold_in_top20": {"dense": hit(dense_picks), "bm25": hit(bm25_picks)},
                "usd_per_1000_sessions": 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions),
            }
            summary["domains"][name] = block
            path.write_text(json.dumps(summary, indent=2) + NEWLINE)
            d = block["dense_minus_bm25"]
            print(
                f"{name:10s} dense alone {100 * d['specialist']:.1f} [{100 * result['session_success_ci'][0]:.1f}, {100 * result['session_success_ci'][1]:.1f}] "
                f"intent {100 * result['node_accuracy']['intent']:.1f} scope {100 * result['node_accuracy']['scope']:.1f} | BM25 {100 * d['model']:.1f} | "
                f"difference {d['difference_points']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] McNemar p={d['mcnemar']['exact_p']:.3g} | "
                f"gold in top 20: dense {100 * block['gold_in_top20']['dense']:.1f} BM25 {100 * block['gold_in_top20']['bm25']:.1f} | ${block['usd_per_1000_sessions']:.2f}/1000",
                flush=True,
            )
    budget.save()
    print(f"DENSE_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


def _pools(domain: dict, seed: int) -> dict[str, list[tuple[str, str]]]:
    labels = domain["labels"]
    codes = api.codes_for(len(labels))
    out_code = "0" * len(codes[0])
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[str]] = {}
    for text, label in domain["train"]:
        by_label.setdefault(label, []).append(text)
    small = []
    for label in sorted(by_label):
        group = by_label[label]
        small += [(group[i], codes[label]) for i in rng.permutation(len(group))[:3]]
    small += [(text, out_code) for text in domain["out_pool"][:4]]
    full = [(text, codes[label]) for text, label in domain["train"]] + [(text, out_code) for text in domain["out_pool"]]
    return {"three_per_intent": small, "full_train": full}


def main() -> None:
    import httpx

    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else spec["domains"]
    budget = api.Budget(api_rules, run_cap=spec["run_cap_usd"])
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": spec["protocol_id"], "domains": {}}
    brief = lambda result: {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        for name in only:
            domain = _domain(name)
            rules, labels, sessions = domain["rules"], domain["labels"], domain["sessions"]
            turns = [turn for session in sessions for turn in session]
            system = v4._system_prompt(rules, labels, "current_only", False)
            pools = _pools(domain, spec["seed"])
            base = domain["specialist"]
            order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
            evaluate = lambda answer_of: brief(v1.evaluate(sessions, v4.decide(sessions, answer_of), None))
            block = {"specialist_alone": evaluate(lambda turn: base[turn["id"]]), "pool_sizes": {key: len(value) for key, value in pools.items()}, "runs": {}}
            print(f"{name}: specialist alone session {100 * block['specialist_alone']['session_success']:.1f}; pools {block['pool_sizes']}", flush=True)
            users = {}
            for pool_name, pool in pools.items():
                index = BM25([text for text, _ in pool])
                users[pool_name] = {}
                for turn in turns:
                    picked = index.top(turn["text"], spec["retrieval"]["k"])[::-1]
                    lines = "\n".join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
                    users[pool_name][turn["id"]] = f"Similar labelled examples (message => code):\n{lines}\n\nCurrent message: {turn['text']}"
            for model in spec["models"]:
                def saved(file):
                    rows = v4._read(file)
                    return {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()} if len(rows) >= len(turns) else None

                reference = {"zero_shot": saved(domain["zero_shot"] / f"a__{model}.jsonl"), "fixed_three_per_intent": saved(domain["fixed"][model]) if model in domain["fixed"] else None}
                for pool_name in pools:
                    jobs = [(turn["id"], system, users[pool_name][turn["id"]]) for turn in turns]
                    print(f"  {name} {model} {pool_name}", flush=True)
                    rows = v4._collect(client, rules["models"][model], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"{name}__{model}__{pool_name}.jsonl", budget)
                    table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
                    entry = {
                        "alone": evaluate(lambda turn: table[turn["id"]]),
                        "usd_per_1000_sessions": 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions),
                        "prompt_tokens_mean": float(np.mean([row["prompt_tokens"] or 0 for row in rows.values()])),
                        "cached_share": float(sum(row.get("cached_tokens") or 0 for row in rows.values()) / max(1, sum(row["prompt_tokens"] or 0 for row in rows.values()))),
                        "parse_failed": sum(value["parse_failed"] for value in table.values()),
                        "reference_alone": {key: (evaluate(lambda turn, t=value: t[turn["id"]]) if value else None) for key, value in reference.items()},
                        "hybrid": {},
                    }
                    for rate in (0.1, 0.25, 0.4, 0.6):
                        chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                        entry["hybrid"][str(rate)] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else base[turn["id"]])
                    block["runs"][f"{model}__{pool_name}"] = entry
                    refs = "  ".join(f"{key} {100 * value['session_success']:.1f}" for key, value in entry["reference_alone"].items() if value)
                    print(
                        f"{name:10s} {model:12s} {pool_name:16s} alone session {100 * entry['alone']['session_success']:.1f} [{100 * entry['alone']['session_success_ci'][0]:.1f}, {100 * entry['alone']['session_success_ci'][1]:.1f}] "
                        f"intent {100 * entry['alone']['node_accuracy']['intent']:.1f} scope {100 * entry['alone']['node_accuracy']['scope']:.1f} ${entry['usd_per_1000_sessions']:.2f}/1000 ({entry['prompt_tokens_mean']:.0f} tokens) | {refs} | "
                        f"hybrid " + " ".join(f"{rate}:{100 * cell['session_success']:.1f}" for rate, cell in entry["hybrid"].items()) + f" | run ${budget.run:.2f}",
                        flush=True,
                    )
                    summary["domains"][name] = block
                    summary["spend_usd"] = {"project_total": budget.total}
                    path.write_text(json.dumps(summary, indent=2) + "\n")
            budget.save()
    print(f"WORKFLOW_RETRIEVAL_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


def fractions() -> None:
    """Amendment of 2026-10-13: retrieval from the same training fraction the specialist was given, on banking."""
    import httpx

    from bakeoff.data_fraction import _block, _subsample

    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    part = spec["amendment_2026-10-13_data_fractions"]
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    v4.OUT = OUT
    domain = _domain("banking77")
    rules, labels, sessions = domain["rules"], domain["labels"], domain["sessions"]
    turns = [turn for session in sessions for turn in session]
    _, _, dev, _, _ = v1._sources()
    system = v4._system_prompt(rules, labels, "current_only", False)
    codes = api.codes_for(len(labels))
    out_code = "0" * len(codes[0])
    budget = api.Budget(api_rules, run_cap=3.0)
    brief = lambda result: {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}
    evaluate = lambda answer_of: brief(v1.evaluate(sessions, v4.decide(sessions, answer_of), None))
    path = OUT / "summary_fractions.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": spec["protocol_id"], "rows": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        for model, fraction_list in part["fractions"].items():
            for fraction in fraction_list:
                rows_train = _subsample(domain["train"], fraction, 20261012)
                pool = [(text, codes[label]) for text, label in rows_train] + [(text, out_code) for text in domain["out_pool"][: max(4, int(round(fraction * 250)))]]
                index = BM25([text for text, _ in pool])
                jobs = []
                for turn in turns:
                    picked = index.top(turn["text"], min(spec["retrieval"]["k"], len(pool)))[::-1]
                    lines = "\n".join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
                    jobs.append((turn["id"], system, f"Similar labelled examples (message => code):\n{lines}\n\nCurrent message: {turn['text']}"))
                print(f"  fraction {fraction:g} {model} pool {len(pool)}", flush=True)
                rows = v4._collect(client, rules["models"][model], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"banking77__{model}__frac-{fraction:g}.jsonl", budget)
                table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
                base = _block(dict(np.load(ROOT / "artifacts" / "data-fraction" / "scores" / f"encoder-150m@{fraction:g}.npz")), dev, turns)
                order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
                entry = {
                    "train_rows": len(rows_train),
                    "retrieval_alone": evaluate(lambda turn: table[turn["id"]]),
                    "encoder_alone": evaluate(lambda turn: base[turn["id"]]),
                    "usd_per_1000_sessions": 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions),
                    "hybrid": {},
                }
                for rate in (0.25, 0.4, 0.6):
                    chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                    entry["hybrid"][str(rate)] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else base[turn["id"]])
                summary["rows"][f"{model}@{fraction:g}"] = entry
                path.write_text(json.dumps(summary, indent=2) + "\n")
                print(
                    f"{model:12s} {fraction:5g} rows {entry['train_rows']:5d} retrieval alone {100 * entry['retrieval_alone']['session_success']:.1f} [{100 * entry['retrieval_alone']['session_success_ci'][0]:.1f}, {100 * entry['retrieval_alone']['session_success_ci'][1]:.1f}] "
                    f"intent {100 * entry['retrieval_alone']['node_accuracy']['intent']:.1f} | encoder alone {100 * entry['encoder_alone']['session_success']:.1f} | hybrid "
                    + " ".join(f"{rate}:{100 * cell['session_success']:.1f}" for rate, cell in entry["hybrid"].items()) + f" | ${entry['usd_per_1000_sessions']:.2f}/1000 run ${budget.run:.2f}",
                    flush=True,
                )
    budget.save()
    print(f"RETRIEVAL_FRACTIONS_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


def fractions_domain(name: str) -> None:
    """Second amendment of 2026-10-13: the matched-data sweep on MASSIVE and CLINC150.

    The hosted replies do not depend on the specialists, so this can run before the specialists of
    data_fraction_domains_protocol.json are trained; rerun it afterwards to fill the encoder columns from saved replies.
    """
    import httpx

    from bakeoff import data_fraction_domains as dfd
    from bakeoff.data_fraction import _subsample
    from bakeoff.metrics import apply_temperature
    from bakeoff.temperature import fit_temperature

    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    part = spec["amendment_2026-10-13_other_domains"]
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    v4.OUT = OUT
    domain = _domain(name)
    rules, labels, sessions = domain["rules"], domain["labels"], domain["sessions"]
    turns = [turn for session in sessions for turn in session]
    dev_in = dfd._domain(name)["dev_in"]
    gold = np.asarray([item.gold_index for item in dev_in], dtype=np.int64)
    system = v4._system_prompt(rules, labels, "current_only", False)
    codes = api.codes_for(len(labels))
    out_code = "0" * len(codes[0])
    budget = api.Budget(api_rules, run_cap=part["run_cap_usd"])
    brief = lambda result: {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}
    evaluate = lambda answer_of: brief(v1.evaluate(sessions, v4.decide(sessions, answer_of), None))
    path = OUT / f"summary_fractions__{name}.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": spec["protocol_id"], "domain": name, "rows": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        for model, fraction_list in part["fractions"].items():
            for fraction in fraction_list:
                rows_train = _subsample(domain["train"], fraction, 20261012)
                pool = [(text, codes[label]) for text, label in rows_train] + [(text, out_code) for text in domain["out_pool"][: max(4, int(round(fraction * len(domain["out_pool"]))))]]
                index = BM25([text for text, _ in pool])
                jobs = []
                for turn in turns:
                    picked = index.top(turn["text"], min(spec["retrieval"]["k"], len(pool)))[::-1]
                    lines = "\n".join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
                    jobs.append((turn["id"], system, f"Similar labelled examples (message => code):\n{lines}\n\nCurrent message: {turn['text']}"))
                print(f"  {name} fraction {fraction:g} {model} pool {len(pool)}", flush=True)
                rows = v4._collect(client, rules["models"][model], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"{name}__{model}__frac-{fraction:g}.jsonl", budget)
                table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
                entry = {"train_rows": len(rows_train), "retrieval_alone": evaluate(lambda turn: table[turn["id"]]), "usd_per_1000_sessions": 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions), "encoder_alone": None, "hybrid": {}}
                cache = dfd.OUT / "scores" / f"{name}__encoder@{fraction:g}.npz"
                if cache.exists():
                    scores = dict(np.load(cache))
                    temperature = fit_temperature(scores["dev_in_scope"], gold)["temperature"]
                    rule = v1._scope_threshold(apply_temperature(scores["dev_in_scope"], temperature).max(axis=1), apply_temperature(scores["dev_out_of_scope"], temperature).max(axis=1))
                    probs = apply_temperature(scores["session_turns"], temperature)
                    base = {turn["id"]: {"scope": bool(row.max() >= rule["tau"]), "intent": int(row.argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)}
                    order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
                    entry["encoder_alone"] = evaluate(lambda turn: base[turn["id"]])
                    for rate in (0.25, 0.4, 0.6):
                        chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                        entry["hybrid"][str(rate)] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else base[turn["id"]])
                summary["rows"][f"{model}@{fraction:g}"] = entry
                path.write_text(json.dumps(summary, indent=2) + "\n")
                encoder = f"{100 * entry['encoder_alone']['session_success']:.1f}" if entry["encoder_alone"] else "not trained yet"
                print(
                    f"{name:9s} {model:12s} {fraction:5g} rows {entry['train_rows']:5d} retrieval alone {100 * entry['retrieval_alone']['session_success']:.1f} [{100 * entry['retrieval_alone']['session_success_ci'][0]:.1f}, {100 * entry['retrieval_alone']['session_success_ci'][1]:.1f}] "
                    f"intent {100 * entry['retrieval_alone']['node_accuracy']['intent']:.1f} scope {100 * entry['retrieval_alone']['node_accuracy']['scope']:.1f} | encoder alone {encoder} | hybrid "
                    + " ".join(f"{rate}:{100 * cell['session_success']:.1f}" for rate, cell in entry["hybrid"].items()) + f" | ${entry['usd_per_1000_sessions']:.2f}/1000 run ${budget.run:.2f}",
                    flush=True,
                )
    budget.save()
    print(f"RETRIEVAL_FRACTIONS_DONE {name} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dense":
        dense()
    elif len(sys.argv) > 2 and sys.argv[1] == "fractions":
        for domain_name in sys.argv[2].split(","):
            fractions_domain(domain_name)
    elif len(sys.argv) > 1 and sys.argv[1] == "fractions":
        fractions()
    else:
        main()

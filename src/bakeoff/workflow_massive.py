"""The workflow v4/v5 ladder on a second domain, MASSIVE. Rules: workflow_massive_protocol.json."""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.data import load_items
from bakeoff.massive import load_massive
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "workflow_massive_protocol.json"
OUT = ROOT / "artifacts" / "workflow-massive"
ENCODERS = {
    "encoder": ROOT / "artifacts" / "size-sweep" / "encoder-150m" / "lr-0.0001" / "massive_en",
    "encoder_2e-5": ROOT / "artifacts" / "primary" / "ettin-encoder",
}
DECODER = ROOT / "artifacts" / "primary" / "ettin-decoder"


def _banking_out_of_scope():
    """Banking77 utterances that no MASSIVE skill covers: (test texts, dev texts)."""
    protocol = load_protocol()
    labels, test, dev, _, _ = v1._sources()
    drop = {i for i, label in enumerate(labels) if any(word in label for word in ("exchange", "currency", "fiat"))}
    return [item.text for item in test if item.gold_index not in drop], [item.text for item in dev if item.gold_index not in drop]


def _sources(rules):
    loaded = load_massive()
    oos_test, oos_dev = _banking_out_of_scope()
    path = OUT / "sessions.json"
    if path.exists():
        sessions = json.loads(path.read_text(encoding="utf-8"))
    else:
        sessions = v1.build_sessions(rules, loaded["items"]["test"], oos_test)
        path.write_text(json.dumps(sessions, indent=1) + "\n", encoding="utf-8")
    return loaded, sessions, oos_dev


def _score_texts(kind: str, folder, texts: list[str], labels, device: str) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if kind == "encoder":
        from bakeoff.train_ettin import EncoderClassifier, _encode

        torch.set_num_threads(10)
        tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
        encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32)
        model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels))
        model.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"])
        model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                encoded = _encode(tokenizer, texts[start : start + 32], 128, torch.device(device))
                rows.append(model(encoded["input_ids"], encoded["attention_mask"]).cpu().numpy().astype(np.float64))
        return np.concatenate(rows, axis=0)
    from bakeoff.data import Item
    from bakeoff.massive import load_primary_protocol
    from bakeoff.train_ettin_decoder import _score_items

    saved = json.loads((folder / "metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(folder / "model")
    model = AutoModelForCausalLM.from_pretrained(folder / "model", dtype=torch.float32)
    model.config.use_cache = False
    model.to("cuda")
    items = [Item(item_id=str(i), row=i, text=text, gold_index=0, gold_label="") for i, text in enumerate(texts)]
    scores, _, _ = _score_items(model, tokenizer, items, labels, load_primary_protocol()["ettin_decoder"], torch.device("cuda"), saved["batch_size"], saved["label_chunk_size"])
    return scores


def specialists(rules, loaded, sessions, oos_dev, device: str = "cpu") -> dict:
    labels = loaded["labels"]
    turns = [turn for session in sessions for turn in session]
    dev = loaded["items"]["dev"][:400]
    sets = {"session_turns": [t["text"] for t in turns], "dev_in_scope": [item.text for item in dev], "dev_out_of_scope": oos_dev[:100]}
    folders = {**{name: ("encoder", folder) for name, folder in ENCODERS.items()}, "decoder": ("decoder", DECODER)}
    out = {}
    for name, (kind, folder) in folders.items():
        cache = OUT / f"{name}__scores.npz"
        if not cache.exists():
            if kind == "decoder" and device != "cuda":
                continue
            scores = {part: _score_texts(kind, folder, texts, labels, device) for part, texts in sets.items()}
            np.savez_compressed(cache, **scores)
            print(f"scored {name}", flush=True)
        scores = dict(np.load(cache))
        gold = np.asarray([item.gold_index for item in dev], dtype=np.int64)
        temperature = fit_temperature(scores["dev_in_scope"], gold)["temperature"]
        dev_in = apply_temperature(scores["dev_in_scope"], temperature).max(axis=1)
        rule = v1._scope_threshold(dev_in, apply_temperature(scores["dev_out_of_scope"], temperature).max(axis=1))
        probs = apply_temperature(scores["session_turns"], temperature)
        out[name] = {
            "answers": {turn["id"]: {"scope": bool(row.max() >= rule["tau"]), "intent": int(row.argmax()), "top": float(row.max())} for turn, row in zip(turns, probs)},
            "probs": {turn["id"]: row for turn, row in zip(turns, probs)},
            "escalation_tau": float(np.quantile(dev_in, 0.25)),
            "rule": {**rule, "temperature": temperature},
        }
    return out


def _few_shot_system(rules, loaded, oos_dev, k: int) -> str:
    labels = loaded["labels"]
    rng = np.random.default_rng(rules["seed"])
    codes = api.codes_for(len(labels))
    by_label: dict[int, list[str]] = {}
    for item in loaded["items"]["train"]:
        by_label.setdefault(item.gold_index, []).append(item.text)
    examples = []
    for label in sorted(by_label):
        group = by_label[label]
        examples += [(group[i], codes[label]) for i in rng.permutation(len(group))[:k]]
    examples += [(text, "00") for text in oos_dev[100:104]]
    order = rng.permutation(len(examples))
    return v4._system_prompt(rules, labels, "current_only", False) + "\n\nLabelled examples (message => code):\n" + "\n".join(f"{examples[i][0]} => {examples[i][1]}" for i in order)


def part_d(rules, rules_v4, loaded, sessions, oos_dev, answers, spec, client, budget) -> dict:
    labels = loaded["labels"]
    out = {}
    brief = lambda result: {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}
    for name, runs in rules["part_d_few_shot"]["runs"].items():
        out[name] = {}
        for k, extent in runs.items():
            subset = sessions if extent.startswith("all") else sessions[:150]
            turns = [turn for session in subset for turn in session]
            system = _few_shot_system(rules, loaded, oos_dev, int(k))
            jobs = [(turn["id"], system, rules["request"]["user"]["current_only"].format(text=turn["text"])) for turn in turns]
            print(f"  part D {name} {k}-shot on {len(subset)} sessions", flush=True)
            rows = v4._collect(client, rules_v4["models"][name], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"d__{name}__{k}shot.jsonl", budget)
            table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
            evaluate = lambda answer_of: brief(v1.evaluate(subset, v4.decide(subset, answer_of), None))
            base = spec["encoder"]["answers"]
            order = sorted(turns, key=lambda turn: base[turn["id"]]["top"])
            entry = {
                "n_sessions": len(subset),
                "alone": evaluate(lambda turn: table[turn["id"]]),
                "zero_shot_alone_same_sessions": evaluate(lambda turn: answers[name][turn["id"]]),
                "encoder_alone_same_sessions": evaluate(lambda turn: base[turn["id"]]),
                "usd_per_1000_sessions": 1000.0 * sum(rows[t["id"]]["cost"] for t in turns) / len(subset),
                "prompt_tokens_mean": float(np.mean([rows[t["id"]]["prompt_tokens"] or 0 for t in turns])),
                "hybrid": {},
            }
            for rate in (0.1, 0.25, 0.4, 0.6):
                chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
                entry["hybrid"][str(rate)] = evaluate(lambda turn: table[turn["id"]] if turn["id"] in chosen else base[turn["id"]])
            out[name][f"{k}-shot"] = entry
            best = max(entry["hybrid"].items(), key=lambda item: item[1]["session_success"])
            print(
                f"{name:14s} {k}-shot alone session {100 * entry['alone']['session_success']:.1f} intent {100 * entry['alone']['node_accuracy']['intent']:.1f} scope {100 * entry['alone']['node_accuracy']['scope']:.1f} "
                f"(zero-shot {100 * entry['zero_shot_alone_same_sessions']['session_success']:.1f}, encoder {100 * entry['encoder_alone_same_sessions']['session_success']:.1f}) "
                f"best hybrid {100 * best[1]['session_success']:.1f}@{best[0]} ${entry['usd_per_1000_sessions']:.2f}/1000",
                flush=True,
            )
    return out


def main() -> None:
    import httpx

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    rules["models"] = rules_v4["models"]
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT  # v4's collectors write beside this module's files
    loaded, sessions, oos_dev = _sources(rules)
    labels = loaded["labels"]
    flat = [turn for session in sessions for turn in session]
    print(f"sessions {len(sessions)} turns {len(flat)} in scope {sum(t['in_scope'] for t in flat)} repeat asked {sum(t['repeat'] is not None for t in flat)}", flush=True)
    spec = specialists(rules, loaded, sessions, oos_dev, device="cuda" if stage == "decoder" else "cpu")
    for name, block in spec.items():
        result = v1.evaluate(sessions, v4.decide(sessions, lambda turn: block["answers"][turn["id"]]), None)
        print(f"{name:14s} alone session {100 * result['session_success']:.1f} turn {100 * result['turn_success']:.1f} " + " ".join(f"{k} {100 * v:.1f}" for k, v in result["node_accuracy"].items()), flush=True)
    budget = api.Budget(api_rules, run_cap=rules["run_cap_usd"])
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"]}
    summary["specialist_rules"] = {name: block["rule"] for name, block in spec.items()}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=45.0) as client:
        answers = v4.part_a(rules, labels, sessions, client, budget, list(rules["models"]))
        summary["part_a"] = v4.score_a(rules, sessions, answers, spec)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        if stage in ("all", "b"):
            summary["part_b"] = v4.part_b(rules, labels, sessions, answers, spec, client, budget)
            path.write_text(json.dumps(summary, indent=2) + "\n")
        if stage in ("all", "d"):
            summary["part_d"] = part_d(rules, rules_v4, loaded, sessions, oos_dev, answers, spec, client, budget)
        summary["spend_usd"] = {"project_total": budget.total}
        path.write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"WORKFLOW_MASSIVE_DONE {stage} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    main()

"""Scripted-orchestrator workflow v1. Frozen rules are in workflow_protocol.json. Processor and Jev API only."""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol

WORKFLOW_PATH = ROOT / "workflow_protocol.json"
# Hosted token-probability workers: workflow name -> api_protocol.json system.
HOSTED = {"deepseek": "deepseek-v4.1-flash", "qwen37": "qwen3.7-flash"}
OUT = ROOT / "artifacts" / "workflow-v1"


def _sources():
    from datasets import load_dataset

    from bakeoff.full_banking77 import PROTOCOL_PATH, load_full_test
    from bakeoff.suite import load_suite_protocol

    protocol = load_protocol()
    test, labels = load_full_test(protocol, json.loads(PROTOCOL_PATH.read_text()))
    dev = load_items(ROOT / "artifacts" / "splits.json", protocol)["items"]["banking77"]["dev"]
    clinc = load_suite_protocol()["datasets"]["clinc150"]
    raw = load_dataset(clinc["hf_id"], clinc["config"], revision=clinc["revision"])
    oos = raw["test"].features["intent"].names.index("oos")
    oos_test = [example["text"] for example in raw["test"] if example["intent"] == oos]
    oos_dev = [example["text"] for example in raw["validation"] if example["intent"] == oos]
    return labels, test, dev, oos_test, oos_dev


def build_sessions(spec: dict, test, oos_test: list[str]) -> list[list[dict]]:
    rng = np.random.default_rng(spec["seed"])
    by_label: dict[int, list] = {}
    for item in test:
        by_label.setdefault(item.gold_index, []).append(item)
    for items in by_label.values():
        rng.shuffle(items)
    oos_pool = list(oos_test)
    rng.shuffle(oos_pool)
    sessions = []
    for session_index in range(spec["sessions"]["n"]):
        turns = []
        last_intent = None
        for turn_index in range(spec["sessions"]["turns"]):
            if rng.random() < 0.2:
                turns.append({"id": f"s{session_index}-t{turn_index}", "text": oos_pool.pop(), "in_scope": False, "intent": None, "repeat": None})
                continue
            available = [label for label, items in by_label.items() if items]
            if last_intent is not None and by_label[last_intent] and rng.random() < 0.3:
                label = last_intent
            else:
                label = int(rng.choice([value for value in available if value != last_intent]))
            item = by_label[label].pop()
            turns.append(
                {
                    "id": f"s{session_index}-t{turn_index}",
                    "text": item.text,
                    "in_scope": True,
                    "intent": label,
                    "repeat": None if last_intent is None else bool(label == last_intent),
                }
            )
            last_intent = label
        sessions.append(turns)
    return sessions


def _history(turns: list[dict], index: int) -> str:
    if index == 0:
        return f"Current message: {turns[index]['text']}"
    earlier = "\n".join(f"{position}. {turn['text']}" for position, turn in enumerate(turns[:index], start=1))
    return f"Previous messages in this conversation:\n{earlier}\nCurrent message: {turns[index]['text']}"


# ---------------------------------------------------------------- Jev


def _jev_post(client, body: dict) -> dict:
    waits = [1, 2, 4, 8, 16]
    for attempt in range(len(waits) + 1):
        try:
            response = client.post("https://api.typesafe.ai/v1/systemone", json=body)
        except Exception:
            if attempt == len(waits):
                raise
            time.sleep(waits[attempt] + random.random())
            continue
        if response.status_code in (429, 499, 500, 502, 503, 504, 529) and attempt < len(waits):
            time.sleep(waits[attempt] + random.random())
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()
    raise RuntimeError("retries exhausted")


def run_jev(spec: dict, sessions, labels: list[str], with_history: bool) -> dict:
    import httpx

    from bakeoff import jev

    name = "jev_raw_history" if with_history else "jev_typed_state"
    path = OUT / f"{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row
    nodes = spec["nodes"]
    jobs = []
    for turns in sessions:
        for index, turn in enumerate(turns):
            if turn["id"] in done:
                continue
            questions = {
                "scope": {"type": "noul", "instructions": nodes["scope"]["question"]},
                "intent": {"type": "choice", "instructions": spec["jev_request"]["intent_instructions"], "criteria": {label: None for label in labels}},
            }
            if with_history:
                questions["repeat"] = {"type": "noul", "instructions": nodes["repeat"]["question"]}
            state = _history(turns, index) if with_history else turn["text"]
            jobs.append((turn["id"], {"model": spec["jev_request"]["model"], "state": state, "questions": questions}))
    key = jev._api_key(json.loads(jev.JEV_PROTOCOL_PATH.read_text()))
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=90.0) as client, path.open("a", encoding="utf-8") as handle:

        def work(job):
            turn_id, body = job
            payload = _jev_post(client, body)
            answers = payload["answers"]
            probs = [float(answers["intent"]["probabilities"][label]) for label in labels]
            return {
                "id": turn_id,
                "scope": float(answers["scope"]["noul"]),
                "intent_probs": probs,
                "repeat": float(answers["repeat"]["noul"]) if "repeat" in answers else None,
                "input_tokens": (payload.get("usage") or {}).get("input_tokens", 0),
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, row in enumerate(pool.map(work, jobs), start=1):
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done[row["id"]] = row
                if count % 250 == 0:
                    print(f"  {name} {count}/{len(jobs)}", flush=True)
    return done


# ---------------------------------------------------------------- local models


def _local_scores(kind: str, texts: list[str], labels: list[str], cache_name: str) -> np.ndarray:
    import torch

    cache = OUT / f"{cache_name}.npz"
    if cache.exists():
        return np.load(cache)["scores"]
    if kind in HOSTED:
        import httpx

        from bakeoff import api_baseline as api
        from bakeoff.data import Item

        rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
        budget = api.Budget(rules, run_cap=1.5)
        items = [Item(item_id=f"{cache_name}-{index}", row=index, text=text, gold_index=0, gold_label="") for index, text in enumerate(texts)]
        with httpx.Client(headers={"Authorization": f"Bearer {api._key(rules)}"}, timeout=90.0) as client:
            rows = api.collect(client, rules, rules["systems"][HOSTED[kind]], items, labels, OUT / f"{cache_name}.jsonl", budget)
        budget.save()
        if any(row.get("error") for row in rows):
            raise RuntimeError(f"{kind}: {sum(bool(row.get('error')) for row in rows)} hosted rows failed")
        scores, failures = api.scores_from(rows, len(labels))
        print(f"  {kind} {cache_name}: {failures} run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)
        np.savez_compressed(cache, scores=scores)
        return scores
    torch.set_num_threads(10)
    if kind == "encoder":
        from transformers import AutoModel, AutoTokenizer

        from bakeoff.train_ettin import EncoderClassifier, _encode

        artifact = ROOT / "artifacts" / "program2" / "banking77-log_loss"
        encoder_spec = json.loads((ROOT / "program2.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(artifact / "encoder")
        encoder = AutoModel.from_pretrained(artifact / "encoder", dtype=torch.float32)
        model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels))
        model.head.load_state_dict(torch.load(artifact / "head.pt", map_location="cpu", weights_only=True)["head"])
        model.eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                encoded = _encode(tokenizer, texts[start : start + 32], encoder_spec["max_length"], torch.device("cpu"))
                rows.append(model(encoded["input_ids"], encoded["attention_mask"]).numpy().astype(np.float64))
        scores = np.concatenate(rows, axis=0)
    elif kind == "decoder":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bakeoff.data import Item
        from bakeoff.train_ettin_decoder import _score_items

        artifact = ROOT / "artifacts" / "program2-decoder" / "banking77-log_loss"
        saved = json.loads((artifact / "metrics.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(artifact / "model")
        model = AutoModelForCausalLM.from_pretrained(artifact / "model", dtype=torch.float32)
        model.config.use_cache = False
        model.to("cuda")
        items = [Item(item_id=str(index), row=index, text=text, gold_index=0, gold_label="") for index, text in enumerate(texts)]
        scores, _, _ = _score_items(model, tokenizer, items, labels, json.loads((ROOT / "program2_decoder.json").read_text()), torch.device("cuda"), saved["batch_size"], saved["label_chunk_size"])
        del model
        torch.cuda.empty_cache()
    elif kind == "qwen4b":
        from bakeoff.readout import READOUT_PATH, IndexReadout
        from bakeoff.suite import load_suite_protocol

        system_spec = load_suite_protocol()["systems"]["qwen3-4b-index_prior"]
        readout_spec = json.loads(READOUT_PATH.read_text())
        readout_spec["model"] = {"hf_id": system_spec["hf_id"], "revision": system_spec["revision"]}
        readout = IndexReadout(readout_spec, prefix_batch=system_spec["prefix_batch"])
        prior = np.load(ROOT / "artifacts" / "suite" / "qwen3-4b-index_prior" / "banking77_full__dev.npz")["index"].mean(axis=0)
        rows = []
        for index, text in enumerate(texts, start=1):
            rows.append(readout.score(text, labels)[0] - prior)
            if index % 250 == 0:
                print(f"  qwen4b {cache_name} {index}/{len(texts)}", flush=True)
        scores = np.stack(rows, axis=0)
        readout.close()
    else:
        from bakeoff.systems import GLiClassChoice

        system = GLiClassChoice(load_protocol()["systems"]["gliclass-base-v3.0"], device="cpu")
        system.load()
        rows = []
        for index, text in enumerate(texts, start=1):
            rows.append(system.score(text, labels)[0])
            if index % 500 == 0:
                print(f"  gliclass {cache_name} {index}/{len(texts)}", flush=True)
        scores = np.stack(rows, axis=0)
    np.savez_compressed(cache, scores=scores)
    return scores


def _temperature(kind: str) -> float:
    if kind in HOSTED:
        return float(json.loads((ROOT / "artifacts" / "api" / HOSTED[kind] / "summary.json").read_text())["datasets"]["banking77_full"]["temperature"]["temperature"])
    if kind == "encoder":
        return float(json.loads((ROOT / "artifacts" / "program2" / "banking77-log_loss" / "metrics.json").read_text())["temperature"]["temperature"])
    if kind == "decoder":
        return float(json.loads((ROOT / "artifacts" / "program2-decoder" / "banking77-log_loss" / "metrics.json").read_text())["temperature"]["temperature"])
    if kind == "qwen4b":
        return float(json.loads((ROOT / "artifacts" / "suite" / "qwen3-4b-index_prior" / "summary.json").read_text())["datasets"]["banking77_full"]["temperature"]["temperature"])
    from bakeoff.temperature import fit_temperature

    rows = [json.loads(line) for line in (ROOT / "artifacts" / "decisions" / "gliclass-base-v3.0__banking77__dev.jsonl").open(encoding="utf-8")]
    return fit_temperature(np.asarray([row["scores"] for row in rows], dtype=np.float64), np.asarray([row["gold_index"] for row in rows], dtype=np.int64))["temperature"]


def _scope_threshold(in_scope: np.ndarray, out_scope: np.ndarray) -> dict:
    best = None
    for tau in np.unique(np.concatenate([in_scope, out_scope])):
        balanced = 0.5 * ((in_scope >= tau).mean() + (out_scope < tau).mean())
        if best is None or balanced > best[1]:
            best = (float(tau), float(balanced))
    return {"tau": best[0], "dev_balanced_accuracy": best[1]}


def run_local(kind: str, sessions, labels, dev, oos_dev) -> tuple[dict, dict]:
    temperature = _temperature(kind)
    turns = [turn for session in sessions for turn in session]
    probs = apply_temperature(_local_scores(kind, [turn["text"] for turn in turns], labels, f"{kind}__session_turns"), temperature)
    dev_in = apply_temperature(_local_scores(kind, [item.text for item in dev], labels, f"{kind}__dev_in_scope"), temperature).max(axis=1)
    dev_out = apply_temperature(_local_scores(kind, oos_dev, labels, f"{kind}__dev_out_of_scope"), temperature).max(axis=1)
    rule = _scope_threshold(dev_in, dev_out)
    rule["escalation_tau"] = float(np.quantile(dev_in, 0.25))
    rule["temperature"] = temperature
    answers = {turn["id"]: {"top": float(row.max()), "intent": int(row.argmax())} for turn, row in zip(turns, probs)}
    return answers, rule


# ---------------------------------------------------------------- scoring


def decide(policy: str, sessions, jev_raw, jev_typed, local, rules) -> list[list[dict]]:
    """Per turn: the policy's scope, intent and repeat decisions, and whether Jev was called."""
    out = []
    for turns in sessions:
        rows = []
        previous_intent = None
        for turn in turns:
            calls = 0
            escalated = False
            if policy == "jev_raw_history":
                answer = jev_raw[turn["id"]]
                scope = answer["scope"] >= 0.5
                intent = int(np.argmax(answer["intent_probs"]))
                repeat = answer["repeat"] >= 0.5
                calls = 1
            else:
                if policy == "jev_typed_state":
                    answer = jev_typed[turn["id"]]
                    scope, intent, calls = answer["scope"] >= 0.5, int(np.argmax(answer["intent_probs"])), 1
                else:
                    kind, _, target = policy.partition("_then_")
                    kind = kind.split("_")[0]
                    answer = local[kind][turn["id"]]
                    scope, intent = answer["top"] >= rules[kind]["tau"], answer["intent"]
                    if target and answer["top"] < rules[kind]["escalation_tau"]:
                        escalated = True
                        if target == "jev":
                            fallback = jev_typed[turn["id"]]
                            scope, intent, calls = fallback["scope"] >= 0.5, int(np.argmax(fallback["intent_probs"])), 1
                        else:
                            # Escalation to another typed-state worker: its own answer and its own scope threshold.
                            fallback = local[target][turn["id"]]
                            scope, intent = fallback["top"] >= rules[target]["tau"], fallback["intent"]
                repeat = previous_intent is not None and intent == previous_intent
            rows.append({"scope": bool(scope), "intent": intent, "repeat": bool(repeat), "jev_calls": calls, "escalated": escalated, "had_previous": previous_intent is not None})
            if scope:
                previous_intent = intent
        out.append(rows)
    return out


def evaluate(sessions, decisions, tokens: dict | None) -> dict:
    node = {"scope": [], "intent": [], "repeat": []}
    repeat_by_dependency = {"both_intents_right": [], "an_intent_wrong": []}
    turn_ok = []
    session_ok = []
    calls = 0
    escalated = 0
    token_total = 0
    for turns, rows in zip(sessions, decisions):
        all_ok = True
        previous_right = None
        for turn, row in zip(turns, rows):
            ok = row["scope"] == turn["in_scope"]
            node["scope"].append(ok)
            intent_right = None
            if turn["in_scope"]:
                intent_right = row["scope"] and row["intent"] == turn["intent"]
                node["intent"].append(bool(intent_right))
                ok = ok and bool(intent_right)
                if turn["repeat"] is not None:
                    repeat_right = row["scope"] and row["repeat"] == turn["repeat"]
                    node["repeat"].append(bool(repeat_right))
                    ok = ok and bool(repeat_right)
                    if previous_right is not None:
                        key = "both_intents_right" if (intent_right and previous_right) else "an_intent_wrong"
                        repeat_by_dependency[key].append(bool(repeat_right))
                previous_right = bool(intent_right)
            turn_ok.append(ok)
            all_ok = all_ok and ok
            calls += row["jev_calls"]
            escalated += int(row["escalated"])
            if tokens and row["jev_calls"]:
                token_total += tokens.get(turn["id"], 0)
        session_ok.append(all_ok)
    rng = np.random.default_rng(20261009)
    per_session_turns = np.asarray(turn_ok, dtype=np.float64).reshape(len(sessions), -1).mean(axis=1)
    flags = np.asarray(session_ok, dtype=np.float64)

    def interval(values):
        means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(2000)]
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    return {
        "node_accuracy": {name: float(np.mean(values)) for name, values in node.items()},
        "node_n": {name: len(values) for name, values in node.items()},
        "repeat_accuracy_by_dependency": {name: (float(np.mean(values)) if values else None) for name, values in repeat_by_dependency.items()},
        "repeat_n_by_dependency": {name: len(values) for name, values in repeat_by_dependency.items()},
        "turn_success": float(np.mean(turn_ok)),
        "turn_success_ci": interval(per_session_turns),
        "session_success": float(flags.mean()),
        "session_success_ci": interval(flags),
        "jev_requests_per_session": calls / len(sessions),
        "jev_input_tokens_per_session": token_total / len(sessions),
        "share_of_turns_escalated": escalated / len(turn_ok),
    }


def main() -> None:
    spec = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    labels, test, dev, oos_test, oos_dev = _sources()
    sessions = build_sessions(spec, test, oos_test)
    (OUT / "sessions.json").write_text(json.dumps(sessions, indent=1) + "\n", encoding="utf-8")
    flat = [turn for session in sessions for turn in session]
    print(
        f"sessions {len(sessions)} turns {len(flat)} in-scope {sum(t['in_scope'] for t in flat)} "
        f"repeat asked {sum(t['repeat'] is not None for t in flat)} repeat true {sum(bool(t['repeat']) for t in flat)}",
        flush=True,
    )
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        return
    jev_typed = run_jev(spec, sessions, labels, with_history=False)
    jev_raw = run_jev(spec, sessions, labels, with_history=True)
    local, rules = {}, {}
    kinds = {policy.split("_")[0] for policy in spec["policies"]} | {policy.partition("_then_")[2] for policy in spec["policies"]}
    for kind in sorted(kinds - {"jev", ""}):
        local[kind], rules[kind] = run_local(kind, sessions, labels, dev, oos_dev)
    summary = {"protocol_id": spec["protocol_id"], "rules": rules, "policies": {}}
    for policy in spec["policies"]:
        tokens = {key: row["input_tokens"] for key, row in (jev_raw if policy == "jev_raw_history" else jev_typed).items()}
        summary["policies"][policy] = evaluate(sessions, decide(policy, sessions, jev_raw, jev_typed, local, rules), tokens)
        row = summary["policies"][policy]
        print(
            f"{policy:22s} scope {row['node_accuracy']['scope']:.3f} intent {row['node_accuracy']['intent']:.3f} repeat {row['node_accuracy']['repeat']:.3f} "
            f"turn {row['turn_success']:.3f} session {row['session_success']:.3f} jev/session {row['jev_requests_per_session']:.2f} escalated {row['share_of_turns_escalated']:.3f}",
            flush=True,
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("WORKFLOW_DONE", flush=True)


if __name__ == "__main__":
    main()

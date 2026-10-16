"""Parts 2 and 3 of robustness_protocol.json: a second hosted model with retrieval, and a repeat of the Gemma run."""

from __future__ import annotations

import json

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.protocol import ROOT

RULES_PATH = ROOT / "robustness_protocol.json"
OUT = ROOT / "artifacts" / "robustness"
SECOND = "qwen3.5-27b"


def _brief(result: dict) -> dict:
    return {key: result[key] for key in ("node_accuracy", "turn_success", "session_success", "session_success_ci")}


def synthetic(client, budget, rules_v4) -> dict:
    from bakeoff import workflow_retrieval as wr

    spec = json.loads(wr.RULES_PATH.read_text(encoding="utf-8"))
    out = {}
    for name in ("banking77", "massive", "clinc150"):
        domain = wr._domain(name)
        rules, labels, sessions = domain["rules"], domain["labels"], domain["sessions"]
        turns = [turn for session in sessions for turn in session]
        system = v4._system_prompt(rules, labels, "current_only", False)
        pool = wr._pools(domain, spec["seed"])["full_train"]
        index = wr.BM25([text for text, _ in pool])
        jobs = []
        for turn in turns:
            picked = index.top(turn["text"], spec["retrieval"]["k"])[::-1]
            lines = "\n".join(f"{pool[i][0]} => {pool[i][1]}" for i in picked)
            jobs.append((turn["id"], system, f"Similar labelled examples (message => code):\n{lines}\n\nCurrent message: {turn['text']}"))
        evaluate = lambda answer_of: _brief(v1.evaluate(sessions, v4.decide(sessions, answer_of), None))
        block = {}
        for model, suffix in ((SECOND, ""), ("gemma-4-31b", "__repeat")):
            print(f"  {name} {model}{suffix}", flush=True)
            rows = v4._collect(client, rules_v4["models"][model], jobs, rules["request"]["max_tokens"]["answer"], OUT / f"{name}__{model}__full_train{suffix}.jsonl", budget)
            table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
            entry = {"alone": evaluate(lambda turn: table[turn["id"]]), "usd_per_1000_sessions": 1000.0 * sum(row["cost"] for row in rows.values()) / len(sessions)}
            if suffix:
                original = {key: v4.parse(row["text"], len(labels)) for key, row in v4._read(wr.OUT / f"{name}__gemma-4-31b__full_train.jsonl").items()}
                entry["share_of_turns_changed"] = float(np.mean([(table[t["id"]]["scope"], table[t["id"]]["intent"]) != (original[t["id"]]["scope"], original[t["id"]]["intent"]) for t in turns]))
                entry["original_alone"] = evaluate(lambda turn: original[turn["id"]])
            block[f"{model}{suffix}"] = entry
            print(
                f"{name:10s} {model + suffix:22s} alone {100 * entry['alone']['session_success']:.1f} [{100 * entry['alone']['session_success_ci'][0]:.1f}, {100 * entry['alone']['session_success_ci'][1]:.1f}] intent {100 * entry['alone']['node_accuracy']['intent']:.1f} scope {100 * entry['alone']['node_accuracy']['scope']:.1f} "
                f"${entry['usd_per_1000_sessions']:.2f}/1000" + (f" | original {100 * entry['original_alone']['session_success']:.1f}, turns changed {100 * entry['share_of_turns_changed']:.1f}%" if suffix else ""),
                flush=True,
            )
        out[name] = block
    return out


def near_oos(client, budget, rules_v4) -> dict:
    from bakeoff import workflow_near_oos as wn
    from bakeoff.workflow_retrieval import BM25

    spec = json.loads(wn.RULES_PATH.read_text(encoding="utf-8"))
    data = wn._data()
    sessions = wn._sessions(data)
    labels = data["labels"]
    codes = api.codes_for(len(labels))
    system = spec["hosted"]["system_prompt"].format(options="\n".join(f"{code}: {label}" for code, label in zip(codes, labels)))
    index = BM25([text for text, _ in data["train"]])
    out = {}
    for kind, group in sessions.items():
        turns = [turn for session in group for turn in session]
        for condition in ("zero_shot", "retrieval"):
            jobs = []
            for turn in turns:
                user = f"Current message: {turn['text']}"
                if condition == "retrieval":
                    picked = index.top(turn["text"], 20)[::-1]
                    lines = "\n".join(f"{data['train'][i][0]} => {codes[data['train'][i][1]]}" for i in picked)
                    user = f"Similar labelled examples (message => code):\n{lines}\n\n{user}"
                jobs.append((turn["id"], system, user))
            print(f"  banking77_oos {kind} {condition} {SECOND}", flush=True)
            rows = v4._collect(client, rules_v4["models"][SECOND], jobs, 24, OUT / f"b77oos__{SECOND}__{kind}__{condition}.jsonl", budget)
            table = {key: v4.parse(row["text"], len(labels)) for key, row in rows.items()}
            result = v1.evaluate(group, v4.decide(group, lambda turn: table[turn["id"]]), None)
            caught = float(np.mean([not table[t["id"]]["scope"] for t in turns if not t["in_scope"]]))
            out[f"{kind}__{condition}"] = {**_brief(result), "out_of_scope_caught": caught}
            print(f"banking77_oos {kind:3s} {condition:9s} {SECOND} alone {100 * result['session_success']:.1f} scope {100 * result['node_accuracy']['scope']:.1f} caught {100 * caught:.1f} intent {100 * result['node_accuracy']['intent']:.1f}", flush=True)
    return out


def multidogo(client, budget, rules_v4) -> dict:
    from bakeoff import workflow_multidogo as wm
    from bakeoff.workflow_retrieval import BM25

    spec = json.loads(wm.RULES_PATH.read_text(encoding="utf-8"))
    test = wm.load("test")
    train = [turn for conversation in wm.load("train") for turn in conversation]
    prompt = spec["prompt"]
    codes = api.codes_for(len(wm.LABELS))
    options = "\n".join(f"{code}: {wm.NAMES[label]}" for code, label in zip(codes, wm.LABELS))
    system = prompt["system"].format(options=options, history_clause="", repeat_line="")
    index = BM25([turn["text"] for turn in train])
    jobs = []
    for conversation in test:
        for turn in conversation:
            picked = index.top(turn["text"], 20)[::-1]
            lines = "\n".join(f"{train[j]['text']} => {codes[train[j]['label']]}" for j in picked)
            jobs.append((turn["id"], system, f"Similar labelled examples (message => code):\n{lines}\n\n" + prompt["user_current"].format(text=turn["text"])))
    print(f"  multidogo {SECOND} retrieval", flush=True)
    rows = v4._collect(client, rules_v4["models"][SECOND], jobs, prompt["max_tokens"], OUT / f"multidogo__{SECOND}__retrieval_current.jsonl", budget)
    table = {key: wm._parse(row["text"]) for key, row in rows.items()}
    return {"retrieval_current": score_multidogo(test, table)}


def score_multidogo(test, table) -> dict:
    from bakeoff import workflow_multidogo as wm

    labels_right, conv_ok = [], []
    for conversation in test:
        gold_repeat = wm._repeat_gold(conversation)
        last, ok = None, True
        for turn, wanted in zip(conversation, gold_repeat):
            answer = table[turn["id"]]
            right = answer["label"] == turn["label"]
            labels_right.append(right)
            typed = last is not None and answer["label"] == last
            if 0 <= answer["label"] < len(wm.TASKS):
                last = answer["label"]
            if wanted is not None:
                right = right and typed == wanted
            ok = ok and right
        conv_ok.append(ok)
    flags = np.asarray(conv_ok, dtype=np.float64)
    rng = np.random.default_rng(20261014)
    draws = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(2000)]
    return {"label_accuracy": float(np.mean(labels_right)), "conversation_success": float(flags.mean()), "conversation_success_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def main() -> None:
    import httpx

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    v4.OUT = OUT
    budget = api.Budget(api_rules, run_cap=rules["run_cap_usd"])
    summary = {"protocol_id": rules["protocol_id"]}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        summary["synthetic"] = synthetic(client, budget, rules_v4)
        (OUT / "summary_api.json").write_text(json.dumps(summary, indent=2) + "\n")
        summary["banking77_oos"] = near_oos(client, budget, rules_v4)
        (OUT / "summary_api.json").write_text(json.dumps(summary, indent=2) + "\n")
        summary["multidogo"] = multidogo(client, budget, rules_v4)
        row = summary["multidogo"]["retrieval_current"]
        print(f"multidogo {SECOND} retrieval label {100 * row['label_accuracy']:.1f} conversations {100 * row['conversation_success']:.1f} [{100 * row['conversation_success_ci'][0]:.1f}, {100 * row['conversation_success_ci'][1]:.1f}]", flush=True)
    summary["spend_usd"] = {"project_total": budget.total}
    (OUT / "summary_api.json").write_text(json.dumps(summary, indent=2) + "\n")
    budget.save()
    print(f"ROBUSTNESS_API_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    main()

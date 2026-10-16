"""Context handoff: one model compacts a conversation, another answers from the notes. Rules: handoff_protocol.json."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow as v1
from bakeoff import workflow_v4 as v4
from bakeoff.protocol import ROOT, load_protocol

RULES_PATH = ROOT / "handoff_protocol.json"
OUT = ROOT / "artifacts" / "handoff"
SETS = {"10_turns": (2, 100), "20_turns": (4, 100), "40_turns": (8, 50)}
LETTERS = "ABCD"


def conversations(sessions, group: int, limit: int) -> list[list[dict]]:
    out = [[turn for session in sessions[start : start + group] for turn in session] for start in range(0, len(sessions), group)]
    return out[:limit]


def build_questions(conversation, labels, seed: int, index: int) -> list[dict]:
    rng = np.random.default_rng([seed, index])
    intents = [turn["intent"] for turn in conversation if turn["in_scope"]]
    counts = Counter(intents)
    distinct = list(dict.fromkeys(intents))
    unraised = [label for label in range(len(labels)) if label not in counts]
    questions = []

    def take(pool, k):
        pool = list(pool)
        return [pool[i] for i in rng.permutation(len(pool))[:k]]

    def choice(kind, text, gold, others):
        options = [gold] + others
        order = rng.permutation(len(options))
        questions.append({"type": kind, "text": text, "options": [labels[options[i]] for i in order], "gold": LETTERS[list(order).index(0)]})

    def padded(gold, raised_pool):
        others = take([i for i in raised_pool if i != gold], 3)
        return others + take(unraised, 3 - len(others))

    if distinct:
        choice("first", "Which of these was the first banking issue the customer raised?", distinct[0], padded(distinct[0], distinct))
        choice("last", "Which of these was the most recent banking issue the customer raised?", intents[-1], padded(intents[-1], distinct))
        for _ in range(2):
            choice("present", "Which of these issues did the customer raise?", take(distinct, 1)[0], take(unraised, 3))
    if len(distinct) >= 3:
        for _ in range(2):
            choice("absent", "Which of these issues did the customer NOT raise?", take(unraised, 1)[0], take(distinct, 3))
    if len(distinct) >= 2:
        for _ in range(2):
            pair = take(distinct, 2)
            earlier = min(pair, key=intents.index)
            choice("order", "Which of these two issues did the customer raise first?", earlier, [i for i in pair if i != earlier])
    repeats = [label for label in distinct if counts[label] >= 2]
    if repeats:
        gold = take(repeats, 1)[0]
        choice("repeated", "Which of these issues did the customer raise more than once?", gold, padded(gold, [i for i in distinct if counts[i] == 1]))
    questions.append({"type": "count_other", "text": "How many of the customer's messages were not banking requests? Answer with a number.", "options": [], "gold": str(sum(not turn["in_scope"] for turn in conversation))})
    return questions


def _messages(conversation) -> str:
    return "\n".join(f"{i}. {turn['text']}" for i, turn in enumerate(conversation, start=1))


def _question_block(questions) -> str:
    lines = []
    for number, question in enumerate(questions, start=1):
        lines.append(f"{number}. {question['text']}")
        lines += [f"   {LETTERS[i]}. {option}" for i, option in enumerate(question["options"])]
    return "\n".join(lines)


def _reader_user(source_kind: str, source_text: str, questions) -> str:
    head = "Handoff notes:" if source_kind == "notes" else "Customer messages:"
    return f"{head}\n{source_text}\n\nQuestions:\n{_question_block(questions)}"


def _cut(notes: str, budget: int) -> tuple[str, bool]:
    words = notes.split()
    return " ".join(words[:budget]), len(words) > budget


def _score(text: str, questions) -> list[dict]:
    found = {int(m.group(1)): m.group(2).upper() for m in re.finditer(r"^\s*(\d+)\s*[:.)]\s*\(?([A-Da-d]|\d+)\b", text, re.M)}
    rows = []
    for number, question in enumerate(questions, start=1):
        answer = found.get(number)
        rows.append({"type": question["type"], "right": answer == question["gold"], "parsed": answer is not None, "answer": answer, "gold": question["gold"]})
    return rows


def _cell(per_conversation: list[list[dict]]) -> dict:
    """Accuracy over the choice questions with an interval over conversations, and per type."""
    choice = [[row for row in rows if row["type"] != "count_other"] for rows in per_conversation]
    right = np.asarray([sum(row["right"] for row in rows) for rows in choice], dtype=np.float64)
    total = np.asarray([len(rows) for rows in choice], dtype=np.float64)
    rng = np.random.default_rng(20261012)
    draws = []
    for _ in range(1000):
        pick = rng.integers(0, len(right), len(right))
        draws.append(right[pick].sum() / total[pick].sum())
    by_type = {}
    for kind in ("first", "last", "present", "absent", "order", "repeated", "count_other"):
        flags = [row["right"] for rows in per_conversation for row in rows if row["type"] == kind]
        if flags:
            by_type[kind] = float(np.mean(flags))
    counts = [(int(row["answer"]), int(row["gold"])) for rows in per_conversation for row in rows if row["type"] == "count_other" and row["answer"] and row["answer"].isdigit()]
    return {
        "accuracy": float(right.sum() / total.sum()),
        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_questions": int(total.sum()),
        "by_type": by_type,
        "count_mean_abs_error": float(np.mean([abs(a - g) for a, g in counts])) if counts else None,
        "unparsed": int(sum(not row["parsed"] for rows in per_conversation for row in rows)),
    }


def _jev_read(rules, name: str, sources: dict[str, str], questions_of: dict[str, list[dict]]) -> dict[str, list[dict]]:
    import httpx

    from bakeoff import jev

    path = OUT / f"jev__{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row
    todo = [key for key in sources if key not in done]
    key = jev._api_key(json.loads(jev.JEV_PROTOCOL_PATH.read_text()))
    workflow_rules = json.loads(v1.WORKFLOW_PATH.read_text(encoding="utf-8"))
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=90.0) as client, path.open("a", encoding="utf-8") as handle:

        def work(conversation_id):
            asked = [(i, q) for i, q in enumerate(questions_of[conversation_id]) if q["type"] != "count_other"]
            body = {
                "model": workflow_rules["jev_request"]["model"],
                "state": sources[conversation_id],
                "questions": {f"q{i}": {"type": "choice", "instructions": q["text"], "criteria": {option: None for option in q["options"]}} for i, q in asked},
            }
            answers = v1._jev_post(client, body)["answers"]
            picks = {}
            for i, q in asked:
                probs = answers[f"q{i}"]["probabilities"]
                picks[str(i)] = LETTERS[int(np.argmax([float(probs[option]) for option in q["options"]]))]
            return {"id": conversation_id, "picks": picks}

        with ThreadPoolExecutor(max_workers=8) as pool:
            for row in pool.map(work, todo):
                handle.write(json.dumps(row) + "\n")
                done[row["id"]] = row
    out = {}
    for conversation_id in sources:
        rows = []
        for i, q in enumerate(questions_of[conversation_id]):
            if q["type"] == "count_other":
                continue
            answer = done[conversation_id]["picks"][str(i)]
            rows.append({"type": q["type"], "right": answer == q["gold"], "parsed": True, "answer": answer, "gold": q["gold"]})
        out[conversation_id] = rows
    return out


def main() -> None:
    import httpx

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SETS)
    labels = load_protocol()["datasets"]["banking77"]["labels"]
    sessions = json.loads((v1.OUT / "sessions.json").read_text(encoding="utf-8"))
    budget = api.Budget(api_rules, run_cap=rules["run_cap_usd"])
    models = rules["models"]
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"], "sets": {}}
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=90.0) as client:
        for set_name in only:
            group, limit = SETS[set_name]
            convs = conversations(sessions, group, limit)
            ids = [f"{set_name}-{i}" for i in range(len(convs))]
            questions_of = {cid: build_questions(conv, labels, rules["seed"], i) for i, (cid, conv) in enumerate(zip(ids, convs))}
            raw = {cid: _messages(conv) for cid, conv in zip(ids, convs)}
            (OUT / f"questions__{set_name}.json").write_text(json.dumps(questions_of, indent=1) + "\n", encoding="utf-8")
            block = {"n_conversations": len(convs), "raw_words_mean": float(np.mean([len(text.split()) for text in raw.values()])), "writers": {}, "raw": {}, "cells": {}}
            reader_system = lambda kind: rules["reader"]["system_prompt"].format(source=rules["reader"]["source_text"][kind])
            answer_tokens = lambda cid: 8 * len(questions_of[cid]) + 10

            def read(reader: str, kind: str, sources: dict[str, str], tag: str) -> dict:
                if reader == "jev":
                    head = "Handoff notes:" if kind == "notes" else "Customer messages:"
                    scored = _jev_read(rules, f"{set_name}__{tag}", {cid: f"{head}\n{text}" for cid, text in sources.items()}, questions_of)
                    return {**_cell([scored[cid] for cid in ids]), "usd_per_conversation": None}
                jobs = [(cid, reader_system(kind), _reader_user(kind, sources[cid], questions_of[cid])) for cid in ids]
                rows = v4._collect(client, rules_v4["models"][reader], jobs, max(answer_tokens(cid) for cid in ids), OUT / f"r__{set_name}__{reader}__{tag}.jsonl", budget)
                cell = _cell([_score(rows[cid]["text"], questions_of[cid]) for cid in ids])
                cell["usd_per_conversation"] = float(np.mean([rows[cid]["cost"] for cid in ids]))
                cell["prompt_tokens_mean"] = float(np.mean([rows[cid]["prompt_tokens"] or 0 for cid in ids]))
                return cell

            for reader in models + ["jev"]:
                block["raw"][reader] = read(reader, "raw", raw, "raw")
                print(f"{set_name} raw            -> {reader:20s} {100 * block['raw'][reader]['accuracy']:.1f}", flush=True)
            path.write_text(json.dumps({**summary, "sets": {**summary["sets"], set_name: block}}, indent=2) + "\n")
            for writer in models:
                for words in rules["note_budgets_words"]:
                    system = rules["writer"]["system_prompt"].format(budget=words)
                    jobs = [(cid, system, raw[cid]) for cid in ids]
                    rows = v4._collect(client, rules_v4["models"][writer], jobs, 3 * words + 40, OUT / f"w__{set_name}__{writer}__{words}.jsonl", budget)
                    cut = {cid: _cut(rows[cid]["text"].strip(), words) for cid in ids}
                    notes = {cid: value[0] for cid, value in cut.items()}
                    block["writers"][f"{writer}@{words}"] = {
                        "words_mean": float(np.mean([len(rows[cid]["text"].split()) for cid in ids])),
                        "share_cut": float(np.mean([value[1] for value in cut.values()])),
                        "usd_per_conversation": float(np.mean([rows[cid]["cost"] for cid in ids])),
                    }
                    line = []
                    for reader in models + ["jev"]:
                        cell = read(reader, "notes", notes, f"{writer}__{words}")
                        block["cells"][f"{writer}@{words}->{reader}"] = cell
                        line.append(f"{reader.split('-')[0]}{rules_v4['models'][reader]['params_b'] if reader != 'jev' else ''} {100 * cell['accuracy']:.1f}")
                    print(f"{set_name} {writer:18s} {words:3d}w ({block['writers'][f'{writer}@{words}']['words_mean']:.0f} written) -> " + "  ".join(line) + f" | run ${budget.run:.2f}", flush=True)
                    summary["sets"][set_name] = block
                    summary["spend_usd"] = {"project_total": budget.total}
                    path.write_text(json.dumps(summary, indent=2) + "\n")
            budget.save()
    print(f"HANDOFF_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


if __name__ == "__main__":
    main()

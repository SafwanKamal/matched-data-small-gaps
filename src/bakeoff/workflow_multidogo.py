"""The workflow on real conversations: MultiDoGO finance. Rules: workflow_multidogo_protocol.json.

Stages: 'hosted' (API), 'train' (GPU), 'report' (processor).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict

import numpy as np

from bakeoff import api_baseline as api
from bakeoff import workflow_v4 as v4
from bakeoff.data import Item
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "workflow_multidogo_protocol.json"
DATA = ROOT / "data" / "multidogo"
OUT = ROOT / "artifacts" / "workflow-multidogo"
SEED = 20261014
TASKS = ["checkbalance", "transfermoney", "reportlostcard", "replacecard", "disputecharge", "updateaddress", "closeaccount", "getroutingnumber", "checkoffereligibility", "orderchecks"]
DIALOGUE = ["contentonly", "confirmation", "rejection", "thankyou", "openinggreeting", "closinggreeting", "outofdomain"]
NAMES = {"checkbalance": "check balance", "transfermoney": "transfer money", "reportlostcard": "report lost card", "replacecard": "replace card", "disputecharge": "dispute charge", "updateaddress": "update address", "closeaccount": "close account", "getroutingnumber": "get routing number", "checkoffereligibility": "check offer eligibility", "orderchecks": "order checks", "contentonly": "content only", "confirmation": "confirmation", "rejection": "rejection", "thankyou": "thank you", "openinggreeting": "opening greeting", "closinggreeting": "closing greeting", "outofdomain": "out of domain"}
LABELS = TASKS + DIALOGUE


def load(split: str) -> list[list[dict]]:
    """Conversations of customer turns in order: {'id', 'text', 'label' (index), 'joined' (had two intents)}."""
    conversations = defaultdict(list)
    for row in csv.DictReader((DATA / f"finance_{split}.tsv").open(encoding="utf-8"), delimiter="\t"):
        raw = row["intent"]
        first = raw.split("<div>")[0]
        label = LABELS.index(first) if first in LABELS else -1
        conversations[row["conversationId"]].append((float(row["turnNumber"]), {"id": f"{split}-{row['conversationId']}-{int(float(row['turnNumber']))}", "text": row["utterance"], "label": label, "joined": "<div>" in raw}))
    # A conversation containing a label that never occurs in training (open account: 1 dev and 3 test turns) is dropped.
    return [[turn for _, turn in sorted(turns, key=lambda pair: pair[0])] for turns in conversations.values() if all(turn["label"] >= 0 for _, turn in turns)]


def _repeat_gold(conversation) -> list:
    out, last = [], None
    for turn in conversation:
        if turn["label"] < len(TASKS):
            out.append(None if last is None else turn["label"] == last)
            last = turn["label"]
        else:
            out.append(None)
    return out


def _with_previous(conversation, index: int) -> str:
    previous = conversation[index - 1]["text"] if index else "(none)"
    return f"previous: {previous} current: {conversation[index]['text']}"


def hosted() -> None:
    import httpx

    from bakeoff.workflow_retrieval import BM25

    spec = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rules_v4 = json.loads(v4.RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    test = load("test")
    train = [turn for conversation in load("train") for turn in conversation]
    prompt = spec["prompt"]
    codes = api.codes_for(len(LABELS))
    options = "\n".join(f"{code}: {NAMES[label]}" for code, label in zip(codes, LABELS))
    systems = {
        False: prompt["system"].format(options=options, history_clause="", repeat_line=""),
        True: prompt["system"].format(options=options, history_clause=prompt["history_clause"], repeat_line=prompt["repeat_line"]),
    }
    index = BM25([turn["text"] for turn in train])
    runs = {
        "gemma_zero_current": ("gemma-4-31b", False, False),
        "gemma_zero_history": ("gemma-4-31b", False, True),
        "gemma_retrieval_current": ("gemma-4-31b", True, False),
        "gemma_retrieval_history": ("gemma-4-31b", True, True),
        "qwen9b_zero_current": ("qwen3.5-9b", False, False),
        "qwen9b_retrieval_current": ("qwen3.5-9b", True, False),
    }
    v4.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    budget = api.Budget(api_rules, run_cap=spec["run_cap_usd"])
    with httpx.Client(headers={"Authorization": f"Bearer {api._key(api_rules)}"}, timeout=60.0) as client:
        for name, (model, retrieve, history) in runs.items():
            jobs = []
            for conversation in test:
                for i, turn in enumerate(conversation):
                    if history and i:
                        earlier = "\n".join(f"{n}. {t['text']}" for n, t in enumerate(conversation[:i], start=1))
                        user = prompt["user_history"].format(earlier=earlier, text=turn["text"])
                    else:
                        user = prompt["user_current"].format(text=turn["text"])
                    if retrieve:
                        picked = index.top(turn["text"], 20)[::-1]
                        lines = "\n".join(f"{train[j]['text']} => {codes[train[j]['label']]}" for j in picked)
                        user = f"Similar labelled examples (message => code):\n{lines}\n\n{user}"
                    jobs.append((turn["id"], systems[history], user))
            print(f"  {name} {len(jobs)} turns", flush=True)
            v4._collect(client, rules_v4["models"][model], jobs, prompt["max_tokens"], OUT / f"{name}.jsonl", budget)
    budget.save()
    print(f"MULTIDOGO_HOSTED_DONE run ${budget.run:.3f} project total ${budget.total:.3f}", flush=True)


def train() -> None:
    import torch

    from bakeoff import train_ettin as enc

    splits = {split: load(split) for split in ("train", "dev", "test")}
    for name, contextual in (("encoder", False), ("encoder_with_previous", True)):
        if (OUT / name / "metrics.json").exists():
            continue

        def rows(split):
            out = []
            for conversation in splits[split]:
                for i, turn in enumerate(conversation):
                    out.append((_with_previous(conversation, i) if contextual else turn["text"], turn["label"]))
            return out

        as_items = lambda split: [Item(item_id=str(i), row=i, text=t, gold_index=g, gold_label="") for i, (t, g) in enumerate(rows(split))]
        spec = json.loads((ROOT / "program2.json").read_text())
        spec.update({"learning_rate": 1e-4, "seed": SEED})
        stub = {"datasets": {"multidogo": {"labels": LABELS}}}
        materialized = {"splits": {"multidogo": {"dev": {"rows": []}}}, "items": {"multidogo": {"dev": as_items("dev"), "test": as_items("test")}}}
        torch.manual_seed(SEED)
        result = enc._train_one(name, "multidogo", "log_loss", spec, stub, materialized, torch.device("cuda"), rows=rows("train"), out_root=OUT)
        print(f"trained {name}: test turn accuracy {result['scaled']['accuracy']:.4f}", flush=True)
    print("MULTIDOGO_TRAIN_DONE", flush=True)


def _parse(text: str) -> dict:
    label = re.search(r"label\s*:\s*\D{0,3}(\d{1,2})", text, re.I)
    repeat = re.search(r"repeat\s*:\s*\W*(yes|no)", text, re.I)
    code = int(label.group(1)) if label else 0
    return {"label": code - 1 if 1 <= code <= len(LABELS) else -1, "repeat": bool(repeat and repeat.group(1).lower() == "yes"), "parsed": label is not None}


def _encoder_scores(name: str, splits: dict, contextual: bool) -> dict:
    """Logits on dev and test turns from a saved encoder, on the processor or GPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff.train_ettin import EncoderClassifier, _encode

    cache = OUT / f"{name}__scores.npz"
    if cache.exists():
        return dict(np.load(cache))
    folder = OUT / name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(folder / "encoder")
    encoder = AutoModel.from_pretrained(folder / "encoder", dtype=torch.float32)
    model = EncoderClassifier(encoder, encoder.config.hidden_size, len(LABELS))
    model.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True)["head"])
    model.to(device).eval()
    out = {}
    for split in ("dev", "test"):
        texts = [(_with_previous(c, i) if contextual else t["text"]) for c in splits[split] for i, t in enumerate(c)]
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                encoded = _encode(tokenizer, texts[start : start + 64], 128, device)
                chunks.append(model(encoded["input_ids"], encoded["attention_mask"]).float().cpu().numpy().astype(np.float64))
        out[split] = np.concatenate(chunks, axis=0)
    np.savez_compressed(cache, **out)
    return out


def report() -> None:
    splits = {split: load(split) for split in ("dev", "test")}
    test = splits["test"]
    turns = [turn for conversation in test for turn in conversation]
    gold_dev = np.asarray([t["label"] for c in splits["dev"] for t in c], dtype=np.int64)
    answers: dict[str, dict] = {}
    confidence = {}
    for name, contextual in (("encoder", False), ("encoder_with_previous", True)):
        if not (OUT / name / "metrics.json").exists():
            continue
        scores = _encoder_scores(name, splits, contextual)
        temperature = fit_temperature(scores["dev"], gold_dev)["temperature"]
        probs = apply_temperature(scores["test"], temperature)
        answers[name] = {turn["id"]: {"label": int(row.argmax()), "repeat": None} for turn, row in zip(turns, probs)}
        confidence[name] = {turn["id"]: float(row.max()) for turn, row in zip(turns, probs)}
    cost = {}
    for path in sorted(OUT.glob("*.jsonl")):
        rows = v4._read(path)
        if len(rows) >= len(turns):
            answers[path.stem] = {key: _parse(row["text"]) for key, row in rows.items()}
            cost[path.stem] = 1000.0 * sum(row["cost"] for row in rows.values()) / len(test)
    if "encoder" in answers and "gemma_retrieval_current" in answers:
        retrieval_rows = v4._read(OUT / "gemma_retrieval_current.jsonl")
        order = sorted(turns, key=lambda turn: confidence["encoder"][turn["id"]])
        for rate in (0.25, 0.4):
            chosen = {turn["id"] for turn in order[: int(round(rate * len(turns)))]}
            answers[f"encoder_then_gemma_retrieval_{rate}"] = {turn["id"]: (answers["gemma_retrieval_current"][turn["id"]] if turn["id"] in chosen else answers["encoder"][turn["id"]]) for turn in turns}
            cost[f"encoder_then_gemma_retrieval_{rate}"] = 1000.0 * sum(retrieval_rows[i]["cost"] for i in chosen) / len(test)

    rng = np.random.default_rng(SEED)
    summary = {"protocol_id": json.loads(RULES_PATH.read_text(encoding="utf-8"))["protocol_id"], "joined_label_turns_test": sum(t["joined"] for t in turns), "systems": {}}
    for name, table in answers.items():
        for repeat_mode in ("typed", "stated"):
            if repeat_mode == "stated" and "history" not in name:
                continue
            labels_right, task_right, repeat_right, conv_ok, ood_pred, ood_gold = [], [], [], [], [], []
            for conversation in test:
                gold_repeat = _repeat_gold(conversation)
                last, ok = None, True
                for turn, wanted in zip(conversation, gold_repeat):
                    answer = table[turn["id"]]
                    right = answer["label"] == turn["label"]
                    labels_right.append(right)
                    if turn["label"] < len(TASKS):
                        task_right.append(right)
                    ood_pred.append(answer["label"] == LABELS.index("outofdomain"))
                    ood_gold.append(turn["label"] == LABELS.index("outofdomain"))
                    typed = last is not None and answer["label"] == last
                    if 0 <= answer["label"] < len(TASKS):
                        last = answer["label"]
                    if wanted is not None:
                        said = typed if repeat_mode == "typed" else bool(answer["repeat"])
                        repeat_right.append(right and said == wanted)
                        right = right and said == wanted
                    ok = ok and right
                conv_ok.append(ok)
            flags = np.asarray(conv_ok, dtype=np.float64)
            draws = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(2000)]
            ood_pred, ood_gold = np.asarray(ood_pred), np.asarray(ood_gold)
            key = name if repeat_mode == "typed" else f"{name}__stated_repeat"
            summary["systems"][key] = {
                "label_accuracy": float(np.mean(labels_right)),
                "task_turn_accuracy": float(np.mean(task_right)),
                "repeat_accuracy": float(np.mean(repeat_right)),
                "out_of_domain_recall": float(ood_pred[ood_gold].mean()),
                "out_of_domain_precision": float(ood_gold[ood_pred].mean()) if ood_pred.any() else None,
                "conversation_success": float(flags.mean()),
                "conversation_success_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "usd_per_1000_conversations": cost.get(name, 0.0),
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"test: {len(test)} conversations, {len(turns)} customer turns, {summary['joined_label_turns_test']} with two labels joined", flush=True)
    for key, row in summary["systems"].items():
        print(
            f"  {key:44s} label {100 * row['label_accuracy']:5.1f} task {100 * row['task_turn_accuracy']:5.1f} repeat {100 * row['repeat_accuracy']:5.1f} ood R {100 * row['out_of_domain_recall']:5.1f} P {100 * (row['out_of_domain_precision'] or 0):5.1f} "
            f"conversations {100 * row['conversation_success']:5.1f} [{100 * row['conversation_success_ci'][0]:.1f}, {100 * row['conversation_success_ci'][1]:.1f}] ${row['usd_per_1000_conversations']:.2f}/1000",
            flush=True,
        )
    print("MULTIDOGO_REPORT_DONE", flush=True)


if __name__ == "__main__":
    {"hosted": hosted, "train": train, "report": report}[sys.argv[1]]()

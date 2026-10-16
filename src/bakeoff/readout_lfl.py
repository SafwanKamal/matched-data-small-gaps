"""Local models through the hosted prompt and readout. Frozen rules are in readout_lfl_protocol.json."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from bakeoff import api_baseline as api
from bakeoff.protocol import ROOT
from bakeoff.readout import READOUT_PATH, IndexReadout
from bakeoff.suite import load_suite, load_suite_protocol
from bakeoff.suite_run import _report

RULES_PATH = ROOT / "readout_lfl_protocol.json"
OUT = ROOT / "artifacts" / "readout-lfl"
TOP_K = 20
MAX_NEW = 6


def generate_content(readout: IndexReadout, api_rules: dict, text: str, labels: list[str], stop_ids: set[int]) -> list[dict]:
    """Greedy decode and return what an API with top_logprobs would: each token with its alternatives."""
    tokenizer = readout.tokenizer
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": api.build_prompt(api_rules, text, labels)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(chat, add_special_tokens=False)
    width = len(api.codes_for(len(labels))[0])
    content, written = [], ""
    for _ in range(MAX_NEW):
        log_probs = readout._next_log_probs([ids])[0]
        values, indices = torch.topk(log_probs, TOP_K)
        chosen = int(indices[0])
        if chosen in stop_ids:
            break
        token = tokenizer.decode([chosen])
        content.append(
            {
                "token": token,
                "logprob": float(values[0]),
                "top_logprobs": [{"token": tokenizer.decode([int(i)]), "logprob": float(v)} for v, i in zip(values, indices)],
            }
        )
        ids.append(chosen)
        written += token.strip()
        if len(written) >= width:
            break
    return content


def run(name: str) -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    api_rules = json.loads(api.API_PATH.read_text(encoding="utf-8"))
    system = rules["systems"][name]
    source = load_suite_protocol()["systems"][system["from"].split("systems.")[1]]
    spec = json.loads(READOUT_PATH.read_text())
    spec["model"] = {"hf_id": source["hf_id"], "revision": source["revision"], "load_in_4bit": system["load_in_4bit"]}
    suite = load_suite(rules["datasets"])
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"protocol_id": rules["protocol_id"], "system": name, "datasets": {}}
    readout = None
    for dataset in rules["datasets"]:
        if dataset in summary["datasets"]:
            print(f"skip {name} {dataset}", flush=True)
            continue
        if readout is None:
            readout = IndexReadout(spec)
            stop_ids = {token for token in (readout.tokenizer.eos_token_id, readout.tokenizer.convert_tokens_to_ids("<|im_end|>")) if token is not None}
        block = suite[dataset]
        labels = block["labels"]
        scores, failures = {}, {}
        for split in ("dev", "test"):
            cache = out / f"{dataset}__{split}.npz"
            if cache.exists():
                stored = np.load(cache)
                scores[split], failures[split] = stored["scores"], {"n": int(len(stored["scores"])), "unscorable": int(stored["unscorable"])}
                continue
            print(f"{name} {dataset} {split} n={len(block[split])} options={len(labels)}", flush=True)
            rows = []
            for index, item in enumerate(block[split], start=1):
                rows.append({"error": None, "content": generate_content(readout, api_rules, item.text, labels, stop_ids)})
                if index % 500 == 0:
                    print(f"  {index}/{len(block[split])}", flush=True)
            scores[split], counts = api.scores_from(rows, len(labels))
            failures[split] = {"n": counts["n"], "unscorable": counts["unscorable"]}
            np.savez_compressed(cache, scores=scores[split], unscorable=counts["unscorable"])
        summary["datasets"][dataset] = _report(name, dataset, block, scores, {"readout": "top-k tree from local softmax, hosted prompt", "failures": failures})
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"LFL_DONE {name}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m bakeoff.readout_lfl <system>")
    run(sys.argv[1])

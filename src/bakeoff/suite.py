"""Wider Choice suite. Frozen rules are in suite_protocol.json."""

from __future__ import annotations

import json
import sys

import numpy as np

from bakeoff.data import Item, load_items
from bakeoff.protocol import ROOT, load_protocol

SUITE_PATH = ROOT / "suite_protocol.json"
OUT = ROOT / "artifacts" / "suite"


def load_suite_protocol() -> dict:
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


def _rngs(spec: dict) -> dict:
    order = spec["seed_spawn_order"]
    children = np.random.SeedSequence(spec["seed"]).spawn(len(order))
    return {name: np.random.default_rng(child) for name, child in zip(order, children, strict=True)}


def _take(rows: list, cap: int, rng) -> list:
    if len(rows) <= cap:
        return rows
    return [rows[index] for index in rng.permutation(len(rows))[:cap].tolist()]


def _items(prefix: str, rows: list[tuple[int, str, int]], labels: list[str]) -> list[Item]:
    return [Item(item_id=f"{prefix}{row}", row=row, text=text, gold_index=gold, gold_label=labels[gold]) for row, text, gold in rows]


def _hub(name: str, spec: dict, rngs: dict, dev_n: int, cap: int) -> dict:
    from datasets import load_dataset

    data = spec["datasets"][name]
    raw = load_dataset(data["hf_id"], data["config"], revision=data["revision"]) if data["config"] else load_dataset(data["hf_id"], revision=data["revision"])
    extra = {}
    if name == "clinc150":
        names = raw["test"].features["intent"].names
        oos = names.index("oos")
        labels = [label.replace("_", " ") for label in names if label != "oos"]
        remap = {index: position for position, index in enumerate(i for i in range(len(names)) if i != oos)}

        def rows(split):
            return [(row, example["text"], remap[example["intent"]]) for row, example in enumerate(raw[split]) if example["intent"] != oos]

        extra["oos"] = [
            Item(item_id=f"clinc150-oos-{row}", row=row, text=example["text"], gold_index=-1, gold_label="oos")
            for row, example in enumerate(raw["test"])
            if example["intent"] == oos
        ]
    elif name == "emotion":
        labels = list(raw["test"].features["label"].names)

        def rows(split):
            return [(row, example["text"], int(example["label"])) for row, example in enumerate(raw[split])]

    elif name == "yahoo":
        labels = list(raw["test"].features["topic"].names)

        def rows(split):
            table = raw[split]
            if split == "train":
                # Draw the dev rows before reading text, so 1.4M rows are not materialised.
                chosen = sorted(rngs["yahoo_dev"].permutation(len(table))[:dev_n].tolist())
                table = table.select(chosen)
                return [(index, f"{example['question_title']}\n{example['question_content']}", int(example["topic"])) for index, example in zip(chosen, table)]
            return [(row, f"{example['question_title']}\n{example['question_content']}", int(example["topic"])) for row, example in enumerate(table)]

    elif name == "anli_r3":
        labels = data["labels"]
        if list(raw["test_r3"].features["label"].names) != labels:
            raise RuntimeError("ANLI label order changed")

        def rows(split):
            return [(row, f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}", int(example["label"])) for row, example in enumerate(raw[split])]

    elif name == "boolq":
        labels = data["labels"]

        def rows(split):
            return [(row, f"Passage: {example['passage']}\nQuestion: {example['question']}?", int(bool(example["answer"]))) for row, example in enumerate(raw[split])]

    else:
        raise ValueError(name)
    if len(labels) != data["options"]:
        raise RuntimeError(f"{name} has {len(labels)} options, protocol says {data['options']}")
    dev_rows = rows(data["dev_from"].split(",")[0])
    test_rows = rows(data["test_from"].split(",")[0])
    if name != "yahoo":
        dev_rows = _take(dev_rows, dev_n, rngs.get(f"{name}_dev"))
    test_rows = _take(test_rows, cap, rngs.get(f"{name}_test"))
    out = {"labels": labels, "dev": _items(f"{name}-dev-", dev_rows, labels), "test": _items(f"{name}-test-", test_rows, labels)}
    out.update(extra)
    return out


def load_suite(names: list[str] | None = None) -> dict:
    spec = load_suite_protocol()
    rngs = _rngs(spec)
    out = {}
    for name in spec["datasets"]:
        if names and name not in names:
            # Generators are spawned per split, so skipping a dataset does not shift another's draw.
            continue
        if name == "banking77_full":
            from bakeoff.full_banking77 import PROTOCOL_PATH, load_full_test

            protocol = load_protocol()
            test, labels = load_full_test(protocol, json.loads(PROTOCOL_PATH.read_text()))
            dev = load_items(ROOT / "artifacts" / "splits.json", protocol)["items"]["banking77"]["dev"]
            out[name] = {"labels": labels, "dev": dev, "test": test}
        elif name == "massive_en":
            from bakeoff.massive import load_massive

            loaded = load_massive()
            out[name] = {"labels": loaded["labels"], "dev": loaded["items"]["dev"], "test": loaded["items"]["test"]}
        else:
            out[name] = _hub(name, spec, rngs, spec["dev_n"], spec["test_cap"])
    return out


def freeze() -> None:
    """Write the item ids of every split. Loads no model."""
    suite = load_suite()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_id": load_suite_protocol()["protocol_id"], "datasets": {}}
    for name, block in suite.items():
        dev_text = {item.text for item in block["dev"]}
        payload["datasets"][name] = {
            "labels": block["labels"],
            "dev_ids": [item.item_id for item in block["dev"]],
            "test_ids": [item.item_id for item in block["test"]],
            "oos_ids": [item.item_id for item in block.get("oos", [])],
            "exact_text_overlap_dev_test": sum(item.text in dev_text for item in block["test"]),
            "test_class_counts_min_max": [int(v) for v in (lambda c: (c.min(), c.max()))(np.bincount([item.gold_index for item in block["test"]], minlength=len(block["labels"])))],
        }
        row = payload["datasets"][name]
        print(f"{name:15s} options={len(block['labels']):3d} dev={len(block['dev']):4d} test={len(block['test']):4d} oos={len(row['oos_ids']):4d} overlap={row['exact_text_overlap_dev_test']} class min/max={row['test_class_counts_min_max']}", flush=True)
    (OUT / "splits.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        freeze()
    else:
        raise SystemExit("usage: python -m bakeoff.suite freeze")

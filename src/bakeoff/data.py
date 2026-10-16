"""Frozen splits. Gold labels are read here and never used to choose a split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset

from bakeoff.protocol import load_protocol


@dataclass(frozen=True)
class Item:
    item_id: str
    row: int
    text: str
    gold_index: int
    gold_label: str


def _spawn_rngs(protocol: dict) -> dict[str, np.random.Generator]:
    order = protocol["seed_spawn_order"]
    children = np.random.SeedSequence(protocol["seed"]).spawn(len(order))
    return {name: np.random.default_rng(child) for name, child in zip(order, children, strict=True)}


def _load_raw(protocol: dict) -> dict[str, dict[str, list]]:
    specs = protocol["datasets"]
    sst = load_dataset(specs["sst2"]["hf_id"], specs["sst2"]["subset"], revision=specs["sst2"]["revision"])
    ag = load_dataset(specs["ag_news"]["hf_id"], revision=specs["ag_news"]["revision"])
    bank = load_dataset(specs["banking77"]["hf_id"], revision=specs["banking77"]["revision"])
    return {"sst2": sst, "ag_news": ag, "banking77": bank}


def _labels_for(name: str, raw, protocol: dict) -> list[str]:
    expected = protocol["datasets"][name]["labels"]
    if name == "banking77":
        published = raw["test"].features["label"].names
        resolved = [label.replace("_", " ") for label in published]
    elif name == "sst2":
        resolved = list(raw["validation"].features["label"].names)
    else:
        resolved = list(raw["test"].features["label"].names)
    if resolved != expected:
        raise RuntimeError(f"{name} labels do not match the frozen protocol")
    return expected


def _rows(name: str, raw, split: str) -> list[tuple[int, str, int]]:
    if name == "sst2":
        table = raw[split]
        rows = []
        for row, example in enumerate(table):
            label = int(example["label"])
            if label < 0:
                raise RuntimeError(f"sst2 {split} row {row} has unlabeled gold {label}")
            rows.append((row, example["sentence"], label))
        return rows
    table = raw[split]
    return [(row, example["text"], int(example["label"])) for row, example in enumerate(table)]


def materialize(protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    raw = _load_raw(protocol)
    rngs = _spawn_rngs(protocol)
    splits: dict[str, dict] = {}
    items: dict[str, dict[str, list[Item]]] = {}
    for name, spec in protocol["datasets"].items():
        labels = _labels_for(name, raw[name], protocol)
        items[name] = {}
        splits[name] = {"labels": labels}
        for split_name, count_key, hf_split in (
            ("dev", "dev_n", spec["dev_hf_split"]),
            ("test", "test_n", spec["test_hf_split"]),
        ):
            rows = _rows(name, raw[name], hf_split)
            n = spec[count_key]
            if n > len(rows):
                raise RuntimeError(f"{name} {split_name} asks for {n} of {len(rows)}")
            if spec.get("test_is_full_split") and split_name == "test":
                chosen = list(range(len(rows)))
            else:
                spawn = f"{name}_{split_name}"
                chosen = rngs[spawn].permutation(len(rows))[:n].tolist()
            if len(chosen) != n:
                raise RuntimeError(f"{name} {split_name} materialized {len(chosen)} != {n}")
            prefix = spec["id_prefix"][split_name]
            built = []
            for row in chosen:
                source_row, text, gold = rows[row]
                if not 0 <= gold < len(labels):
                    raise RuntimeError(f"{name} gold {gold} out of range")
                built.append(
                    Item(
                        item_id=f"{prefix}{source_row}",
                        row=source_row,
                        text=text,
                        gold_index=gold,
                        gold_label=labels[gold],
                    )
                )
            items[name][split_name] = built
            splits[name][split_name] = {
                "hf_split": hf_split,
                "rows": [item.row for item in built],
                "ids": [item.item_id for item in built],
            }
        dev_ids = set(splits[name]["dev"]["ids"])
        test_ids = set(splits[name]["test"]["ids"])
        if dev_ids & test_ids:
            raise RuntimeError(f"{name} dev and test ids overlap")
        dev_text = {item.text for item in items[name]["dev"]}
        overlap = sum(1 for item in items[name]["test"] if item.text in dev_text)
        splits[name]["exact_text_overlap_dev_test"] = overlap
        splits[name]["dev_n"] = len(items[name]["dev"])
        splits[name]["test_n"] = len(items[name]["test"])
        splits[name]["source_n"] = {
            "dev": len(_rows(name, raw[name], spec["dev_hf_split"])),
            "test": len(_rows(name, raw[name], spec["test_hf_split"])),
        }
    return {"protocol_id": protocol["protocol_id"], "splits": splits, "items": items}


def write_splits(path: Path, protocol: dict | None = None) -> dict:
    materialized = materialize(protocol)
    payload = {
        "protocol_id": materialized["protocol_id"],
        "splits": materialized["splits"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return materialized


def load_items(path: Path, protocol: dict | None = None) -> dict:
    """Rebuild items from the dataset and require the frozen row lists to match."""
    protocol = protocol or load_protocol()
    frozen = json.loads(path.read_text())
    if frozen["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("splits.json was built for a different protocol")
    fresh = materialize(protocol)
    for name, spec in fresh["splits"].items():
        for split_name in ("dev", "test"):
            if spec[split_name]["rows"] != frozen["splits"][name][split_name]["rows"]:
                raise RuntimeError(f"frozen {name} {split_name} rows do not match the protocol seed")
    return fresh

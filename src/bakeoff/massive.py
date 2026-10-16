"""Official MASSIVE English intent split. Slots are not read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bakeoff.data import Item
from bakeoff.protocol import ROOT

PROTOCOL_PATH = ROOT / "primary_protocol.json"


def load_primary_protocol(path: Path | None = None) -> dict:
    return json.loads((path or PROTOCOL_PATH).read_text())


def protocol_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((path or PROTOCOL_PATH).read_bytes()).hexdigest()


def load_massive(protocol: dict | None = None) -> dict:
    """Load en-US.jsonl in file order. Official partitions are not filtered."""
    protocol = protocol or load_primary_protocol()
    spec = protocol["datasets"]["massive_en"]
    path = ROOT / spec["local_file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise RuntimeError("en-US.jsonl sha256 does not match the frozen protocol")
    labels: list[str] = spec["labels"]
    label_index = {label: index for index, label in enumerate(labels)}
    if list(label_index) != labels or labels != sorted(labels):
        raise RuntimeError("frozen MASSIVE labels are not unique and alphabetical")
    items: dict[str, list[Item]] = {"train": [], "dev": [], "test": []}
    ids: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    texts: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    seen_intents: set[str] = set()
    within_duplicate_keys = 0
    seen_text: dict[tuple[str, str], int] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["locale"] != "en-US":
                raise RuntimeError(f"unexpected locale {row['locale']}")
            part = row["partition"]
            if part not in items:
                raise RuntimeError(f"unexpected partition {part}")
            if "annot_utt" not in row or "utt" not in row or "intent" not in row:
                raise RuntimeError("MASSIVE row is missing utt or intent")
            source_id = str(row["id"])
            if not source_id.isdigit():
                raise RuntimeError(f"non-numeric MASSIVE id {source_id}")
            if source_id in ids[part]:
                raise RuntimeError(f"duplicate id {source_id} in {part}")
            display = str(row["intent"]).replace("_", " ")
            if display not in label_index:
                raise RuntimeError(f"intent {row['intent']} is not in the frozen label list")
            seen_intents.add(display)
            text = str(row["utt"])
            key = (part, text)
            seen_text[key] = seen_text.get(key, 0) + 1
            items[part].append(
                Item(
                    item_id=f"massive-en-{part}-{source_id}",
                    row=int(source_id),
                    text=text,
                    gold_index=label_index[display],
                    gold_label=display,
                )
            )
            ids[part].add(source_id)
            texts[part].add(text)
    within_duplicate_keys = sum(1 for count in seen_text.values() if count > 1)
    counts = {part: len(rows) for part, rows in items.items()}
    expected = {"train": spec["train_n"], "dev": spec["dev_n"], "test": spec["test_n"]}
    if counts != expected:
        raise RuntimeError(f"MASSIVE counts {counts} != {expected}")
    if set(seen_intents) != set(labels):
        raise RuntimeError("MASSIVE intents do not match the frozen label list")
    overlap = {
        "train_dev": len(ids["train"] & ids["dev"]),
        "train_test": len(ids["train"] & ids["test"]),
        "dev_test": len(ids["dev"] & ids["test"]),
    }
    if overlap != {"train_dev": 0, "train_test": 0, "dev_test": 0}:
        raise RuntimeError(f"MASSIVE ids overlap across partitions: {overlap}")
    text_overlap = {
        "train_dev": len(texts["train"] & texts["dev"]),
        "train_test": len(texts["train"] & texts["test"]),
        "dev_test": len(texts["dev"] & texts["test"]),
        "within_partition_duplicate_keys": within_duplicate_keys,
    }
    frozen_overlap = spec["text_overlap_kept"]
    if text_overlap != frozen_overlap:
        raise RuntimeError(f"text overlap {text_overlap} != frozen {frozen_overlap}")
    return {
        "items": items,
        "counts": counts,
        "sha256": digest,
        "text_overlap_kept": text_overlap,
        "labels": labels,
        "slots": "ignored",
        "text_field": "utt",
    }

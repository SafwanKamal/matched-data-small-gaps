"""MASSIVE loader checks. No GPU and no test scoring."""

from __future__ import annotations

import json

from bakeoff.massive import load_massive, load_primary_protocol
from bakeoff.protocol import ROOT


def test_official_split_and_labels() -> None:
    protocol = load_primary_protocol()
    loaded = load_massive(protocol)
    spec = protocol["datasets"]["massive_en"]
    assert loaded["counts"] == {"train": 11514, "dev": 2033, "test": 2974}
    assert spec["test_is_full_split"] is True
    assert spec["dev_is_full_split"] is True
    assert protocol["subsample"] is None
    assert loaded["labels"] == spec["labels"]
    assert len(loaded["labels"]) == 60
    assert loaded["slots"] == "ignored"
    train_ids = {item.row for item in loaded["items"]["train"]}
    dev_ids = {item.row for item in loaded["items"]["dev"]}
    test_ids = {item.row for item in loaded["items"]["test"]}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)
    first = loaded["items"]["test"][0]
    assert first.row == 0
    assert first.text == "wake me up at five am this week"
    assert first.gold_label == "alarm set"
    assert first.gold_index == spec["labels"].index("alarm set")
    assert protocol["llm_aggrefact"]["scored"] is False
    assert protocol["ettin_encoder"]["losses"] == ["log_loss"]
    assert protocol["ettin_decoder"]["losses"] == ["log_loss"]
    assert protocol["ettin_decoder"]["generate"] is False
    program1 = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["systems"] == program1["systems"]
    program2 = json.loads((ROOT / "program2.json").read_text())
    decoder = json.loads((ROOT / "program2_decoder.json").read_text())
    for key in (
        "model",
        "revision",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "max_grad_norm",
        "seed",
        "max_length",
    ):
        assert protocol["ettin_encoder"][key] == program2[key]
    for key in (
        "model",
        "revision",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "epochs",
        "batch_size",
        "label_chunk_size",
        "eval_batch_size",
        "max_grad_norm",
        "seed",
        "max_prompt_tokens",
        "generate",
    ):
        assert protocol["ettin_decoder"][key] == decoder[key]


def main() -> None:
    test_official_split_and_labels()
    print("ok")


if __name__ == "__main__":
    main()

"""Wrapper-variety curve for the Banking77 encoder. Frozen rules are in variety_protocol.json."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from bakeoff import context_pilot
from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature
from bakeoff.train_ettin import EncoderClassifier, _batches, _encode, _score_items, _train_rows, decision_loss

RULES_PATH = ROOT / "variety_protocol.json"
OUT = ROOT / "artifacts" / "variety"
SEP = "\x1f"


def render(convention: dict, target: str, earlier: list[str], position: str) -> str:
    lines = [convention["earlier"].format(d=text, n=index) for index, text in enumerate(earlier, start=1)]
    target_line = convention["target"].format(t=target)
    header = [convention["header"]] if convention["header"] else []
    if position == "last":
        parts = header + lines + [target_line]
    elif position == "first":
        parts = [target_line] + header + lines
    else:
        half = len(lines) // 2
        parts = header + lines[:half] + [target_line] + header + lines[half:]
    return "\n".join(parts)


def test_conditions(rules: dict) -> dict:
    pilot = json.loads(context_pilot.PILOT_PATH.read_text(encoding="utf-8"))
    extended = copy.deepcopy(pilot)
    # One extra template exposes the pilot's own distractor draws so new layouts reuse them.
    extended["conditions"]["_fields"] = SEP.join(["{target}"] + [f"{{d{index}}}" for index in range(1, 9)])
    built = context_pilot.build(extended)["banking77"]["conditions"]
    conditions = {name: built[name] for name in ("clean", "chatter", "typed_summary")}
    from dataclasses import replace

    families = [("seen", rules["conventions"]["training"][0])] + [(f"held_out_{index}", convention) for index, convention in enumerate(rules["conventions"]["held_out"], start=1)]
    for family, convention in families:
        for position in ("last", "first", "middle"):
            items = []
            for item in built["_fields"]:
                fields = item.text.split(SEP)
                items.append(replace(item, text=render(convention, fields[0], fields[1:], position)))
            conditions[f"{family}__{position}"] = items
    return conditions


def _wrap(text: str, gold: int, by_label: dict, conventions: list[dict], clean_share: float, rng) -> str:
    if rng.random() < clean_share:
        return text
    others = [label for label in by_label if label != gold]
    chosen = rng.choice(others, size=int(rng.integers(1, 9)), replace=False).tolist()
    earlier = [by_label[label][int(rng.integers(0, len(by_label[label])))] for label in chosen]
    convention = conventions[int(rng.integers(0, len(conventions)))]
    return render(convention, text, earlier, "last" if rng.random() < 0.5 else "first")


def train_one(level: int, seed: int, rules: dict, spec: dict, rows, by_label, labels, dev, conditions, device) -> dict:
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    encoder = AutoModel.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
    model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    steps = (len(rows) + spec["batch_size"] - 1) // spec["batch_size"]
    total = steps * spec["epochs"]
    warmup = int(spec["warmup_ratio"] * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    rng = np.random.default_rng([rules["wrap_seed"], level, seed])
    conventions = rules["conventions"]["training"][:level]
    for epoch in range(spec["epochs"]):
        model.train()
        losses = []
        for batch in _batches(rows, spec["batch_size"], seed, epoch):
            texts = [_wrap(text, gold, by_label, conventions, rules["training_inputs"]["clean_share"], rng) for text, gold in batch]
            encoded = _encode(tokenizer, texts, spec["max_length"], device)
            gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = decision_loss(model(encoded["input_ids"], encoded["attention_mask"]), gold, "log_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if len(losses) % 50 == 0:
                torch.cuda.empty_cache()
        print(f"level {level} seed {seed} epoch {epoch + 1} mean_loss {np.mean(losses):.4f}", flush=True)
    dev_scores, dev_gold, _ = _score_items(model, tokenizer, dev, spec, device, 32)
    temperature = fit_temperature(dev_scores, dev_gold)["temperature"]
    gold = np.asarray([item.gold_index for item in conditions["clean"]], dtype=np.int64)
    result = {"temperature": temperature, "conditions": {}}
    saved = {}
    for name, items in conditions.items():
        scores, _, truncated = _score_items(model, tokenizer, items, spec, device, 32)
        probs = apply_temperature(scores, temperature)
        result["conditions"][name] = {
            "accuracy": float((probs.argmax(axis=1) == gold).mean()),
            "brier": float(((probs - np.eye(len(labels))[gold]) ** 2).sum(axis=1).mean()),
            "mean_top_probability": float(probs.max(axis=1).mean()),
            "share_over_cap": truncated / len(items),
        }
        saved[name] = scores
    np.savez_compressed(OUT / f"level-{level}__seed-{seed}.npz", gold=gold, **saved)
    held = [value["accuracy"] for name, value in result["conditions"].items() if name.startswith("held_out")]
    result["held_out_mean_accuracy"] = float(np.mean(held))
    for position in ("last", "first", "middle"):
        result[f"held_out_{position}"] = float(np.mean([value["accuracy"] for name, value in result["conditions"].items() if name.startswith("held_out") and name.endswith(position)]))
    row = result["conditions"]
    print(
        f"level {level} seed {seed}: clean {row['clean']['accuracy']:.3f} held-out {result['held_out_mean_accuracy']:.3f} "
        f"(last {result['held_out_last']:.3f} first {result['held_out_first']:.3f} middle {result['held_out_middle']:.3f}) "
        f"seen-middle {row['seen__middle']['accuracy']:.3f} chatter {row['chatter']['accuracy']:.3f} typed {row['typed_summary']['accuracy']:.3f}",
        flush=True,
    )
    del model, encoder, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    protocol = load_protocol()
    spec = json.loads((ROOT / "program2.json").read_text())
    spec["max_length"] = 512
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    labels = protocol["datasets"]["banking77"]["labels"]
    dev = materialized["items"]["banking77"]["dev"]
    rows = _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"]))
    by_label: dict[int, list[str]] = {}
    for text, gold in rows:
        by_label.setdefault(gold, []).append(text)
    conditions = test_conditions(rules)
    (OUT / "examples.json").write_text(json.dumps({name: items[0].text for name, items in conditions.items()}, indent=2) + "\n", encoding="utf-8")
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": rules["protocol_id"], "runs": {}}
    for seed in rules["seeds"]:
        for level in rules["levels"]:
            key = f"level-{level}__seed-{seed}"
            if key in summary["runs"]:
                print(f"skip {key}", flush=True)
                continue
            summary["runs"][key] = train_one(level, seed, rules, spec, rows, by_label, labels, dev, conditions, device)
            path.write_text(json.dumps(summary, indent=2) + "\n")
    print("VARIETY_DONE", flush=True)


if __name__ == "__main__":
    main()

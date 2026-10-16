"""Fine-tune the Ettin encoder on wrapped Banking77 inputs. Frozen rules are in wrapped_protocol.json."""

from __future__ import annotations

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

WRAPPED_PATH = ROOT / "wrapped_protocol.json"
OUT = ROOT / "artifacts" / "wrapped"


def _wrap(text: str, gold: int, by_label: dict, templates: dict, rng) -> str:
    draw = rng.random()
    if draw < 1 / 3:
        return text
    others = [label for label in by_label if label != gold]
    if draw < 2 / 3:
        label = others[int(rng.integers(0, len(others)))]
        pool = by_label[label]
        return templates["related_prior"].format(d1=pool[int(rng.integers(0, len(pool)))], target=text)
    count = int(rng.integers(2, 9))
    chosen = rng.choice(others, size=count, replace=False).tolist()
    earlier = "\n".join(f"- {by_label[label][int(rng.integers(0, len(by_label[label])))]}" for label in chosen)
    return f"Earlier messages (already resolved):\n{earlier}\nCurrent message: {text}"


def _evaluate(model, tokenizer, conditions: dict, dev, spec: dict, device) -> dict:
    dev_scores, dev_gold, _ = _score_items(model, tokenizer, dev, spec, device, 32)
    temperature = fit_temperature(dev_scores, dev_gold)["temperature"]
    gold = np.asarray([item.gold_index for item in conditions["clean"]], dtype=np.int64)
    scores = {}
    extra = {}
    for name, items in conditions.items():
        values, _, truncated = _score_items(model, tokenizer, items, spec, device, 32)
        scores[name] = values
        extra[name] = {"share_over_cap": truncated / len(items)}
    return {"scores": scores, "temperature": temperature, "gold": gold, "extra": extra}


def main() -> None:
    rules = json.loads(WRAPPED_PATH.read_text(encoding="utf-8"))
    pilot = json.loads(context_pilot.PILOT_PATH.read_text(encoding="utf-8"))
    protocol = load_protocol()
    spec = json.loads((ROOT / "program2.json").read_text())
    spec["max_length"] = rules["changes"]["max_length"]
    torch.manual_seed(spec["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    labels = protocol["datasets"]["banking77"]["labels"]
    dev = materialized["items"]["banking77"]["dev"]
    rows = _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"]))
    by_label: dict[int, list[str]] = {}
    for text, gold in rows:
        by_label.setdefault(gold, []).append(text)
    conditions = context_pilot.build(pilot)["banking77"]["conditions"]
    summary = {"protocol_id": rules["protocol_id"], "models": {}}

    # The bare-trained checkpoint with the longer cap, so truncation is separated from input shape.
    artifact = ROOT / "artifacts" / "program2" / "banking77-log_loss"
    tokenizer = AutoTokenizer.from_pretrained(artifact / "encoder")
    bare = EncoderClassifier(AutoModel.from_pretrained(artifact / "encoder", dtype=torch.float32), 768, len(labels)).to(device)
    bare.head.load_state_dict(torch.load(artifact / "head.pt", map_location="cpu", weights_only=True)["head"])
    result = _evaluate(bare, tokenizer, conditions, dev, spec, device)
    summary["models"]["bare_512"] = context_pilot._report("bare_512", "banking77", result["scores"], result["temperature"], result["gold"], result["extra"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    del bare

    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    encoder = AutoModel.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
    model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    steps = (len(rows) + spec["batch_size"] - 1) // spec["batch_size"]
    total = steps * spec["epochs"]
    warmup = int(spec["warmup_ratio"] * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    rng = np.random.default_rng(rules["seed"])
    losses = []
    for epoch in range(spec["epochs"]):
        model.train()
        epoch_losses = []
        for batch in _batches(rows, spec["batch_size"], spec["seed"], epoch):
            texts = [_wrap(text, gold, by_label, pilot["conditions"], rng) for text, gold in batch]
            encoded = _encode(tokenizer, texts, spec["max_length"], device)
            gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = decision_loss(model(encoded["input_ids"], encoded["attention_mask"]), gold, "log_loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
            if len(epoch_losses) % 50 == 0:
                print(f"wrapped epoch {epoch + 1} step {len(epoch_losses)}/{steps} loss {epoch_losses[-1]:.4f}", flush=True)
        losses.extend(epoch_losses)
        print(f"wrapped epoch {epoch + 1} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
    result = _evaluate(model, tokenizer, conditions, dev, spec, device)
    block = context_pilot._report("wrapped_512", "banking77", result["scores"], result["temperature"], result["gold"], result["extra"])
    block["mean_train_loss"] = float(np.mean(losses))
    summary["models"]["wrapped_512"] = block
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"head": model.head.state_dict(), "labels": labels}, OUT / "head.pt")
    model.encoder.save_pretrained(OUT / "encoder")
    tokenizer.save_pretrained(OUT / "encoder")
    np.savez_compressed(OUT / "test_scores.npz", gold=result["gold"], **result["scores"])
    print("WRAPPED_DONE", flush=True)


if __name__ == "__main__":
    main()

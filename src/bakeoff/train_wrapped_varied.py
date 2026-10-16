"""Wrapped fine-tune with the target first or last. Frozen rules are in wrapped_varied_protocol.json."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from bakeoff import context_pilot
from bakeoff.data import load_items
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.train_ettin import EncoderClassifier, _batches, _encode, _train_rows, decision_loss
from bakeoff.train_wrapped import _evaluate

RULES_PATH = ROOT / "wrapped_varied_protocol.json"


def _wrap(text: str, gold: int, by_label: dict, rules: dict, rng) -> str:
    names = ["clean", "one_earlier_target_last", "one_earlier_target_first", "history_target_last", "history_target_first"]
    weights = [rules[name] for name in names]
    name = names[int(rng.choice(len(names), p=np.asarray(weights) / sum(weights)))]
    if name == "clean":
        return text
    others = [label for label in by_label if label != gold]

    def draw(label):
        pool = by_label[label]
        return pool[int(rng.integers(0, len(pool)))]

    template = rules["templates"][name]
    if name.startswith("one_earlier"):
        return template.format(d1=draw(others[int(rng.integers(0, len(others)))]), target=text)
    chosen = rng.choice(others, size=int(rng.integers(2, 9)), replace=False).tolist()
    return template.format(earlier="\n".join(f"- {draw(label)}" for label in chosen), target=text)


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    pilot = json.loads(context_pilot.PILOT_PATH.read_text(encoding="utf-8"))
    extended = copy.deepcopy(pilot)
    # Distractors are drawn per item before any template is filled, so adding conditions leaves the draws unchanged.
    extended["conditions"].update(rules["test"]["extra_conditions"])
    protocol = load_protocol()
    spec = json.loads((ROOT / "program2.json").read_text())
    spec["max_length"] = 512
    torch.manual_seed(spec["seed"])
    device = torch.device("cuda")
    out = ROOT / "artifacts" / rules["out"]
    out.mkdir(parents=True, exist_ok=True)

    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    labels = protocol["datasets"]["banking77"]["labels"]
    dev = materialized["items"]["banking77"]["dev"]
    rows = _train_rows("banking77", protocol, set(materialized["splits"]["banking77"]["dev"]["rows"]))
    by_label: dict[int, list[str]] = {}
    for text, gold in rows:
        by_label.setdefault(gold, []).append(text)
    conditions = context_pilot.build(extended)["banking77"]["conditions"]
    summary = {"protocol_id": rules["protocol_id"], "models": {}}

    previous = ROOT / "artifacts" / "wrapped"
    tokenizer = AutoTokenizer.from_pretrained(previous / "encoder")
    last_only = EncoderClassifier(AutoModel.from_pretrained(previous / "encoder", dtype=torch.float32), 768, len(labels)).to(device)
    last_only.head.load_state_dict(torch.load(previous / "head.pt", map_location="cpu", weights_only=True)["head"])
    result = _evaluate(last_only, tokenizer, conditions, dev, spec, device)
    summary["models"]["target_last_only"] = context_pilot._report("target_last_only", "banking77", result["scores"], result["temperature"], result["gold"], result["extra"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    del last_only
    torch.cuda.empty_cache()

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
            texts = [_wrap(text, gold, by_label, rules["training_inputs"], rng) for text, gold in batch]
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
                torch.cuda.empty_cache()
        losses.extend(epoch_losses)
        print(f"varied epoch {epoch + 1} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
    result = _evaluate(model, tokenizer, conditions, dev, spec, device)
    block = context_pilot._report("target_first_or_last", "banking77", result["scores"], result["temperature"], result["gold"], result["extra"])
    block["mean_train_loss"] = float(np.mean(losses))
    summary["models"]["target_first_or_last"] = block
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"head": model.head.state_dict(), "labels": labels}, out / "head.pt")
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    np.savez_compressed(out / "test_scores.npz", gold=result["gold"], **result["scores"])
    print("VARIED_DONE", flush=True)


if __name__ == "__main__":
    main()

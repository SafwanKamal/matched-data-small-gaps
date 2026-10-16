"""Program 2: finetune the Ettin 150M encoder. Dev is temperature only."""

from __future__ import annotations

import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature, schema_valid, score_probabilities
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

PROGRAM2_PATH = ROOT / "program2.json"


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, n_labels: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Weights stay float32 so AdamW updates at 2e-5 are not rounded away; the forward is bfloat16.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def decision_loss(logits: torch.Tensor, gold: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "log_loss":
        return F.cross_entropy(logits, gold)
    if kind == "brier":
        probs = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, gold.unsqueeze(1), 1.0)
        return ((probs - one_hot) ** 2).sum(dim=-1).mean()
    raise ValueError(kind)


def _train_rows(name: str, protocol: dict, excluded: set[int]) -> list[tuple[str, int]]:
    spec = protocol["datasets"][name]
    if name == "sst2":
        raw = load_dataset(spec["hf_id"], spec["subset"], revision=spec["revision"])["train"]
        rows = []
        for row, example in enumerate(raw):
            if row in excluded:
                continue
            rows.append((example["sentence"], int(example["label"])))
        return rows
    raw = load_dataset(spec["hf_id"], revision=spec["revision"])["train"]
    return [
        (example["text"], int(example["label"]))
        for row, example in enumerate(raw)
        if row not in excluded
    ]


def _batches(rows: list, batch_size: int, seed: int, epoch: int):
    order = list(range(len(rows)))
    rng = random.Random(seed + epoch)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start : start + batch_size]]


def _encode(tokenizer, texts: list[str], max_length: int, device: torch.device):
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _score_items(model, tokenizer, items, spec, device, batch_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval()
    scores = []
    gold = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            texts = [item.text for item in chunk]
            lengths = tokenizer(texts, truncation=False, padding=False)["input_ids"]
            truncated += sum(len(ids) > spec["max_length"] for ids in lengths)
            encoded = _encode(tokenizer, texts, spec["max_length"], device)
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            scores.append(logits.detach().cpu().numpy().astype(np.float64))
            gold.extend(item.gold_index for item in chunk)
    return np.concatenate(scores, axis=0), np.asarray(gold, dtype=np.int64), truncated


def _metrics(scores: np.ndarray, gold: np.ndarray, temperature: float) -> dict:
    probs = apply_temperature(scores, temperature)
    if not all(schema_valid(row, probs.shape[1]) for row in probs):
        raise RuntimeError("trained head emitted a distribution that is not a simplex")
    scored = score_probabilities(probs, gold)
    scored.pop("risk_coverage", None)
    scored.pop("ece_bins", None)
    return scored


def _train_one(job: str, dataset: str, kind: str, spec: dict, protocol: dict, materialized, device, rows=None, out_root=None) -> dict:
    labels = protocol["datasets"][dataset]["labels"]
    excluded = set(materialized["splits"][dataset]["dev"]["rows"])
    # The size sweep passes its own training rows and output folder; Program 2 uses the defaults.
    rows = rows if rows is not None else _train_rows(dataset, protocol, excluded)
    if any(gold < 0 or gold >= len(labels) for _, gold in rows):
        raise RuntimeError(f"{dataset} gold outside the frozen label set")
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(torch.cuda.get_device_name(0))

    def build(batch_size: int):
        encoder = AutoModel.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
        built = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
        devices = {parameter.device.type for parameter in built.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"Ettin parameters are on {devices}")
        opt = torch.optim.AdamW(built.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
        steps = max(1, (len(rows) + batch_size - 1) // batch_size)
        total = steps * spec["epochs"]
        warmup = int(spec["warmup_ratio"] * total)

        def lr_at(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            remain = max(1, total - warmup)
            return max(0.0, (total - step) / remain)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        return built, opt, sched, steps

    batch_size = spec["batch_size"]
    oom_fallback = False
    model, optimizer, scheduler, steps_per_epoch = build(batch_size)
    losses: list[float] = []
    epoch = 0
    while epoch < spec["epochs"]:
        try:
            model.train()
            epoch_losses = []
            for batch in _batches(rows, batch_size, spec["seed"], epoch):
                encoded = _encode(tokenizer, [text for text, _ in batch], spec["max_length"], device)
                gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(encoded["input_ids"], encoded["attention_mask"])
                loss = decision_loss(logits, gold, kind)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.extend(epoch_losses)
            epoch += 1
            print(f"{job} epoch {epoch} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            if oom_fallback:
                raise
            oom_fallback = True
            batch_size = spec["oom_batch_fallback"]
            print(f"{job} OOM, reloading and retrying at batch {batch_size}", flush=True)
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            model, optimizer, scheduler, steps_per_epoch = build(batch_size)
            losses = []
            epoch = 0

    dev_scores, dev_gold, _ = _score_items(
        model, tokenizer, materialized["items"][dataset]["dev"], spec, device, spec["eval_batch_size"]
    )
    fitted = fit_temperature(dev_scores, dev_gold)
    test_scores, test_gold, n_truncated = _score_items(
        model, tokenizer, materialized["items"][dataset]["test"], spec, device, spec["eval_batch_size"]
    )
    out_dir = (out_root or ROOT / "artifacts" / "program2") / job
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"head": model.head.state_dict(), "dataset": dataset, "loss": kind, "labels": labels},
        out_dir / "head.pt",
    )
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    result = {
        "dataset": dataset,
        "loss": kind,
        "train_n": len(rows),
        "excluded_dev_n": len(excluded),
        "epochs": spec["epochs"],
        "batch_size": batch_size,
        "oom_fallback": oom_fallback,
        "mean_train_loss": float(np.mean(losses)) if losses else None,
        "temperature": fitted,
        "scaled": _metrics(test_scores, test_gold, fitted["temperature"]),
        "unscaled": _metrics(test_scores, test_gold, 1.0),
        "n_truncated_test": n_truncated,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "device": torch.cuda.get_device_name(0),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out_dir / "test_scores.npz", scores=test_scores, gold=test_gold)
    print(
        f"{job} test brier_scaled {result['scaled']['brier']:.4f} acc {result['scaled']['accuracy']:.4f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return result


def main(seed: int | None = None) -> None:
    spec = json.loads(PROGRAM2_PATH.read_text())
    protocol = load_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    device = torch.device("cuda")
    torch.manual_seed(spec["seed"])
    results = []
    failures = []
    out_root = ROOT / "artifacts" / "program2"
    # An extra seed repeats the frozen recipe and writes under seed-<n>/. The frozen seed keeps its paths.
    prefix = ""
    if seed is not None:
        if seed not in spec["extra_seeds"]:
            raise RuntimeError(f"seed {seed} is not listed in extra_seeds")
        spec["seed"] = seed
        prefix = f"seed-{seed}/"
        torch.manual_seed(seed)
    (out_root / prefix).mkdir(parents=True, exist_ok=True)
    for dataset in spec["datasets"]:
        for kind in spec["losses"]:
            job = f"{prefix}{dataset}-{kind}"
            done = out_root / job / "metrics.json"
            if done.exists():
                print(f"skip {job}", flush=True)
                results.append(json.loads(done.read_text()))
                continue
            print(f"start {job}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                results.append(_train_one(job, dataset, kind, spec, protocol, materialized, device))
            except Exception:
                failures.append({"job": job, "traceback": traceback.format_exc()})
                print(failures[-1]["traceback"], flush=True)
                torch.cuda.empty_cache()
    payload = {"protocol_id": spec["protocol_id"], "runs": results, "failures": failures}
    payload["seed"] = spec["seed"]
    (out_root / prefix / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_root / prefix / 'summary.json'}", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--seed":
        main(int(sys.argv[2]))
    else:
        main()

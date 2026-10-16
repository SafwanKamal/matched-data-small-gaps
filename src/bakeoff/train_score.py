"""Program 3: flat softmax versus a CORAL head on SST-5. Same encoder budget as Program 2."""

from __future__ import annotations

import json
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from transformers import AutoModel, AutoTokenizer

from bakeoff.metrics import apply_temperature, score_probabilities
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature
from bakeoff.train_ettin import EncoderClassifier, _batches, _encode, decision_loss

PROGRAM3_PATH = ROOT / "program3.json"


def ranked_probability_score(probs: np.ndarray, gold: np.ndarray) -> float:
    values = np.asarray(probs, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    cdf = np.cumsum(values, axis=1)[:, :-1]
    levels = np.arange(values.shape[1] - 1)
    truth = (gold[:, None] <= levels[None, :]).astype(np.float64)
    return float(np.mean(np.sum((cdf - truth) ** 2, axis=1)))


def expectation_mae(probs: np.ndarray, gold: np.ndarray) -> float:
    values = np.asarray(probs, dtype=np.float64)
    levels = np.arange(values.shape[1], dtype=np.float64)
    expected = values @ levels
    return float(np.mean(np.abs(expected - np.asarray(gold, dtype=np.float64))))


def coral_distribution(cumulative_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Class probabilities from CORAL sigmoids. Returns probs and a repair flag per row."""
    survival = torch.sigmoid(cumulative_logits)
    pieces = [1.0 - survival[:, :1]]
    if survival.shape[1] > 1:
        pieces.append(survival[:, :-1] - survival[:, 1:])
    pieces.append(survival[:, -1:])
    raw = torch.cat(pieces, dim=1)
    repaired = raw < 0
    clipped = raw.clamp(min=0.0)
    total = clipped.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return clipped / total, repaired.any(dim=1)


class CoralClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, n_levels: int, bias_init: torch.Tensor | None = None):
        super().__init__()
        self.encoder = encoder
        self.weight = nn.Parameter(torch.empty(hidden))
        # Zero thresholds coincide, and then only the two end levels can receive mass.
        self.bias = nn.Parameter(torch.zeros(n_levels - 1) if bias_init is None else bias_init.clone().float())
        nn.init.normal_(self.weight, std=0.02)

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def cumulative_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.pooled(input_ids, attention_mask) @ self.weight.unsqueeze(1) + self.bias


def coral_bias_init(gold: list[int], n_levels: int) -> torch.Tensor:
    """Logit of the training frequency of gold > k, for k = 0..K-2."""
    values = np.asarray(gold, dtype=np.int64)
    rates = np.asarray([(values > level).mean() for level in range(n_levels - 1)], dtype=np.float64)
    rates = np.clip(rates, 1e-4, 1.0 - 1e-4)
    return torch.tensor(np.log(rates / (1.0 - rates)), dtype=torch.float32)


def _load_rows(spec: dict) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]], int]:
    data_spec = spec["dataset"]
    raw = load_dataset(data_spec["hf_id"], revision=data_spec["revision"])
    labels = data_spec["labels"]

    def rows(split: str) -> list[tuple[str, int]]:
        loaded = []
        for example in raw[split]:
            gold = int(example[data_spec["label_field"]])
            if gold < 0 or gold >= len(labels):
                raise RuntimeError(f"SST-5 gold {gold} is outside the frozen label set")
            loaded.append((example[data_spec["text_field"]], gold))
        return loaded

    dev = rows(data_spec["dev_split"])
    test = rows(data_spec["test_split"])
    held = {text for text, _ in dev} | {text for text, _ in test}
    train = []
    excluded = 0
    for text, gold in rows(data_spec["train_split"]):
        if data_spec["exclude_train_texts_in_dev_or_test"] and text in held:
            excluded += 1
            continue
        train.append((text, gold))
    return train, dev, test, excluded


def _score_softmax(model, tokenizer, rows, spec, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    gold = []
    with torch.inference_mode():
        for start in range(0, len(rows), spec["eval_batch_size"]):
            batch = rows[start : start + spec["eval_batch_size"]]
            encoded = _encode(tokenizer, [text for text, _ in batch], spec["max_length"], device)
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            scores.append(logits.detach().cpu().numpy().astype(np.float64))
            gold.extend(label for _, label in batch)
    return np.concatenate(scores, axis=0), np.asarray(gold, dtype=np.int64)


def _score_coral(model, tokenizer, rows, spec, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores = []
    gold = []
    repairs = []
    with torch.inference_mode():
        for start in range(0, len(rows), spec["eval_batch_size"]):
            batch = rows[start : start + spec["eval_batch_size"]]
            encoded = _encode(tokenizer, [text for text, _ in batch], spec["max_length"], device)
            probs, repaired = coral_distribution(model.cumulative_logits(encoded["input_ids"], encoded["attention_mask"]))
            logged = torch.log(probs.clamp(min=1e-12))
            scores.append(logged.detach().cpu().numpy().astype(np.float64))
            gold.extend(label for _, label in batch)
            repairs.extend(bool(flag) for flag in repaired.detach().cpu().numpy())
    return np.concatenate(scores, axis=0), np.asarray(gold, dtype=np.int64), float(np.mean(repairs))


def _pack(scores: np.ndarray, gold: np.ndarray, temperature: float, repair: float | None) -> dict:
    probs = apply_temperature(scores, temperature)
    scored = score_probabilities(probs, gold)
    scored.pop("risk_coverage", None)
    scored.pop("ece_bins", None)
    scored["ranked_probability_score"] = ranked_probability_score(probs, gold)
    scored["expectation_mae"] = expectation_mae(probs, gold)
    if repair is not None:
        scored["coral_repair_fraction"] = repair
    return scored


def _fit_and_save(job: str, model, tokenizer, dev, test, spec, device, kind: str, train_n: int, excluded: int, batch_size: int, oom: bool, losses: list[float]):
    if kind == "softmax_log_loss":
        dev_scores, dev_gold = _score_softmax(model, tokenizer, dev, spec, device)
        test_scores, test_gold = _score_softmax(model, tokenizer, test, spec, device)
        repair = None
    else:
        dev_scores, dev_gold, _ = _score_coral(model, tokenizer, dev, spec, device)
        test_scores, test_gold, repair = _score_coral(model, tokenizer, test, spec, device)
    fitted = fit_temperature(dev_scores, dev_gold)
    out_dir = ROOT / "artifacts" / "program3" / job
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "softmax_log_loss":
        torch.save({"head": model.head.state_dict(), "kind": kind}, out_dir / "head.pt")
    else:
        torch.save({"weight": model.weight.detach().cpu(), "bias": model.bias.detach().cpu(), "kind": kind}, out_dir / "head.pt")
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    result = {
        "dataset": "sst5",
        "loss": kind,
        "train_n": train_n,
        "excluded_overlap_n": excluded,
        "dev_n": len(dev),
        "epochs": spec["epochs"],
        "batch_size": batch_size,
        "oom_fallback": oom,
        "mean_train_loss": float(np.mean(losses)) if losses else None,
        "temperature": fitted,
        "scaled": _pack(test_scores, test_gold, fitted["temperature"], repair),
        "unscaled": _pack(test_scores, test_gold, 1.0, repair),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "device": torch.cuda.get_device_name(0),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out_dir / "test_scores.npz", scores=test_scores, gold=test_gold)
    print(
        f"{job} test brier {result['scaled']['brier']:.4f} rps {result['scaled']['ranked_probability_score']:.4f} "
        f"acc {result['scaled']['accuracy']:.4f}",
        flush=True,
    )
    return result


def _train_softmax(spec, train, dev, test, excluded, device) -> dict:
    job = "sst5-softmax_log_loss"
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    n_labels = len(spec["dataset"]["labels"])

    def build(batch_size: int):
        encoder = AutoModel.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
        model = EncoderClassifier(encoder, encoder.config.hidden_size, n_labels).to(device)
        devices = {parameter.device.type for parameter in model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(f"parameters are on {devices}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
        steps = max(1, (len(train) + batch_size - 1) // batch_size) * spec["epochs"]
        warmup = int(spec["warmup_ratio"] * steps)

        def lr_at(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (steps - step) / max(1, steps - warmup))

        return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    batch_size = spec["batch_size"]
    oom = False
    model, optimizer, scheduler = build(batch_size)
    losses: list[float] = []
    epoch = 0
    while epoch < spec["epochs"]:
        try:
            model.train()
            epoch_losses = []
            for batch in _batches(train, batch_size, spec["seed"], epoch):
                encoded = _encode(tokenizer, [text for text, _ in batch], spec["max_length"], device)
                gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = decision_loss(model(encoded["input_ids"], encoded["attention_mask"]), gold, "log_loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.extend(epoch_losses)
            epoch += 1
            print(f"{job} epoch {epoch} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            if oom:
                raise
            oom = True
            batch_size = spec["oom_batch_fallback"]
            print(f"{job} OOM, reloading at batch {batch_size}", flush=True)
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            model, optimizer, scheduler = build(batch_size)
            losses = []
            epoch = 0
    result = _fit_and_save(job, model, tokenizer, dev, test, spec, device, job.split("-", 1)[1], len(train), excluded, batch_size, oom, losses)
    del model
    torch.cuda.empty_cache()
    return result


def _train_coral(spec, train, dev, test, excluded, device) -> dict:
    job = "sst5-coral"
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    n_labels = len(spec["dataset"]["labels"])

    def build(batch_size: int):
        encoder = AutoModel.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
        bias_init = coral_bias_init([gold for _, gold in train], n_labels)
        model = CoralClassifier(encoder, encoder.config.hidden_size, n_labels, bias_init).to(device)
        rest = [parameter for name, parameter in model.named_parameters() if name != "bias"]
        optimizer = torch.optim.AdamW(
            [
                {"params": rest, "lr": spec["learning_rate"], "weight_decay": spec["weight_decay"]},
                {"params": [model.bias], "lr": spec["coral"]["bias_learning_rate"], "weight_decay": 0.0},
            ]
        )
        steps = max(1, (len(train) + batch_size - 1) // batch_size) * spec["epochs"]
        warmup = int(spec["warmup_ratio"] * steps)

        def lr_at(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (steps - step) / max(1, steps - warmup))

        return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    batch_size = spec["batch_size"]
    oom = False
    model, optimizer, scheduler = build(batch_size)
    levels = torch.arange(n_labels - 1, device=device)
    losses = []
    epoch = 0
    while epoch < spec["epochs"]:
        try:
            model.train()
            epoch_losses = []
            for batch in _batches(train, batch_size, spec["seed"], epoch):
                encoded = _encode(tokenizer, [text for text, _ in batch], spec["max_length"], device)
                gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                target = (gold.unsqueeze(1) > levels).float()
                optimizer.zero_grad(set_to_none=True)
                logits = model.cumulative_logits(encoded["input_ids"], encoded["attention_mask"])
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.extend(epoch_losses)
            epoch += 1
            print(f"{job} epoch {epoch} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            if oom:
                raise
            oom = True
            batch_size = spec["oom_batch_fallback"]
            print(f"{job} OOM, reloading at batch {batch_size}", flush=True)
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            model, optimizer, scheduler = build(batch_size)
            losses = []
            epoch = 0
    result = _fit_and_save(job, model, tokenizer, dev, test, spec, device, "coral", len(train), excluded, batch_size, oom, losses)
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    spec = json.loads(PROGRAM3_PATH.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(torch.cuda.get_device_name(0))
    torch.manual_seed(spec["seed"])
    train, dev, test, excluded = _load_rows(spec)
    print(f"sst5 train {len(train)} excluded {excluded} dev {len(dev)} test {len(test)}", flush=True)
    out_root = ROOT / "artifacts" / "program3"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    jobs = {"softmax_log_loss": _train_softmax, "coral": _train_coral}
    for kind in spec["heads"]:
        job = f"sst5-{kind}"
        done = out_root / job / "metrics.json"
        if done.exists():
            print(f"skip {job}", flush=True)
            results.append(json.loads(done.read_text()))
            continue
        print(f"start {job}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        try:
            results.append(jobs[kind](spec, train, dev, test, excluded, torch.device("cuda")))
        except Exception:
            failures.append({"job": job, "traceback": traceback.format_exc()})
            print(failures[-1]["traceback"], flush=True)
            torch.cuda.empty_cache()
    payload = {"protocol_id": spec["protocol_id"], "runs": results, "failures": failures}
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

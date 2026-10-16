"""Program 2 paired decoder: closed-set label readout, no generation."""

from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature, schema_valid
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature
from bakeoff.train_ettin import _batches, _metrics, _train_rows, decision_loss

DECODER_PATH = ROOT / "program2_decoder.json"


def _continuation_ids(tokenizer, text: str, label: str, spec: dict) -> tuple[list[int], int, bool]:
    """Concatenate ids so a label cannot merge into the last text token."""
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    truncated = len(text_ids) > spec["max_prompt_tokens"]
    if truncated:
        text_ids = text_ids[: spec["max_prompt_tokens"]]
    newline_ids = tokenizer.encode(spec["prompt_suffix"], add_special_tokens=False)
    label_ids = tokenizer.encode(label, add_special_tokens=False)
    if not newline_ids or not label_ids:
        raise RuntimeError(f"empty prompt suffix or label ids for {label!r}")
    ids: list[int] = []
    if spec["prepend_bos"]:
        ids.append(tokenizer.bos_token_id)
    ids.extend(text_ids)
    ids.extend(newline_ids)
    prompt_len = len(ids)
    ids.extend(label_ids)
    return ids, prompt_len, truncated


def _mean_continuation_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean log-probability of tokens at positions [prompt_len, length)."""
    shift_ids = input_ids[:, 1:]
    positions = torch.arange(shift_ids.shape[1], device=logits.device).unsqueeze(0)
    token_index = positions + 1
    mask = (token_index >= prompt_lens.unsqueeze(1)) & (token_index < lengths.unsqueeze(1))
    # A full float logit tensor for a long batch is several GB. Short rows use one kernel.
    float_bytes = shift_ids.shape[0] * shift_ids.shape[1] * logits.shape[-1] * 4
    if float_bytes <= 1_200_000_000:
        shift_logits = logits[:, :-1, :].float()
        chosen = shift_logits.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)
        log_prob = chosen - torch.logsumexp(shift_logits, dim=-1)
        total = (log_prob * mask).sum(dim=1)
        count = mask.sum(dim=1)
    else:
        total = torch.zeros(input_ids.shape[0], device=logits.device, dtype=torch.float32)
        count = torch.zeros_like(total)
        for start in range(0, shift_ids.shape[1], 16):
            stop = min(start + 16, shift_ids.shape[1])
            step = logits[:, start:stop, :].float()
            chosen = step.gather(2, shift_ids[:, start:stop].unsqueeze(-1)).squeeze(-1)
            log_prob = chosen - torch.logsumexp(step, dim=-1)
            block_mask = mask[:, start:stop]
            total = total + (log_prob * block_mask).sum(dim=1)
            count = count + block_mask.sum(dim=1)
    if torch.any(count <= 0):
        raise RuntimeError("a continuation had no scored tokens")
    return total / count


def _loss_grad(scores: torch.Tensor, gold: torch.Tensor, kind: str) -> torch.Tensor:
    """Batch-mean gradient of decision_loss with respect to the label scores."""
    probs = torch.softmax(scores, dim=-1)
    batch = scores.shape[0]
    if kind == "log_loss":
        grad = probs.clone()
        grad[torch.arange(batch, device=scores.device), gold] -= 1.0
        return grad / batch
    if kind == "brier":
        one_hot = torch.zeros_like(probs)
        one_hot[torch.arange(batch, device=scores.device), gold] = 1.0
        delta = probs - one_hot
        dot = (delta * probs).sum(dim=-1, keepdim=True)
        return (2.0 * probs * (delta - dot)) / batch
    raise ValueError(kind)


def _pack(tokenizer, texts: list[str], labels: list[str], spec: dict, device: torch.device):
    sequences = []
    prompt_lens = []
    truncated = 0
    for text in texts:
        text_truncated = False
        for label in labels:
            ids, prompt_len, was_truncated = _continuation_ids(tokenizer, text, label, spec)
            sequences.append(ids)
            prompt_lens.append(prompt_len)
            text_truncated = text_truncated or was_truncated
        truncated += int(text_truncated)
    width = max(len(sequence) for sequence in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    lengths = torch.zeros(len(sequences), dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention[row, : len(sequence)] = 1
        lengths[row] = len(sequence)
    prompt = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    return input_ids, attention, prompt, lengths, truncated


def _score_chunk(model, tokenizer, texts: list[str], labels: list[str], spec: dict, device: torch.device):
    input_ids, attention, prompt_lens, lengths, truncated = _pack(tokenizer, texts, labels, spec, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
    scores = _mean_continuation_logprob(logits, input_ids, prompt_lens, lengths)
    return scores.view(len(texts), len(labels)), truncated


def _score_texts(model, tokenizer, texts: list[str], labels: list[str], spec: dict, device, chunk_size: int):
    parts = []
    truncated = 0
    for start in range(0, len(labels), chunk_size):
        chunk = labels[start : start + chunk_size]
        scores, n_truncated = _score_chunk(model, tokenizer, texts, chunk, spec, device)
        parts.append(scores)
        truncated += n_truncated
    # Truncation is a property of the text, so each chunk would count it again.
    n_chunks = (len(labels) + chunk_size - 1) // chunk_size
    if truncated % n_chunks != 0:
        raise RuntimeError("truncation counts disagreed across label chunks")
    return torch.cat(parts, dim=1), truncated // n_chunks


def _apply_loss(model, tokenizer, texts, labels, gold, kind, spec, device, chunk_size: int) -> torch.Tensor:
    """Backpropagate a proper score without keeping every label chunk's logits."""
    with torch.no_grad():
        scores, _ = _score_texts(model, tokenizer, texts, labels, spec, device, chunk_size)
    grad = _loss_grad(scores, gold, kind)
    loss_value = decision_loss(scores, gold, kind).detach()
    offset = 0
    for start in range(0, len(labels), chunk_size):
        chunk = labels[start : start + chunk_size]
        chunk_scores, _ = _score_chunk(model, tokenizer, texts, chunk, spec, device)
        width = chunk_scores.shape[1]
        if offset == 0 and not getattr(_apply_loss, "drift_logged", False):
            drift = (chunk_scores.detach() - scores[:, :width]).abs().max().item()
            print(f"decoder score drift {drift:.6f}", flush=True)
            _apply_loss.drift_logged = True
            if not np.isfinite(drift) or drift > 0.1:
                raise RuntimeError(f"decoder scoring passes diverged by {drift}")
        chunk_scores.backward(grad[:, offset : offset + width])
        offset += width
    return loss_value


def _score_items(model, tokenizer, items, labels, spec, device, batch_size: int, chunk_size: int):
    model.eval()
    scores = []
    gold = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            row_scores, n_truncated = _score_texts(
                model, tokenizer, [item.text for item in chunk], labels, spec, device, chunk_size
            )
            scores.append(row_scores.detach().cpu().numpy().astype(np.float64))
            gold.extend(item.gold_index for item in chunk)
            truncated += n_truncated
    return np.concatenate(scores, axis=0), np.asarray(gold, dtype=np.int64), truncated


def _latency(model, tokenizer, items, labels, spec, device, warmup: int) -> dict:
    model.eval()
    timed = []
    with torch.inference_mode():
        for item in items["dev"][:warmup]:
            _score_texts(model, tokenizer, [item.text], labels, spec, device, spec["label_chunk_size"])
            torch.cuda.synchronize()
        for item in items["test"]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            _score_texts(model, tokenizer, [item.text], labels, spec, device, spec["label_chunk_size"])
            torch.cuda.synchronize()
            timed.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(timed, dtype=np.float64)
    return {
        "n": int(array.size),
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
    }


def _build(spec, n_labels_unused, device, batch_size: int, chunk_size: int):
    del n_labels_unused
    model = AutoModelForCausalLM.from_pretrained(spec["model"], revision=spec["revision"], dtype=torch.float32)
    model.config.use_cache = False
    model.to(device)
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cuda"}:
        raise RuntimeError(f"Ettin decoder parameters are on {devices}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    return model, optimizer, batch_size, chunk_size


def _train_one(job: str, dataset: str, kind: str, spec: dict, protocol: dict, materialized, device, rows=None, out_root=None) -> dict:
    labels = protocol["datasets"][dataset]["labels"]
    excluded = set(materialized["splits"][dataset]["dev"]["rows"])
    # The size sweep passes its own training rows and output folder; Program 2 uses the defaults.
    rows = rows if rows is not None else _train_rows(dataset, protocol, excluded)
    if any(gold < 0 or gold >= len(labels) for _, gold in rows):
        raise RuntimeError(f"{dataset} gold outside the frozen label set")
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    if tokenizer.bos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("decoder tokenizer is missing bos or pad")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(torch.cuda.get_device_name(0))
    # Fail before any update if a frozen label is not a continuation.
    for label in labels:
        _continuation_ids(tokenizer, "continuation check", label, spec)

    _apply_loss.drift_logged = False
    # Whether batch 8 fits on Banking77 depends on what else holds GPU memory, so the size is pinned per dataset.
    pinned = spec.get("per_dataset", {}).get(dataset, {})
    batch_size = pinned.get("batch_size", spec["batch_size"])
    chunk_size = pinned.get("label_chunk_size", spec["label_chunk_size"])
    oom_fallback = False
    model, optimizer, batch_size, chunk_size = _build(spec, len(labels), device, batch_size, chunk_size)
    steps_per_epoch = max(1, (len(rows) + batch_size - 1) // batch_size)
    total = steps_per_epoch * spec["epochs"]
    warmup = int(spec["warmup_ratio"] * total)

    def lr_at(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        remain = max(1, total - warmup)
        return max(0.0, (total - step) / remain)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    losses: list[float] = []
    epoch = 0
    while epoch < spec["epochs"]:
        try:
            model.train()
            epoch_losses = []
            for batch in _batches(rows, batch_size, spec["seed"], epoch):
                texts = [text for text, _ in batch]
                gold = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = _apply_loss(model, tokenizer, texts, labels, gold, kind, spec, device, chunk_size)
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                epoch_losses.append(float(loss.detach().cpu()))
                if len(epoch_losses) % 50 == 0:
                    # Varying sequence lengths fragment the allocator's cache until the Windows driver spills
                    # into system memory, which slows a step tenfold. Releasing the cache does not change a number.
                    torch.cuda.empty_cache()
                if len(epoch_losses) % 100 == 0:
                    print(f"{job} epoch {epoch + 1} step {len(epoch_losses)} loss {epoch_losses[-1]:.4f}", flush=True)
            losses.extend(epoch_losses)
            epoch += 1
            print(f"{job} epoch {epoch} mean_loss {np.mean(epoch_losses):.4f}", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            cuda_failure = isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA" in str(error)
            if not cuda_failure:
                raise
            if oom_fallback:
                raise
            oom_fallback = True
            batch_size = spec["oom_batch_fallback"]
            chunk_size = spec["oom_label_chunk_fallback"]
            print(f"{job} OOM, reloading at batch {batch_size} chunk {chunk_size}", flush=True)
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            model, optimizer, batch_size, chunk_size = _build(spec, len(labels), device, batch_size, chunk_size)
            steps_per_epoch = max(1, (len(rows) + batch_size - 1) // batch_size)
            total = steps_per_epoch * spec["epochs"]
            warmup = int(spec["warmup_ratio"] * total)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
            losses = []
            epoch = 0

    dev_scores, dev_gold, _ = _score_items(
        model,
        tokenizer,
        materialized["items"][dataset]["dev"],
        labels,
        spec,
        device,
        spec["eval_batch_size"],
        chunk_size,
    )
    fitted = fit_temperature(dev_scores, dev_gold)
    test_scores, test_gold, n_truncated = _score_items(
        model,
        tokenizer,
        materialized["items"][dataset]["test"],
        labels,
        spec,
        device,
        spec["eval_batch_size"],
        chunk_size,
    )
    latency = _latency(
        model,
        tokenizer,
        materialized["items"][dataset],
        labels,
        spec,
        device,
        load_protocol()["latency"]["warmup"],
    )
    out_dir = (out_root or ROOT / "artifacts" / "program2-decoder") / job
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")
    result = {
        "dataset": dataset,
        "loss": kind,
        "model": spec["model"],
        "revision": spec["revision"],
        "train_n": len(rows),
        "excluded_dev_n": len(excluded),
        "epochs": spec["epochs"],
        "batch_size": batch_size,
        "label_chunk_size": chunk_size,
        "oom_fallback": oom_fallback,
        "mean_train_loss": float(np.mean(losses)) if losses else None,
        "temperature": fitted,
        "scaled": _metrics(test_scores, test_gold, fitted["temperature"]),
        "unscaled": _metrics(test_scores, test_gold, 1.0),
        "n_truncated_test": n_truncated,
        "latency": latency,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "device": torch.cuda.get_device_name(0),
    }
    if not all(schema_valid(row, len(labels)) for row in apply_temperature(test_scores, fitted["temperature"])):
        raise RuntimeError("decoder readout was not a simplex")
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out_dir / "test_scores.npz", scores=test_scores, gold=test_gold)
    print(
        f"{job} test brier_scaled {result['scaled']['brier']:.4f} acc {result['scaled']['accuracy']:.4f} "
        f"p50 {latency['p50_ms']:.1f} ms",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return result


def main(seed: int | None = None) -> None:
    spec = json.loads(DECODER_PATH.read_text())
    if spec["generate"] or spec["rationale"]:
        raise RuntimeError("decoder protocol must not generate a rationale")
    protocol = load_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    device = torch.device("cuda")
    torch.manual_seed(spec["seed"])
    results = []
    failures = []
    out_root = ROOT / "artifacts" / "program2-decoder"
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


def probe() -> None:
    """One Banking77 backward at the frozen batch. Does not write metrics."""
    spec = json.loads(DECODER_PATH.read_text())
    protocol = load_protocol()
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    labels = protocol["datasets"]["banking77"]["labels"]
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    for label in labels:
        _continuation_ids(tokenizer, "a film that is not negative", label, spec)
        _continuation_ids(tokenizer, "negative", "negative", spec)
    model, optimizer, _, _ = _build(spec, len(labels), device, spec["batch_size"], spec["label_chunk_size"])
    texts = [
        "I want a refund",
        "activate my card",
        "the card was charged twice",
        "where is my pin",
        "lost my card yesterday",
        "apple pay is not working",
        "what is the age limit",
        "atm ate my cash",
    ]
    gold = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats()
    _apply_loss.drift_logged = False
    started = time.perf_counter()
    loss = _apply_loss(model, tokenizer, texts, labels, gold, "log_loss", spec, device, spec["label_chunk_size"])
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"probe step_s {elapsed:.2f}", flush=True)
    print(
        f"probe loss {float(loss):.4f} peak_mib {torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        probe()
    elif len(sys.argv) > 2 and sys.argv[1] == "--seed":
        main(int(sys.argv[2]))
    else:
        main()

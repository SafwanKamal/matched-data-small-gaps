"""Program 4: one frozen SST-2 encode, a Choice head and two Noul sigmoids."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature, score_probabilities
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature
from bakeoff.train_ettin import PROGRAM2_PATH, EncoderClassifier, _encode

PROGRAM4_PATH = ROOT / "program4.json"


class SigmoidPair(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.head = nn.Linear(hidden, 2)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled)


def _pool_split(model, tokenizer, items, max_length: int, device, batch_size: int = 64):
    model.eval()
    pooled = []
    gold = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            encoded = _encode(tokenizer, [item.text for item in chunk], max_length, device)
            hidden = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled.append(vector.detach().cpu())
            gold.extend(item.gold_index for item in chunk)
    return torch.cat(pooled, dim=0), np.asarray(gold, dtype=np.int64)


def _choice_scores(head: nn.Linear, pooled: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return head(pooled).numpy().astype(np.float64)


def _train_sigmoids(pooled: torch.Tensor, gold: np.ndarray, spec: dict) -> nn.Linear:
    torch.manual_seed(spec["seed"])
    head = nn.Linear(pooled.shape[1], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    gold_t = torch.tensor(gold, dtype=torch.long)
    order = np.arange(len(gold))
    for epoch in range(spec["epochs"]):
        rng = np.random.default_rng(spec["seed"] + epoch)
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), spec["batch_size"]):
            index = order[start : start + spec["batch_size"]]
            batch = pooled[index]
            target = F.one_hot(gold_t[index], num_classes=2).float()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(batch), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"noul epoch {epoch + 1} mean_loss {np.mean(losses):.4f}", flush=True)
    return head


def _renormalized_sigmoid_scores(head: nn.Linear, pooled: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        probs = torch.sigmoid(head(pooled))
        total = probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        simplex = probs / total
        return torch.log(simplex.clamp(min=1e-12)).numpy().astype(np.float64)


def _metrics(scores: np.ndarray, gold: np.ndarray, temperature: float) -> dict:
    probs = apply_temperature(scores, temperature)
    scored = score_probabilities(probs, gold)
    scored.pop("risk_coverage", None)
    scored.pop("ece_bins", None)
    return scored


def _latency(model, noul: nn.Linear, tokenizer, items, max_length: int, device, warmup: int) -> dict:
    model.eval()
    noul = noul.to(device)

    def once(text: str) -> None:
        encoded = _encode(tokenizer, [text], max_length, device)
        hidden = model.encoder(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        model.head(pooled)
        noul(pooled)

    timed_one = []
    timed_two = []
    with torch.inference_mode():
        for item in items["dev"][:warmup]:
            once(item.text)
            torch.cuda.synchronize()
        for item in items["test"]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            once(item.text)
            torch.cuda.synchronize()
            timed_one.append((time.perf_counter() - started) * 1000.0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            once(item.text)
            once(item.text)
            torch.cuda.synchronize()
            timed_two.append((time.perf_counter() - started) * 1000.0)
    one = np.asarray(timed_one, dtype=np.float64)
    two = np.asarray(timed_two, dtype=np.float64)
    return {
        "one_forward_p50_ms": float(np.percentile(one, 50, method="linear")),
        "one_forward_p95_ms": float(np.percentile(one, 95, method="linear")),
        "two_forward_p50_ms": float(np.percentile(two, 50, method="linear")),
        "two_forward_p95_ms": float(np.percentile(two, 95, method="linear")),
        "n": int(one.size),
    }


def main() -> None:
    spec = json.loads(PROGRAM4_PATH.read_text())
    protocol = load_protocol()
    encoder_spec = json.loads(PROGRAM2_PATH.read_text())
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    trunk = ROOT / spec["trunk"]
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(trunk / "encoder")
    encoder = AutoModel.from_pretrained(trunk / "encoder", dtype=torch.bfloat16)
    labels = protocol["datasets"]["sst2"]["labels"]
    model = EncoderClassifier(encoder, encoder.config.hidden_size, len(labels)).to(device)
    saved = torch.load(trunk / "head.pt", map_location="cpu", weights_only=False)
    model.head.load_state_dict(saved["head"])
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.eval()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    items = materialized["items"]["sst2"]
    from bakeoff.train_ettin import _train_rows

    excluded = set(materialized["splits"]["sst2"]["dev"]["rows"])
    raw_rows = _train_rows("sst2", protocol, excluded)
    class _Row:
        def __init__(self, text, gold_index):
            self.text = text
            self.gold_index = gold_index

    train_items = [_Row(text, gold) for text, gold in raw_rows]
    print(f"pooling sst2 train {len(train_items)}", flush=True)
    train_pool, train_gold = _pool_split(model, tokenizer, train_items, encoder_spec["max_length"], device)
    dev_pool, dev_gold = _pool_split(model, tokenizer, items["dev"], encoder_spec["max_length"], device)
    test_pool, test_gold = _pool_split(model, tokenizer, items["test"], encoder_spec["max_length"], device)
    choice_head = model.head.cpu()
    choice_test = _choice_scores(choice_head, test_pool)
    choice_dev = _choice_scores(choice_head, dev_pool)
    noul = _train_sigmoids(train_pool, train_gold, spec)
    noul_test = _renormalized_sigmoid_scores(noul, test_pool)
    noul_dev = _renormalized_sigmoid_scores(noul, dev_pool)
    choice_t = fit_temperature(choice_dev, dev_gold)
    noul_t = fit_temperature(noul_dev, dev_gold)
    choice_probs = apply_temperature(choice_test, choice_t["temperature"])
    noul_probs = apply_temperature(noul_test, noul_t["temperature"])
    agreement = float(np.mean(np.argmax(choice_probs, axis=1) == np.argmax(noul_probs, axis=1)))
    positive_gap = float(np.mean(np.abs(choice_probs[:, 1] - noul_probs[:, 1])))
    model.to(device)
    latency = _latency(model, noul, tokenizer, items, encoder_spec["max_length"], device, protocol["latency"]["warmup"])
    result = {
        "protocol_id": spec["protocol_id"],
        "trunk_frozen": True,
        "train_n": len(train_items),
        "choice": {
            "temperature": choice_t,
            "scaled": _metrics(choice_test, test_gold, choice_t["temperature"]),
        },
        "noul_renormalized": {
            "temperature": noul_t,
            "scaled": _metrics(noul_test, test_gold, noul_t["temperature"]),
        },
        "argmax_agreement": agreement,
        "mean_abs_positive_gap": positive_gap,
        "latency": latency,
        "device": torch.cuda.get_device_name(0),
    }
    out = ROOT / "artifacts" / "program4" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": noul.state_dict()}, out.parent / "noul_head.pt")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"choice brier {result['choice']['scaled']['brier']:.4f} "
        f"noul brier {result['noul_renormalized']['scaled']['brier']:.4f} "
        f"agreement {agreement:.4f} one {latency['one_forward_p50_ms']:.2f} ms "
        f"two {latency['two_forward_p50_ms']:.2f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Context-wrapping pilot. Frozen rules are in context_pilot_protocol.json. Runs on the processor and the Jev API."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace

import numpy as np

from bakeoff.data import Item, load_items
from bakeoff.metrics import apply_temperature, score_probabilities
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

PILOT_PATH = ROOT / "context_pilot_protocol.json"
OUT = ROOT / "artifacts" / "context-pilot"


def _pools(spec: dict) -> dict:
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    from bakeoff.train_ettin import _train_rows

    banking_labels = protocol["datasets"]["banking77"]["labels"]
    excluded = set(materialized["splits"]["banking77"]["dev"]["rows"])
    banking_pool = _train_rows("banking77", protocol, excluded)

    from datasets import load_dataset

    from bakeoff.suite import load_suite, load_suite_protocol

    clinc_spec = load_suite_protocol()["datasets"]["clinc150"]
    raw = load_dataset(clinc_spec["hf_id"], clinc_spec["config"], revision=clinc_spec["revision"])["train"]
    names = raw.features["intent"].names
    oos = names.index("oos")
    remap = {index: position for position, index in enumerate(i for i in range(len(names)) if i != oos)}
    clinc_pool = [(example["text"], remap[example["intent"]]) for example in raw if example["intent"] != oos]
    clinc = load_suite(["clinc150"])["clinc150"]
    return {
        "banking77": {"labels": banking_labels, "items": materialized["items"]["banking77"]["test"], "pool": banking_pool},
        "clinc150": {"labels": clinc["labels"], "items": clinc["test"][:500], "pool": clinc_pool},
    }


def build(spec: dict) -> dict:
    """Wrapped items per dataset and condition. Distractors are drawn once per item and shared by all conditions."""
    pools = _pools(spec)
    children = np.random.SeedSequence(spec["seed"]).spawn(len(pools))
    out = {}
    for (dataset, block), child in zip(pools.items(), children):
        rng = np.random.default_rng(child)
        labels = block["labels"]
        by_label: dict[int, list[str]] = {}
        for text, gold in block["pool"]:
            by_label.setdefault(gold, []).append(text)
        wrapped = {name: [] for name in spec["conditions"]}
        for item in block["items"]:
            others = [label for label in sorted(by_label) if label != item.gold_index]
            chosen = rng.choice(others, size=8, replace=False).tolist()
            fields = {"target": item.text}
            for position, label in enumerate(chosen, start=1):
                texts = by_label[label]
                fields[f"d{position}"] = texts[int(rng.integers(0, len(texts)))]
                fields[f"l{position}"] = labels[label]
            for name, template in spec["conditions"].items():
                wrapped[name].append(replace(item, text=template.format(**fields)))
        out[dataset] = {"labels": labels, "conditions": wrapped}
    return out


def _summary(probs: np.ndarray, gold: np.ndarray, clean_right: np.ndarray | None) -> dict:
    scored = score_probabilities(probs, gold)
    right = probs.argmax(axis=1) == gold
    row = {
        "n": int(len(gold)),
        "accuracy": float(right.mean()),
        "brier": scored["brier"],
        "ece": scored["ece"],
        "mean_top_probability": float(probs.max(axis=1).mean()),
    }
    if clean_right is not None:
        row["broken"] = float((clean_right & ~right).sum() / max(1, clean_right.sum()))
        row["repaired"] = float((~clean_right & right).sum() / max(1, (~clean_right).sum()))
    return row


def _report(system: str, dataset: str, scores: dict, temperature: float, gold: np.ndarray, extra: dict | None = None) -> dict:
    clean = apply_temperature(scores["clean"], temperature)
    clean_right = clean.argmax(axis=1) == gold
    rows = {}
    for name, values in scores.items():
        rows[name] = _summary(apply_temperature(values, temperature), gold, None if name == "clean" else clean_right)
        if extra and name in extra:
            rows[name].update(extra[name])
        print(
            f"{system:24s} {dataset:10s} {name:26s} acc {rows[name]['accuracy']:.3f} brier {rows[name]['brier']:.3f} "
            f"top {rows[name]['mean_top_probability']:.3f} broken {rows[name].get('broken', 0.0):.3f}",
            flush=True,
        )
    return {"temperature": temperature, "conditions": rows}


def _save(system: str, dataset: str, block: dict) -> None:
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"protocol_id": json.loads(PILOT_PATH.read_text())["protocol_id"], "systems": {}}
    summary["systems"].setdefault(system, {})[dataset] = block
    path.write_text(json.dumps(summary, indent=2) + "\n")


def run_jev(spec: dict, data: dict) -> None:
    import httpx

    from bakeoff import jev

    base = json.loads(jev.JEV_PROTOCOL_PATH.read_text())
    variants = {
        "jev-generic": base["request"]["instructions"],
        "jev-current": "Which label best describes the customer's current message? Ignore earlier messages that are already resolved.",
    }
    with httpx.Client(headers={"Authorization": f"Bearer {jev._api_key(base)}"}, timeout=90.0) as client:
        for system, instructions in variants.items():
            request = copy.deepcopy(base)
            request["request"]["instructions"] = instructions
            for dataset, block in data.items():
                gold = np.asarray([item.gold_index for item in block["conditions"]["clean"]], dtype=np.int64)
                scores = {}
                for name, items in block["conditions"].items():
                    path = OUT / system / f"{dataset}__{name}.jsonl"
                    for _ in range(4):
                        rows = jev._collect(client, request, items, block["labels"], dataset, name, path)
                        errors = [row for row in rows if row["error"] is not None]
                        if not errors:
                            break
                    if errors:
                        raise RuntimeError(f"{system} {dataset} {name}: {len(errors)} rows still in error")
                    scores[name] = np.asarray([row["scores"] for row in rows], dtype=np.float64)
                _save(system, dataset, _report(system, dataset, scores, 1.0, gold))


def run_cpu(spec: dict, data: dict) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from bakeoff.systems import GLiClassChoice
    from bakeoff.train_ettin import EncoderClassifier, _score_items

    torch.set_num_threads(10)
    protocol = load_protocol()
    banking = data["banking77"]
    gold = np.asarray([item.gold_index for item in banking["conditions"]["clean"]], dtype=np.int64)

    artifact = ROOT / "artifacts" / "program2" / "banking77-log_loss"
    encoder_spec = json.loads((ROOT / "program2.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(artifact / "encoder")
    encoder = AutoModel.from_pretrained(artifact / "encoder", dtype=torch.float32)
    model = EncoderClassifier(encoder, encoder.config.hidden_size, len(banking["labels"]))
    model.head.load_state_dict(torch.load(artifact / "head.pt", map_location="cpu", weights_only=True)["head"])
    model.eval()
    scores = {}
    extra = {}
    for name, items in banking["conditions"].items():
        values, _, truncated = _score_items(model, tokenizer, items, encoder_spec, torch.device("cpu"), 32)
        scores[name] = values
        extra[name] = {"share_over_128_tokens": truncated / len(items)}
    saved = np.load(artifact / "test_scores.npz")["scores"]
    check = {"max_abs_score_difference_vs_gpu": float(np.abs(scores["clean"] - saved).max()), "argmax_changes_vs_gpu": int((scores["clean"].argmax(axis=1) != saved.argmax(axis=1)).sum())}
    print(f"encoder cpu vs gpu on the clean items: {check}", flush=True)
    temperature = float(json.loads((artifact / "metrics.json").read_text())["temperature"]["temperature"])
    block = _report("ettin-encoder-log_loss", "banking77", scores, temperature, gold, extra)
    block["cpu_vs_gpu_check"] = check
    _save("ettin-encoder-log_loss", "banking77", block)
    del model, encoder

    system = GLiClassChoice(protocol["systems"]["gliclass-base-v3.0"], device="cpu")
    system.load()
    temperatures = {
        "banking77": fit_temperature(
            *(lambda rows: (np.asarray([r["scores"] for r in rows], dtype=np.float64), np.asarray([r["gold_index"] for r in rows], dtype=np.int64)))(
                [json.loads(line) for line in (ROOT / "artifacts" / "decisions" / "gliclass-base-v3.0__banking77__dev.jsonl").open(encoding="utf-8")]
            )
        )["temperature"],
        "clinc150": json.loads((ROOT / "artifacts" / "suite" / "gliclass-base-v3.0" / "summary.json").read_text())["datasets"]["clinc150"]["temperature"]["temperature"],
    }
    for dataset, block in data.items():
        gold = np.asarray([item.gold_index for item in block["conditions"]["clean"]], dtype=np.int64)
        scores = {}
        for name, items in block["conditions"].items():
            cache = OUT / "gliclass-base-v3.0" / f"{dataset}__{name}.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                scores[name] = np.load(cache)["scores"]
                continue
            print(f"gliclass cpu {dataset} {name}", flush=True)
            scores[name] = np.stack([system.score(item.text, block["labels"])[0] for item in items], axis=0)
            np.savez_compressed(cache, scores=scores[name])
        _save("gliclass-base-v3.0", dataset, _report("gliclass-base-v3.0", dataset, scores, temperatures[dataset], gold))


def run_gpu(spec: dict, data: dict) -> None:
    """The Ettin decoder and Qwen3-4B on the Banking77 pilot items. Added once the GPU was free."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bakeoff.readout import READOUT_PATH, IndexReadout
    from bakeoff.suite import load_suite_protocol
    from bakeoff.train_ettin_decoder import _score_items

    banking = data["banking77"]
    labels = banking["labels"]
    gold = np.asarray([item.gold_index for item in banking["conditions"]["clean"]], dtype=np.int64)

    artifact = ROOT / "artifacts" / "program2-decoder" / "banking77-log_loss"
    saved = json.loads((artifact / "metrics.json").read_text())
    decoder_spec = json.loads((ROOT / "program2_decoder.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(artifact / "model")
    model = AutoModelForCausalLM.from_pretrained(artifact / "model", dtype=torch.float32)
    model.config.use_cache = False
    model.to("cuda")
    scores, extra = {}, {}
    for name, items in banking["conditions"].items():
        values, _, truncated = _score_items(model, tokenizer, items, labels, decoder_spec, torch.device("cuda"), saved["batch_size"], saved["label_chunk_size"])
        scores[name] = values
        extra[name] = {"share_over_128_prompt_tokens": truncated / len(items)}
        torch.cuda.empty_cache()
    _save("ettin-decoder-log_loss", "banking77", _report("ettin-decoder-log_loss", "banking77", scores, float(saved["temperature"]["temperature"]), gold, extra))
    del model
    torch.cuda.empty_cache()

    system_spec = load_suite_protocol()["systems"]["qwen3-4b-index_prior"]
    readout_spec = json.loads(READOUT_PATH.read_text())
    readout_spec["model"] = {"hf_id": system_spec["hf_id"], "revision": system_spec["revision"]}
    suite_dir = ROOT / "artifacts" / "suite" / "qwen3-4b-index_prior"
    prior = np.load(suite_dir / "banking77_full__dev.npz")["index"].mean(axis=0)
    temperature = float(json.loads((suite_dir / "summary.json").read_text())["datasets"]["banking77_full"]["temperature"]["temperature"])
    readout = IndexReadout(readout_spec, prefix_batch=system_spec["prefix_batch"])
    scores = {}
    for name, items in banking["conditions"].items():
        cache = OUT / "qwen3-4b-index_prior" / f"banking77__{name}.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            scores[name] = np.load(cache)["scores"]
            continue
        print(f"qwen3-4b banking77 {name}", flush=True)
        scores[name] = np.stack([readout.score(item.text, labels)[0] for item in items], axis=0) - prior
        np.savez_compressed(cache, scores=scores[name])
    _save("qwen3-4b-index_prior", "banking77", _report("qwen3-4b-index_prior", "banking77", scores, temperature, gold))


def main() -> None:
    spec = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    data = build(spec)
    examples = {dataset: {name: items[0].text for name, items in block["conditions"].items()} for dataset, block in data.items()}
    (OUT / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    if command in ("jev", "all"):
        run_jev(spec, data)
    if command in ("cpu", "all"):
        run_cpu(spec, data)
    if command == "gpu":
        run_gpu(spec, data)
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()

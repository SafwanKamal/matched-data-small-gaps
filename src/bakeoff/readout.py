"""Zero-shot index readout for a causal LLM. Frozen rules are in readout_protocol.json."""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

from bakeoff.data import load_items
from bakeoff.metrics import apply_temperature, schema_valid, score_probabilities
from bakeoff.protocol import ROOT, load_protocol
from bakeoff.temperature import fit_temperature

READOUT_PATH = ROOT / "readout_protocol.json"
OUT = ROOT / "artifacts" / "readout"
VARIANTS = ["label_string", "index", "index_prior"]


class IndexReadout:
    def __init__(self, spec: dict, prefix_batch: int = 64):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.prefix_batch = prefix_batch

        model = spec["model"]
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(model["hf_id"], revision=model["revision"])
        if model.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            # NF4 weights with a bfloat16 compute type, for a model too large for the card at 16 bits.
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(model["hf_id"], revision=model["revision"], quantization_config=quantization, device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model["hf_id"], revision=model["revision"], dtype=torch.bfloat16
            ).to("cuda")
        self.model.eval()
        self.digit_ids = {}
        for digit in "0123456789":
            ids = self.tokenizer.encode(digit, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"digit {digit} is not one token")
            self.digit_ids[digit] = ids[0]
        self.prefill_ids = self.tokenizer.encode(spec["prompt"]["prefill"], add_special_tokens=False)

    def codes(self, n: int) -> list[str]:
        width = len(str(n))
        return [str(position).zfill(width) for position in range(1, n + 1)]

    def prompt_ids(self, text: str, labels: list[str]) -> list[int]:
        prompt = self.spec["prompt"]
        codes = self.codes(len(labels))
        options = "\n".join(prompt["option_line"].format(code=code, label=label) for code, label in zip(codes, labels))
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["user"].format(options=options, text=text)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer.encode(chat, add_special_tokens=False) + self.prefill_ids

    def _next_log_probs(self, sequences: list[list[int]]) -> torch.Tensor:
        """Log-softmax over the vocabulary at the last position of equal-length sequences."""
        input_ids = torch.tensor(sequences, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hidden = self.model.base_model(input_ids=input_ids).last_hidden_state[:, -1]
            logits = self.model.get_output_embeddings()(hidden).float()
            return torch.log_softmax(logits, dim=-1)

    def score(self, text: str, labels: list[str]) -> tuple[np.ndarray, bool]:
        base = self.prompt_ids(text, labels)
        truncated = len(base) > self.spec["context_limit"]
        codes = self.codes(len(labels))
        width = len(codes[0])
        total = {code: 0.0 for code in codes}
        prefixes = [""]
        for depth in range(width):
            sequences = [base + [self.digit_ids[digit] for digit in prefix] for prefix in prefixes]
            # Prefix sequences are forwarded in chunks so a long option list fits beside a larger model.
            log_probs = torch.cat(
                [self._next_log_probs(sequences[start : start + self.prefix_batch]) for start in range(0, len(sequences), self.prefix_batch)],
                dim=0,
            )
            row_of = {prefix: row for row, prefix in enumerate(prefixes)}
            for code in codes:
                total[code] += float(log_probs[row_of[code[:depth]], self.digit_ids[code[depth]]])
            prefixes = sorted({code[: depth + 1] for code in codes})
        return np.asarray([total[code] for code in codes], dtype=np.float64), truncated

    def close(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


class BracketReadout(IndexReadout):
    """Unpadded codes closed by a bracket, renormalised over the continuations some option allows."""

    def __init__(self, spec: dict, prefix_batch: int = 64):
        super().__init__(spec, prefix_batch)
        ids = self.tokenizer.encode("]", add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError("closing bracket is not one token")
        self.close_id = ids[0]

    def codes(self, n: int) -> list[str]:
        return [str(position) for position in range(1, n + 1)]

    def score(self, text: str, labels: list[str]) -> tuple[np.ndarray, bool]:
        base = self.prompt_ids(text, labels)
        truncated = len(base) > self.spec["context_limit"]
        paths = [tuple(self.digit_ids[digit] for digit in code) + (self.close_id,) for code in self.codes(len(labels))]
        children: dict[tuple, set[int]] = {}
        for path in paths:
            for depth in range(len(path)):
                children.setdefault(path[:depth], set()).add(path[depth])
        total = [0.0] * len(paths)
        for depth in range(max(len(path) for path in paths)):
            nodes = sorted(node for node, allowed in children.items() if len(node) == depth and len(allowed) > 1)
            if not nodes:
                continue
            sequences = [base + list(node) for node in nodes]
            log_probs = torch.cat(
                [self._next_log_probs(sequences[start : start + self.prefix_batch]) for start in range(0, len(sequences), self.prefix_batch)],
                dim=0,
            )
            row_of = {node: row for row, node in enumerate(nodes)}
            norms = {node: float(torch.logsumexp(log_probs[row_of[node], sorted(children[node])], dim=0)) for node in nodes}
            for index, path in enumerate(paths):
                node = path[:depth]
                if len(path) > depth and node in row_of:
                    total[index] += float(log_probs[row_of[node], path[depth]]) - norms[node]
        return np.asarray(total, dtype=np.float64), truncated


def check() -> None:
    """Dev-only comparison of option-code formats for the suite's 4B model. Rules: readout_check_protocol.json."""
    from bakeoff.suite import OUT as SUITE_OUT
    from bakeoff.suite import load_suite, load_suite_protocol

    rules = json.loads((ROOT / "readout_check_protocol.json").read_text())
    system_spec = load_suite_protocol()["systems"]["qwen3-4b-index_prior"]
    spec = json.loads(READOUT_PATH.read_text())
    spec["model"] = {"hf_id": system_spec["hf_id"], "revision": system_spec["revision"]}
    out = ROOT / "artifacts" / "readout-check"
    out.mkdir(parents=True, exist_ok=True)
    suite = load_suite()
    system = BracketReadout(spec, prefix_batch=rules["prefix_batch"])
    names = ["padded", "padded_prior", "bracket", "bracket_prior"]
    table = {}
    for dataset, block in suite.items():
        items = block["dev"][:400]
        labels = block["labels"]
        gold = np.asarray([item.gold_index for item in items], dtype=np.int64)
        padded = np.load(SUITE_OUT / "qwen3-4b-index_prior" / f"{dataset}__dev.npz")["index"][: len(items)]
        cache = out / f"{dataset}__dev_bracket.npz"
        if cache.exists():
            bracket = np.load(cache)["scores"]
        else:
            print(f"bracket readout {dataset} dev n={len(items)} options={len(labels)}", flush=True)
            bracket = np.stack([system.score(item.text, labels)[0] for item in items], axis=0)
            np.savez_compressed(cache, scores=bracket, gold=gold)
        variants = {
            "padded": padded,
            "padded_prior": padded - padded.mean(axis=0),
            "bracket": bracket,
            "bracket_prior": bracket - bracket.mean(axis=0),
        }
        table[dataset] = {"n_options": len(labels), "n": len(items), "variants": {}}
        for name in names:
            fitted = fit_temperature(variants[name], gold)
            scored = _metrics(variants[name], gold, fitted["temperature"])
            table[dataset]["variants"][name] = {
                "temperature": fitted["temperature"],
                "dev_accuracy": scored["accuracy"],
                "dev_accuracy_unscaled_argmax": float((variants[name].argmax(axis=1) == gold).mean()),
                "dev_nll_over_ln_k": scored["log_loss"] / math.log(len(labels)),
                "dev_brier": scored["brier"],
            }
        row = table[dataset]["variants"]
        print(f"{dataset:15s} " + "  ".join(f"{name} acc {row[name]['dev_accuracy']:.3f} nll/lnK {row[name]['dev_nll_over_ln_k']:.3f}" for name in names), flush=True)
        means = {name: float(np.mean([table[d]["variants"][name]["dev_nll_over_ln_k"] for d in table])) for name in names}
        selected = min(names, key=lambda name: (means[name], names.index(name)))
        (out / "summary.json").write_text(json.dumps({"protocol_id": rules["protocol_id"], "complete": len(table) == len(suite), "selection_scores": means, "selected": selected, "datasets": table}, indent=2) + "\n")
    print(json.dumps({"selection_scores": means, "selected": selected}), flush=True)
    print("CHECK_DONE", flush=True)
    system.close()


def _score_split(system: IndexReadout, items, labels: list[str], timed: bool) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    scores = []
    latencies = []
    truncated = 0
    for index, item in enumerate(items, start=1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        row, was_truncated = system.score(item.text, labels)
        torch.cuda.synchronize()
        if timed:
            latencies.append((time.perf_counter() - started) * 1000.0)
        if not np.isfinite(row).all():
            raise RuntimeError(f"non-finite index score for {item.item_id}")
        scores.append(row)
        truncated += int(was_truncated)
        if index % 200 == 0:
            print(f"  index readout {index}/{len(items)}", flush=True)
    gold = np.asarray([item.gold_index for item in items], dtype=np.int64)
    return np.stack(scores, axis=0), gold, latencies, truncated


def _label_string_scores(dataset: str, split: str, items) -> np.ndarray:
    path = ROOT / "artifacts" / "decisions" / f"qwen3-0.6b__{dataset}__{split}.jsonl"
    by_id = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_id[row["item_id"]] = row
    return np.asarray([by_id[item.item_id]["scores"] for item in items], dtype=np.float64)


def _metrics(scores: np.ndarray, gold: np.ndarray, temperature: float) -> dict:
    probs = apply_temperature(scores, temperature)
    if not all(schema_valid(row, probs.shape[1]) for row in probs):
        raise RuntimeError("readout was not a simplex")
    scored = score_probabilities(probs, gold)
    scored.pop("risk_coverage", None)
    scored.pop("ece_bins", None)
    return scored


def dev() -> None:
    spec = json.loads(READOUT_PATH.read_text())
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    system = IndexReadout(spec)
    table = {}
    for dataset in spec["datasets"]:
        labels = protocol["datasets"][dataset]["labels"]
        items = materialized["items"][dataset]["dev"]
        print(f"dev index readout {dataset} n={len(items)} options={len(labels)}", flush=True)
        index, gold, _, truncated = _score_split(system, items, labels, timed=False)
        prior = index.mean(axis=0)
        np.savez_compressed(OUT / f"{dataset}__dev.npz", index=index, gold=gold, prior=prior)
        variants = {
            "label_string": _label_string_scores(dataset, "dev", items),
            "index": index,
            "index_prior": index - prior,
        }
        table[dataset] = {"n_options": len(labels), "n": len(items), "truncated": truncated, "variants": {}}
        for name in VARIANTS:
            fitted = fit_temperature(variants[name], gold)
            scored = _metrics(variants[name], gold, fitted["temperature"])
            table[dataset]["variants"][name] = {
                "temperature": fitted,
                "dev_accuracy": scored["accuracy"],
                "dev_nll": scored["log_loss"],
                "dev_nll_over_ln_k": scored["log_loss"] / math.log(len(labels)),
                "dev_brier": scored["brier"],
            }
            print(f"  {name:13s} acc {scored['accuracy']:.4f} nll {scored['log_loss']:.4f} T {fitted['temperature']:.3f}", flush=True)
    means = {name: float(np.mean([table[d]["variants"][name]["dev_nll_over_ln_k"] for d in spec["datasets"]])) for name in VARIANTS}
    selected = min(VARIANTS, key=lambda name: (means[name], VARIANTS.index(name)))
    payload = {"protocol_id": spec["protocol_id"], "selection_scores": means, "selected": selected, "datasets": table}
    (OUT / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selection_scores": means, "selected": selected}), flush=True)
    system.close()


def test() -> None:
    spec = json.loads(READOUT_PATH.read_text())
    selection_path = OUT / "selection.json"
    if not selection_path.exists():
        raise RuntimeError("selection.json is missing; test stays unscored")
    selection = json.loads(selection_path.read_text())
    if selection["protocol_id"] != spec["protocol_id"]:
        raise RuntimeError("selection belongs to a different protocol")
    selected = selection["selected"]
    if selected == "label_string":
        print("label_string was selected; its test metrics are the Program 1 table", flush=True)
        return
    protocol = load_protocol()
    materialized = load_items(ROOT / "artifacts" / "splits.json", protocol)
    system = IndexReadout(spec)
    warmup = protocol["latency"]["warmup"]
    results = {name: {"protocol_id": spec["protocol_id"], "variant": name, "selected": name == selected, "datasets": {}} for name in ("index", "index_prior")}
    for dataset in spec["datasets"]:
        labels = protocol["datasets"][dataset]["labels"]
        items = materialized["items"][dataset]
        for item in items["dev"][:warmup]:
            system.score(item.text, labels)
        print(f"test index readout {dataset} n={len(items['test'])}", flush=True)
        index, gold, latencies, truncated = _score_split(system, items["test"], labels, timed=True)
        stored = np.load(OUT / f"{dataset}__dev.npz")
        np.savez_compressed(OUT / f"{dataset}__test.npz", index=index, gold=gold)
        timing = np.asarray(latencies, dtype=np.float64)
        for name, scores in (("index", index), ("index_prior", index - stored["prior"])):
            temperature = selection["datasets"][dataset]["variants"][name]["temperature"]["temperature"]
            results[name]["datasets"][dataset] = {
                "n_options": len(labels),
                "temperature": temperature,
                "scaled": _metrics(scores, gold, temperature),
                "unscaled": _metrics(scores, gold, 1.0),
                "n_truncated": truncated,
                "output_tokens": 0,
                "latency": {
                    "n": int(timing.size),
                    "p50_ms": float(np.percentile(timing, 50, method="linear")),
                    "p95_ms": float(np.percentile(timing, 95, method="linear")),
                },
            }
            scaled = results[name]["datasets"][dataset]["scaled"]
            print(f"  {name:11s} acc {scaled['accuracy']:.4f} brier {scaled['brier']:.4f}", flush=True)
    (OUT / "test_selected.json").write_text(json.dumps(results[selected], indent=2) + "\n")
    other = "index_prior" if selected == "index" else "index"
    (OUT / "test_not_selected.json").write_text(json.dumps(results[other], indent=2) + "\n")
    system.close()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "dev":
        dev()
    elif command == "test":
        test()
    elif command == "check":
        check()
    else:
        raise SystemExit("usage: python -m bakeoff.readout dev|test|check")

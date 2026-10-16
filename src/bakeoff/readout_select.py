"""Choose each local Qwen3.5 model's readout per dataset on dev. Frozen rules are in readout_select_protocol.json."""

from __future__ import annotations

import json

import numpy as np

from bakeoff.metrics import apply_temperature, score_probabilities
from bakeoff.protocol import ROOT
from bakeoff.suite import load_suite
from bakeoff.suite_run import _report
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "readout_select_protocol.json"
OUT = ROOT / "artifacts" / "readout-select"


def _index_scores(folder, dataset: str) -> dict:
    dev = np.load(folder / f"{dataset}__dev.npz")["index"]
    prior = dev.mean(axis=0)
    out = {"dev": dev - prior, "test": np.load(folder / f"{dataset}__test.npz")["index"] - prior}
    oos = folder / f"{dataset}__oos.npz"
    if oos.exists():
        out["oos"] = np.load(oos)["index"] - prior
    return out


def _generated_scores(folder, dataset: str) -> dict | None:
    if not (folder / f"{dataset}__dev.npz").exists():
        return None
    out = {"dev": np.load(folder / f"{dataset}__dev.npz")["scores"]}
    for split in ("test", "oos"):
        path = folder / f"{dataset}__{split}.npz"
        if path.exists():
            out[split] = np.load(path)["scores"]
    return out


def _dev_nll(scores: np.ndarray, gold: np.ndarray) -> float:
    fitted = fit_temperature(scores, gold)
    return float(score_probabilities(apply_temperature(scores, fitted["temperature"]), gold)["log_loss"])


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    suite = load_suite(rules["datasets"])
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"protocol_id": rules["protocol_id"], "systems": {}}
    for system, folders in rules["systems"].items():
        block_out = {}
        for dataset in rules["datasets"]:
            block = suite[dataset]
            index_dir, generated_dir = ROOT / folders["index"], ROOT / folders["generated"]
            if not (index_dir / f"{dataset}__dev.npz").exists():
                continue
            gold_dev = np.asarray([item.gold_index for item in block["dev"]], dtype=np.int64)
            gold_test = np.asarray([item.gold_index for item in block["test"]], dtype=np.int64)
            candidates = {"index": _index_scores(index_dir, dataset)}
            generated = _generated_scores(generated_dir, dataset)
            if generated is not None:
                candidates["generated"] = generated
            dev_values = {name: _dev_nll(scores["dev"], gold_dev) for name, scores in candidates.items()}
            chosen = min(dev_values, key=lambda name: (dev_values[name], name != "index"))
            if "test" not in candidates[chosen]:
                print(f"{system} {dataset}: {chosen} selected but its test split is not scored yet", flush=True)
                block_out[dataset] = {"dev_nll": dev_values, "selected": chosen, "pending": True}
                continue
            test_accuracy = {}
            for name, scores in candidates.items():
                if "test" in scores:
                    temperature = fit_temperature(scores["dev"], gold_dev)["temperature"]
                    test_accuracy[name] = float((apply_temperature(scores["test"], temperature).argmax(axis=1) == gold_test).mean())
            row = _report(f"{system} [{chosen}]", dataset, block, candidates[chosen], {"dev_nll": dev_values, "selected": chosen, "test_accuracy_by_readout": test_accuracy})
            block_out[dataset] = row
        summary["systems"][system] = block_out
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SELECT_DONE", flush=True)


if __name__ == "__main__":
    main()

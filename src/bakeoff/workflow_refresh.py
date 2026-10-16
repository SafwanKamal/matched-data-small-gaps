"""Rerun workflows v1 to v3 with the specialists trained at the dev-selected rate. Rules: workflow_refresh_protocol.json."""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np

from bakeoff import workflow as v1
from bakeoff import workflow_v2 as v2
from bakeoff import workflow_v3 as v3
from bakeoff.protocol import ROOT
from bakeoff.temperature import fit_temperature

RULES_PATH = ROOT / "workflow_refresh_protocol.json"
OLD = {"v1": v1.OUT, "v2": v2.OUT, "v3": v3.OUT}
NEW = {key: ROOT / "artifacts" / "refresh" / path.name for key, path in OLD.items()}
PARTS = ("session_turns", "dev_in_scope", "dev_out_of_scope")


def prepare() -> None:
    """Put the new specialists' scores, and everything reused, where the v1 code looks for them. Decoder needs the GPU."""
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    out = NEW["v1"]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sessions.json", "jev_typed_state.jsonl", "jev_raw_history.jsonl"):
        shutil.copyfile(OLD["v1"] / name, out / name)
    for kind in ("deepseek", "qwen37", "gliclass", "qwen4b"):
        for part in PARTS:
            shutil.copyfile(OLD["v1"] / f"{kind}__{part}.npz", out / f"{kind}__{part}.npz")
    encoder = dict(np.load(ROOT / "artifacts" / "data-fraction" / "scores" / "encoder-150m@1.npz"))
    for part in PARTS:
        np.savez_compressed(out / f"encoder__{part}.npz", scores=encoder[part])
    if not (out / "decoder__session_turns.npz").exists():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from bakeoff.data import Item
        from bakeoff.data_fraction import _texts
        from bakeoff.train_ettin_decoder import _score_items

        folder = ROOT / rules["specialists"]["decoder"]
        saved = json.loads((folder / "metrics.json").read_text())
        _, labels, _, texts = _texts()
        tokenizer = AutoTokenizer.from_pretrained(folder / "model")
        model = AutoModelForCausalLM.from_pretrained(folder / "model", dtype=torch.float32)
        model.config.use_cache = False
        model.to("cuda")
        spec = json.loads((ROOT / "program2_decoder.json").read_text())
        for part in PARTS:
            items = [Item(item_id=str(i), row=i, text=text, gold_index=0, gold_label="") for i, text in enumerate(texts[part])]
            scores, _, _ = _score_items(model, tokenizer, items, labels, spec, torch.device("cuda"), saved["batch_size"], saved["label_chunk_size"])
            np.savez_compressed(out / f"decoder__{part}.npz", scores=scores)
            print(f"scored decoder {part}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("REFRESH_PREPARED", flush=True)


def _patch() -> None:
    v1.OUT, v2.OUT, v3.OUT = NEW["v1"], NEW["v2"], NEW["v3"]
    original = v1._temperature
    _, _, dev, _, _ = v1._sources()
    gold = np.asarray([item.gold_index for item in dev], dtype=np.int64)

    def temperature(kind: str) -> float:
        if kind in ("encoder", "decoder"):
            return float(fit_temperature(np.load(NEW["v1"] / f"{kind}__dev_in_scope.npz")["scores"], gold)["temperature"])
        return original(kind)

    v1._temperature = temperature


def compare() -> None:
    for key, name in (("v1", "summary.json"), ("v2", "summary.json")):
        old = json.loads((OLD[key] / name).read_text())["policies"]
        new = json.loads((NEW[key] / name).read_text())["policies"]
        print(f"\nworkflow {key}: session success %, old -> new")
        for policy in old:
            if policy in new:
                print(f"  {policy:48s} {100 * old[policy]['session_success']:5.1f} -> {100 * new[policy]['session_success']:5.1f}   turn {100 * old[policy]['turn_success']:5.1f} -> {100 * new[policy]['turn_success']:5.1f}")
    old, new = (json.loads((folder / "summary.json").read_text()) for folder in (OLD["v3"], NEW["v3"]))
    print("\nworkflow v3: ranking AUROC old -> new", {k: (round(old["ranking_auroc"][k], 3), round(new["ranking_auroc"][k], 3)) for k in old["ranking_auroc"]})
    print(f"  no escalation {100 * old['no_escalation']['session_success']:.1f} -> {100 * new['no_escalation']['session_success']:.1f}")
    for cell in old["results"]:
        print(f"  {cell:14s} " + "  ".join(f"{name} {100 * old['results'][cell][name]['session_success']:.1f}->{100 * new['results'][cell][name]['session_success']:.1f}" for name in old["results"][cell]))


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "prepare"):
        prepare()
    if stage in ("all", "run"):
        _patch()
        for module in (v1, v2, v3):
            module.OUT.mkdir(parents=True, exist_ok=True)
            module.main()
    if stage in ("all", "run", "compare"):
        compare()
    print("REFRESH_DONE", flush=True)


if __name__ == "__main__":
    main()

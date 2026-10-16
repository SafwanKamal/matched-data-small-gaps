"""Repeat the MASSIVE finetunes under extra seeds. Rules are in primary_seeds.json; the recipe is primary_protocol.json."""

from __future__ import annotations

import json

import torch

from bakeoff import primary
from bakeoff.massive import load_massive, load_primary_protocol
from bakeoff.protocol import ROOT

SEEDS_PATH = ROOT / "primary_seeds.json"


def main() -> None:
    rules = json.loads(SEEDS_PATH.read_text())
    base_out = primary.OUT
    probe = json.loads(primary.PROBE_PATH.read_text())
    loaded = load_massive(load_primary_protocol())
    for seed in rules["extra_seeds"]:
        protocol = load_primary_protocol()
        protocol["ettin_encoder"]["seed"] = seed
        protocol["ettin_decoder"]["seed"] = seed
        # The frozen window has passed; these runs are not gated by it.
        protocol["finetune_must_finish_by_utc"] = "2099-01-01T00:00:00Z"
        primary.OUT = base_out / f"seed-{seed}"
        print(f"=== MASSIVE seed {seed}", flush=True)
        torch.manual_seed(seed)
        encoder = primary.finetune_encoder(protocol, loaded, probe["encoder"]["projected_seconds"])
        torch.manual_seed(seed)
        decoder = primary.finetune_decoder(protocol, loaded, probe["decoder"]["projected_seconds"])
        summary = {"seed": seed, "encoder": encoder, "decoder": decoder}
        (primary.OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    primary.OUT = base_out
    print("MASSIVE_SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()

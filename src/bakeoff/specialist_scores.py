"""Per-turn specialist scores, small enough to ship with the release.

    python -m bakeoff.specialist_scores

The analyses read the specialist through its cached features (embeddings and logits, about 1 GB), which are not in git.
This writes, next to each cached feature file, a file in `specialist-scores/` with only what the analyses use: the
logits of the development and test parts, the gold labels, the Mahalanobis and kNN detector scores, and (for the
confirmatory sets) the kNN vote of run C's rule. The loaders fall back to these files when the features are absent, and
every analysis output stays byte-identical. Covers the confirmatory sets, the MASSIVE draws and run D.
"""

from __future__ import annotations

import numpy as np

from bakeoff.protocol import ROOT
from bakeoff.revision_free import knn_vote
from bakeoff.robustness_seeds import embedding_scores

RUNS = ["confirmatory", "massive-robustness", "retrieval-suppression"]
PARTS = ("dev_in", "dev_out", "turns")


def compact(features: dict, vote: bool) -> dict:
    out = {f"{p}_logits": features[f"{p}_logits"] for p in PARTS}
    out["dev_gold"], out["train_gold"] = features["dev_gold"], features["train_gold"]
    for part, scores in embedding_scores(features).items():
        for name, values in scores.items():
            out[f"{name}_{part}"] = values
    if vote:
        for part in PARTS:
            out[f"knn_vote_{part}"] = knn_vote(features, part)
    return out


def main() -> None:
    for run in RUNS:
        folder = ROOT / "artifacts" / run
        target = folder / "specialist-scores"
        target.mkdir(exist_ok=True)
        for path in sorted((folder / "features").glob("*.npz")):
            features = dict(np.load(path))
            np.savez_compressed(target / path.name, **compact(features, run == "confirmatory" and "__normalised__" in path.name))
        total = sum(p.stat().st_size for p in target.glob("*.npz"))
        print(f"{run}: {len(list(target.glob('*.npz')))} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

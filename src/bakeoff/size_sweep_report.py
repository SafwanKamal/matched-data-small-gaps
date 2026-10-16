"""Tabulate the Ettin size sweep: matched-rate and dev-selected rows per size. Reads saved metrics only."""

from __future__ import annotations

import json

from bakeoff.protocol import ROOT
from bakeoff.size_sweep import OUT, RULES_PATH

REFERENCE = {
    ("encoder", "banking77"): "artifacts/program2/banking77-log_loss/metrics.json",
    ("encoder", "sst2"): "artifacts/program2/sst2-log_loss/metrics.json",
    ("encoder", "massive_en"): "artifacts/primary/ettin-encoder/metrics.json",
    ("decoder", "banking77"): "artifacts/program2-decoder/banking77-log_loss/metrics.json",
    ("decoder", "sst2"): "artifacts/program2-decoder/sst2-log_loss/metrics.json",
    ("decoder", "massive_en"): "artifacts/primary/ettin-decoder/metrics.json",
}


def _row(metrics: dict, rate: float) -> dict:
    return {
        "learning_rate": rate,
        "accuracy": metrics["scaled"]["accuracy"],
        "brier": metrics["scaled"]["brier"],
        "log_loss": metrics["scaled"]["log_loss"],
        "ece": metrics["scaled"]["ece"],
        "dev_nll": metrics["temperature"]["dev_nll"],
        "p50_ms": (metrics.get("latency") or {}).get("p50_ms"),
        "parameters": metrics.get("parameters"),
    }


def main() -> None:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    rates = [rules["learning_rates"]["matched"]] + rules["learning_rates"]["extra"]
    table = {}
    for kind in ("encoder", "decoder"):
        for size in rules["sizes"]:
            for dataset in rules["datasets"]:
                rows = []
                for rate in rates:
                    path = OUT / f"{kind}-{size}" / f"lr-{rate:g}" / dataset / "sweep.json"
                    if path.exists():
                        rows.append(_row(json.loads(path.read_text()), rate))
                    elif size == "150m" and rate == rates[0]:
                        rows.append(_row(json.loads((ROOT / REFERENCE[(kind, dataset)]).read_text()), rate))
                if not rows:
                    continue
                matched = next((row for row in rows if row["learning_rate"] == rates[0]), None)
                selected = min(rows, key=lambda row: row["dev_nll"])
                table[f"{kind}-{size}/{dataset}"] = {"matched": matched, "dev_selected": selected, "rates_run": [row["learning_rate"] for row in rows]}
    (OUT / "summary.json").write_text(json.dumps({"protocol_id": rules["protocol_id"], "table": table}, indent=2) + "\n")
    for dataset in rules["datasets"]:
        print(f"\n{dataset}: accuracy % (Brier) at the matched rate | dev-selected rate")
        for kind in ("encoder", "decoder"):
            for size in rules["sizes"]:
                entry = table.get(f"{kind}-{size}/{dataset}")
                if not entry:
                    continue
                cell = lambda row: "      -      " if row is None else f"{100 * row['accuracy']:5.1f} ({row['brier']:.3f})"
                chosen = entry["dev_selected"]
                latency = chosen["p50_ms"]
                print(
                    f"  {kind:7s} {size:5s} {cell(entry['matched'])} | {cell(chosen)} lr {chosen['learning_rate']:g} "
                    f"p50 {latency if latency is None else round(latency, 1)} ms rates {entry['rates_run']}"
                )


if __name__ == "__main__":
    main()

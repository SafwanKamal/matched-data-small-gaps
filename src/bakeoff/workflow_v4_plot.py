"""Cost against session success for workflow v4. Reads artifacts/workflow-v4/summary.json only."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bakeoff.workflow_v4 import OUT, RATES


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    part_a = summary["part_a"]
    models = sorted(part_a["models"].items(), key=lambda item: item[1]["params_b"])
    colors = plt.cm.viridis([i / max(1, len(models) - 1) for i in range(len(models))])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    ax = axes[0]
    for (name, row), color in zip(models, colors):
        curve = row["escalation"]["encoder"]
        xs = [curve[str(rate)]["by_confidence"]["usd_per_1000_sessions"] for rate in RATES if rate > 0]
        ys = [100 * curve[str(rate)]["by_confidence"]["session_success"] for rate in RATES if rate > 0]
        ax.plot(xs, ys, "-o", color=color, markersize=3.5, linewidth=1.3, label=f"{name} ({row['params_b']}B)")
        ax.plot([row["alone"]["usd_per_1000_sessions"]], [100 * row["alone"]["session_success"]], "s", color=color, markersize=8, markeredgecolor="black")
        for form, marker in (("blind", "^"), ("informed", "*")):
            entry = (summary.get("part_b") or {}).get(name, {}).get(form)
            if entry:
                ax.plot([entry["usd_per_1000_sessions"]], [100 * entry["session_success"]], marker, color=color, markersize=11, markeredgecolor="black")
    for kind, style in (("encoder", "--"), ("decoder", ":")):
        ax.axhline(100 * part_a["specialists"][kind][ "session_success"], color="gray", linestyle=style, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("API cost per 1000 five-turn sessions (USD, log scale)")
    ax.set_ylabel("sessions with every decision right (%)")
    ax.set_title("Encoder first, escalating its least confident turns", fontsize=11)
    ax.text(0.5, -0.17, "line: 5% to 100% of turns escalated    square: model alone    triangle / star: de-escalation, blind / informed\ndashed / dotted grey: fine-tuned encoder / decoder alone, no API cost", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    sizes = [row["params_b"] for _, row in models]
    alone = [100 * row["alone"]["session_success"] for _, row in models]
    best = [100 * max(row["escalation"]["encoder"][str(rate)]["by_confidence"]["session_success"] for rate in RATES if rate < 1.0) for _, row in models]
    best_decoder = [100 * max(row["escalation"]["decoder"][str(rate)]["by_confidence"]["session_success"] for rate in RATES if rate < 1.0) for _, row in models]
    ax.plot(sizes, alone, "s", color="#1f4e79", label="model alone")
    ax.plot(sizes, best, "o-", color="#c47b2b", label="encoder + escalation, best rate below 100%")
    ax.plot(sizes, best_decoder, "o--", color="#2f6f4e", label="decoder + escalation, best rate below 100%")
    for (name, row), x, y in zip(models, sizes, alone):
        ax.annotate(name, (x, y), fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.axhline(100 * part_a["specialists"]["encoder"]["session_success"], color="gray", linestyle="--", linewidth=1)
    ax.axhline(100 * part_a["specialists"]["decoder"]["session_success"], color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("parameters of the hosted model (billions, log scale)")
    ax.set_ylabel("sessions with every decision right (%)")
    ax.set_title("Session success against the size of the hosted model", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "cost_effect.png", dpi=150)
    print(f"wrote {OUT / 'cost_effect.png'}")


if __name__ == "__main__":
    main()

"""Paper figures F2 (main comparison) and F3 (session success against training data). Reads saved summaries only."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bakeoff.protocol import ROOT

OUT = ROOT / "docs" / "figures"
A = ROOT / "artifacts"
# Reference palette slots 1-3 (validated all-pairs in its documentation); text and grid in neutral ink.
SPECIALIST, GEMMA, QWEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def figure_main() -> None:
    seeds = _load(A / "robustness" / "summary_seeds.json")
    detectors = _load(A / "oos-detectors" / "summary.json")["domains"]
    near = _load(A / "workflow-near-oos" / "summary.json")["sets"]
    retrieval = _load(A / "workflow-retrieval" / "summary.json")["domains"]
    second = _load(A / "robustness" / "summary_api.json")
    multidogo = _load(A / "workflow-multidogo" / "summary.json")["systems"]

    def first_seed_synthetic(name):
        rows = detectors[name]["detectors"]
        best = max(rows, key=lambda d: rows[d]["dev_balanced_accuracy"])
        return rows[best]["session_success"]

    def first_seed_near(kind):
        block = near[kind]
        return block["rows"][f"encoder_{block['dev_selected_detector']}"]["session_success"]

    rows = []
    for name, label in (("banking77", "Banking77"), ("massive", "MASSIVE"), ("clinc150", "CLINC150")):
        values = [first_seed_synthetic(name)] + [seeds[name][s]["dev_selected"] for s in ("20261015", "20261016")]
        rows.append((label, values, retrieval[name]["runs"]["gemma-4-31b__full_train"]["alone"]["session_success"], second["synthetic"][name]["qwen3.5-27b"]["alone"]["session_success"]))
    for kind, label in (("ood", "BANKING77-OOS, off-topic out-of-scope"), ("id", "BANKING77-OOS, held-out-intent out-of-scope")):
        values = [first_seed_near(kind)] + [seeds["banking77_oos"][s][kind]["dev_selected"] for s in ("20261015", "20261016")]
        rows.append((label, values, near[kind]["rows"]["gemma_retrieval_alone"]["session_success"], second["banking77_oos"][f"{kind}__retrieval"]["session_success"]))
    values = [multidogo["encoder"]["conversation_success"]] + [seeds["multidogo"][s]["conversation_success"] for s in ("20261015", "20261016")]
    rows.append(("MultiDoGO finance, real conversations", values, multidogo["gemma_retrieval_current"]["conversation_success"], second["multidogo"]["retrieval_current"]["conversation_success"]))

    fig, ax = plt.subplots(figsize=(7.4, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    for i, (label, values, gemma, qwen) in enumerate(rows):
        y = len(rows) - 1 - i
        low, high, mean = 100 * min(values), 100 * max(values), 100 * sum(values) / len(values)
        ax.plot([low, high], [y, y], color=SPECIALIST, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([mean], [y], s=52, color=SPECIALIST, edgecolor=SURFACE, linewidth=2, zorder=3, label="Specialist, distance detector (mean, range of 3 seeds)" if i == 0 else None)
        ax.scatter([100 * gemma], [y + 0.18], s=52, marker="D", color=GEMMA, edgecolor=SURFACE, linewidth=2, zorder=3, label="Gemma 4 31B, retrieval" if i == 0 else None)
        ax.scatter([100 * qwen], [y - 0.18], s=58, marker="^", color=QWEN, edgecolor=SURFACE, linewidth=2, zorder=3, label="Qwen3.5 27B, retrieval" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in reversed(rows)], color=INK, fontsize=8.5)
    ax.set_xlim(20, 90)
    ax.set_xlabel("sessions with every decision right (%)  (conversations for MultiDoGO)", color=MUTED, fontsize=8.5)
    fig.suptitle("Best-equipped specialist against general models given the same labelled data", color=INK, fontsize=10, x=0.02, ha="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3, frameon=False, fontsize=7.5, labelcolor=INK, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"f2_main_comparison.{suffix}", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def figure_data_size() -> None:
    panels = [
        ("Banking77 (77 intents)", 77, _load(A / "data-fraction" / "summary.json")["specialists"], "encoder-150m@{f}", _load(A / "oos-detectors" / "summary_fractions.json"), _load(A / "workflow-retrieval" / "summary_fractions.json")["rows"], "banking77"),
        ("MASSIVE (60 intents)", 60, _load(A / "data-fraction-domains" / "summary__massive.json")["specialists"], "encoder@{f}", _load(A / "oos-detectors" / "summary_fractions__massive.json"), _load(A / "workflow-retrieval" / "summary_fractions__massive.json")["rows"], "massive"),
        ("CLINC150 (150 intents)", 150, _load(A / "data-fraction-domains" / "summary__clinc150.json")["specialists"], "encoder@{f}", _load(A / "oos-detectors" / "summary_fractions__clinc150.json"), _load(A / "workflow-retrieval" / "summary_fractions__clinc150.json")["rows"], "clinc150"),
    ]
    full_detectors = _load(A / "oos-detectors" / "summary.json")["domains"]
    full_retrieval = _load(A / "workflow-retrieval" / "summary.json")["domains"]
    zero = {"banking77": 0.422, "massive": 0.505, "clinc150": 0.625}
    fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), facecolor=SURFACE, sharey=True)
    for ax, (title, n_labels, specialists, key, detector_rows, retrieval_rows, name) in zip(axes, panels):
        _style(ax)
        xs, soft, det, ret = [], [], [], []
        for f in fractions:
            entry = specialists[key.format(f=f"{f:g}")]
            xs.append(retrieval_rows[f"gemma-4-31b@{f:g}"]["train_rows"] / n_labels)
            soft.append(100 * entry["alone"]["session_success"])
            row = detector_rows[f"{f:g}"]
            det.append(100 * row["detectors"][row["dev_selected"]]["session_success"])
            ret.append(100 * retrieval_rows[f"gemma-4-31b@{f:g}"]["retrieval_alone"]["session_success"])
        full = specialists[key.format(f="1")]
        # Full training sets: Banking77 without the Program 1 dev rows, MASSIVE official train, CLINC150 in-scope train.
        xs.append({"banking77": 9603, "massive": 11514, "clinc150": 15000}[name] / n_labels)
        soft.append(100 * full["alone"]["session_success"])
        rows = full_detectors[name]["detectors"]
        det.append(100 * rows[max(rows, key=lambda d: rows[d]["dev_balanced_accuracy"])]["session_success"])
        ret.append(100 * full_retrieval[name]["runs"]["gemma-4-31b__full_train"]["alone"]["session_success"])
        ax.axhline(100 * zero[name], color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
        ax.plot(xs, ret, color=GEMMA, linewidth=2, marker="D", markersize=5, markeredgecolor=SURFACE, label="Gemma 4 31B, retrieval from the same rows")
        ax.plot(xs, det, color=SPECIALIST, linewidth=2, marker="o", markersize=5.5, markeredgecolor=SURFACE, label="Specialist, distance-based scope detector")
        ax.plot(xs, soft, color=QWEN, linewidth=2, marker="^", markersize=5.5, markeredgecolor=SURFACE, label="Specialist, softmax threshold")
        ax.set_xscale("log")
        ax.text(xs[0], 100 * zero[name] - 1.5, "Gemma zero-shot", color=MUTED, fontsize=7.5, ha="left", va="top")
        ax.set_xticks([1, 3, 10, 30, 100])
        ax.set_xticklabels(["1", "3", "10", "30", "100"])
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("training examples per intent", color=MUTED, fontsize=8.5)
        ax.set_ylim(0, 90)
    axes[0].set_ylabel("sessions with every decision right (%)", color=MUTED, fontsize=8.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"f3_data_size.{suffix}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_handoff() -> None:
    """F4: writer by reader accuracy at the 200-word budget, three conversation lengths."""
    from matplotlib.colors import LinearSegmentedColormap

    sets = _load(A / "handoff" / "summary.json")["sets"]
    writers = ["qwen3.5-9b", "qwen3.5-27b", "gemma-4-31b", "qwen3.5-122b-a10b"]
    readers = writers + ["jev"]
    short = {"qwen3.5-9b": "Qwen3.5 9B", "qwen3.5-27b": "Qwen3.5 27B", "gemma-4-31b": "Gemma 4 31B", "qwen3.5-122b-a10b": "Qwen3.5 122B", "jev": "Jev 1.13.0"}
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), facecolor=SURFACE)
    for ax, (name, title) in zip(axes, (("10_turns", "10 turns"), ("20_turns", "20 turns"), ("40_turns", "40 turns"))):
        block = sets[name]
        rows = [[100 * block["raw"][r]["accuracy"] for r in readers]] + [[100 * block["cells"][f"{w}@200->{r}"]["accuracy"] for r in readers] for w in writers]
        image = ax.imshow(rows, cmap=ramp, vmin=40, vmax=95, aspect="auto")
        for i, row in enumerate(rows):
            for k, value in enumerate(row):
                ax.text(k, i, f"{value:.0f}", ha="center", va="center", fontsize=8, color="white" if value >= 75 else INK)
        ax.set_xticks(range(len(readers)))
        ax.set_xticklabels([short[r] for r in readers], rotation=35, ha="right", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(["Raw conversation"] + [f"Notes by {short[w]}" for w in writers] if ax is axes[0] else [""] * len(rows), fontsize=7.5, color=INK)
        ax.set_title(f"{title}, 200-word notes", fontsize=9.5, color=INK, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        for k in range(len(readers) + 1):
            ax.axvline(k - 0.5, color=SURFACE, linewidth=2)
        for i in range(len(rows) + 1):
            ax.axhline(i - 0.5, color=SURFACE, linewidth=2)
    axes[1].set_xlabel("reader (answers from the notes alone)", color=MUTED, fontsize=8.5)
    bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    bar.set_label("questions answered correctly (%)", color=MUTED, fontsize=8)
    bar.ax.tick_params(labelsize=7.5, colors=MUTED)
    bar.outline.set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"f4_handoff.{suffix}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_main()
    figure_data_size()
    figure_handoff()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

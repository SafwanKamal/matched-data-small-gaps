"""Main-text figures for paper draft 3: confirmatory results, operating points, MASSIVE rows. Reads analysis artifacts only."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bakeoff.protocol import ROOT

OUT = ROOT / "docs" / "figures"
A = ROOT / "artifacts"
# Reference palette (docs/figures style): categorical slots in fixed order; neutral ink for text and grid.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def j(*parts):
    return json.loads(A.joinpath(*parts).read_text(encoding="utf-8"))


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=220, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def forest() -> None:
    """Paired differences against Gemma 4 31B with 95% cluster-bootstrap intervals (weighted turn accuracy)."""
    conf = j("confirmatory", "analysis.json")["sets"]
    fresh = j("posthoc", "round3b.json")["freshness"]["sets"]
    r4 = j("posthoc", "round4.json")["jev_vs_gemma"]
    rows = [
        ("Specialist $-$ Gemma", lambda s: conf[s]["tests"]["specialist vs gemma-4-31b"], SLOT[0], "o", True),
        ("  same, never-seen turns", lambda s: fresh[s]["tests_never_seen"]["specialist vs gemma-4-31b"], SLOT[0], "o", False),
        ("Combination $-$ Gemma", lambda s: conf[s]["tests"]["combination vs gemma-4-31b"], SLOT[3], "P", True),
        ("Cascade 0.4 $-$ Gemma", lambda s: conf[s]["tests"]["cascade_0.4 vs gemma-4-31b"], SLOT[4], "v", True),
        ("Jev + retrieval $-$ Gemma", lambda s: r4[s], SLOT[5], "s", True),
        ("Specialist $-$ Qwen 27B", lambda s: conf[s]["tests"]["specialist vs qwen3.5-27b"], SLOT[2], "^", True),
    ]
    sets = [("banking77", "Banking77"), ("massive", "MASSIVE"), ("clinc150", "CLINC150")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), facecolor=SURFACE, sharey=True)
    for ax, (key, title) in zip(axes, sets):
        ax.axvline(0, color=MUTED, linewidth=0.8)
        for k, (label, get, colour, marker, filled) in enumerate(rows):
            t = get(key)
            y = len(rows) - 1 - k
            ax.plot(t["ci95"], [y, y], color=colour, linewidth=1.4, solid_capstyle="round")
            ax.scatter([t["difference_points"]], [y], marker=marker, s=30, facecolor=colour if filled else SURFACE, edgecolor=colour, linewidth=1.1, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in reversed(rows)])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("difference, weighted points (95% CI)")
        style(ax)
    fig.tight_layout()
    save(fig, "f6_confirmatory")


def confirmatory() -> None:
    conf = j("confirmatory", "analysis.json")["sets"]
    fresh = j("posthoc", "round3b.json")["freshness"]["sets"]
    jev = j("jev-rerun", "analysis.json")["J2"]
    systems = [("Specialist", "specialist", "specialist", SLOT[0], "o"), ("Gemma 4 31B", "gemma-4-31b", "gemma-4-31b", SLOT[1], "D"), ("Qwen3.5 27B", "qwen3.5-27b", "qwen3.5-27b", SLOT[2], "^"), ("Combination", "gemma-4-31b: combination", "gemma-4-31b: combination", SLOT[3], "P"), ("Cascade 0.4", "gemma-4-31b: cascade_0.4", "gemma-4-31b: cascade_0.4", SLOT[4], "v"), ("Jev + retrieval", None, "jev retrieval", SLOT[5], "s")]
    sets = [("banking77", "Banking77"), ("massive", "MASSIVE"), ("clinc150", "CLINC150")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, sets):
        sy = conf[key]["systems"]["prior_0.2"]
        for k, (label, conf_key, fresh_key, colour, marker) in enumerate(systems):
            y = len(systems) - 1 - k
            allv = 100 * (sy[conf_key]["weighted"] if conf_key else jev[key]["systems"]["jev retrieval"]["weighted"])
            newv = 100 * fresh[key]["weighted_never_seen"][fresh_key]
            ax.plot([allv, newv], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter([allv], [y], marker=marker, s=34, color=colour, edgecolor=SURFACE, linewidth=1, zorder=3, label="all turns" if k == 0 else None)
            ax.scatter([newv], [y], marker=marker, s=34, facecolor=SURFACE, edgecolor=colour, linewidth=1.2, zorder=3, label="never-seen turns" if k == 0 else None)
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels([s[0] for s in reversed(systems)] if key == "banking77" else [""] * len(systems))
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("weighted turn accuracy (%)")
        style(ax)
    handles = [plt.Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=5, label="all turns (filled)"), plt.Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=MUTED, linestyle="", markersize=5, label="never-seen turns (open; post hoc)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "f6_confirmatory")


def operating_points() -> None:
    d2 = j("retrieval-mechanism", "analysis.json")
    models = [("Gemma 4 31B", "gemma-4-31b", SLOT[1], "D"), ("Qwen3.5 27B", "qwen3.5-27b", SLOT[2], "^"), ("Llama 3.3 70B", "llama-3.3-70b", SLOT[3], "v"), ("DeepSeek V4 Pro", "deepseek-v4-pro", SLOT[4], "P"), ("Jev 1.13.0", "jev-1.13.0", SLOT[5], "s")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), facecolor=SURFACE)
    panels = [(axes[0], "banking77_oos", "BANKING77-OOS (fresh, clean pool)", [("zero_shot", "retrieval")], ["random_examples", "rule_after_examples"]), (axes[1], "clinc_single_domain", "CLINC single-domain", [("zero_shot_10", "retrieval_10"), ("zero_shot_20", "retrieval_20")], [])]
    for ax, key, title, pairs, extras in panels:
        for label, m, colour, marker in models:
            arms = d2[key][m]
            for k, (a, b) in enumerate(pairs):
                x0, y0 = 100 * arms[a]["in_scope_kept"], 100 * arms[a]["held_out_rejected"]
                x1, y1 = 100 * arms[b]["in_scope_kept"], 100 * arms[b]["held_out_rejected"]
                ls = "-" if k == 0 else (0, (2, 1.5))
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1, linestyle=ls, shrinkA=3, shrinkB=3))
                ax.scatter([x0], [y0], marker=marker, s=26, facecolor=SURFACE, edgecolor=colour, linewidth=1.1, zorder=3)
                ax.scatter([x1], [y1], marker=marker, s=30, color=colour, edgecolor=SURFACE, linewidth=0.8, zorder=4)
            for extra, emarker in zip(extras, ("x", "+")):
                ax.scatter([100 * arms[extra]["in_scope_kept"]], [100 * arms[extra]["held_out_rejected"]], marker=emarker, s=30, color=colour, linewidth=1.2, zorder=4)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("in-scope requests accepted (%)")
        style(ax)
    axes[0].set_ylabel("held-out requests rejected (%)")
    handles = [plt.Line2D([], [], marker=mk, color=c, linestyle="", markersize=5, label=l) for l, _, c, mk in models]
    handles += [plt.Line2D([], [], marker="o", markerfacecolor=SURFACE, markeredgecolor=MUTED, linestyle="", markersize=5, label="zero-shot (open)"), plt.Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=5, label="retrieval (filled)"), plt.Line2D([], [], marker="x", color=MUTED, linestyle="", markersize=5, label="random examples"), plt.Line2D([], [], marker="+", color=MUTED, linestyle="", markersize=6, label="rule after examples"), plt.Line2D([], [], color=MUTED, linestyle=(0, (2, 1.5)), label="20 options (CLINC)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=5, frameon=False, fontsize=7.2)
    fig.tight_layout()
    save(fig, "f7_operating_points")


def massive_rows() -> None:
    mr = j("massive-robustness", "analysis.json")["draws"]
    conf = j("confirmatory", "analysis.json")["sets"]["massive"]["systems"]["prior_0.2"]
    draws = [("original\nemail, iot,\nplay", "original"), ("draw 1\nalarm, play,\nsocial", "draw_1"), ("draw 2\ncalendar,\niot, play", "draw_2"), ("draw 3\naudio, take-\naway, weather", "draw_3")]
    arms = [("no rows", "gemma-4-31b no_rows", SLOT[1]), ("rows, other domains", "gemma-4-31b cross_source_rows", SLOT[3]), ("rows, tested domains", "gemma-4-31b same_source_rows", SLOT[2]), ("instruction", "gemma-4-31b instruction", SLOT[5])]
    fig, ax = plt.subplots(figsize=(3.35, 2.5), facecolor=SURFACE)
    width = 0.19
    for k, (label, key, colour) in enumerate(arms):
        vals = []
        for _, d in draws:
            sy = mr[d]["systems"]
            if d == "original":
                v = {"gemma-4-31b no_rows": 100 * conf["gemma-4-31b"]["out_of_scope_caught"], "gemma-4-31b same_source_rows": 100 * conf["gemma-4-31b with rows"]["out_of_scope_caught"]}.get(key)
                v = v if v is not None else (100 * sy[key]["held_out_rejected"] if key in sy else np.nan)
            else:
                v = 100 * sy[key]["held_out_rejected"]
            vals.append(v)
        x = np.arange(len(draws)) + (k - 1.5) * width
        ax.bar(x, vals, width=width - 0.02, color=colour, label=label, edgecolor=SURFACE, linewidth=0.5)
    spec = [100 * conf["specialist"]["out_of_scope_caught"]] + [100 * mr[d]["systems"]["specialist"]["held_out_rejected"] for _, d in draws[1:]]
    ax.scatter(np.arange(len(draws)), spec, marker="o", s=22, color=SLOT[0], edgecolor=SURFACE, zorder=4, label="specialist")
    ax.set_xticks(np.arange(len(draws)))
    ax.set_xticklabels([d[0] for d in draws], fontsize=6.6)
    ax.set_ylabel("held-out-domain requests\nrejected by Gemma (%)")
    ax.set_ylim(0, 100)
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.55), ncol=3, frameon=False, fontsize=6.6)
    save(fig, "f8_massive_rows")


if __name__ == "__main__":
    forest()
    operating_points()
    massive_rows()
    print("figures written")

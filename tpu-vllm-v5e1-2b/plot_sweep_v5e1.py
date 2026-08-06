#!/usr/bin/env python3
"""Plots the v5e-1 concurrency x context sweep from sweep_results_v5e1.csv.

Writes two PNGs next to the CSV. Deliberately separate from plot_grid.py and the
other comparison scripts, which still carry v6e labels and read sibling-project CSVs.

Usage:  python plot_sweep_v5e1.py [csv] [out_prefix]
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Validated palette (see the dataviz reference palette). Sequential blue 100->700 for
# magnitude; the ordinal 4-step subset for the ordered concurrency series.
SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# Total KV cache, in tokens, measured from the running server (num_gpu_blocks x block_size).
KV_CACHE_TOKENS = 321376


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)]
    for r in rows:
        for k, v in r.items():
            r[k] = float(v) if v not in (None, "") else None
    return rows


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def heatmap(rows, out):
    concs = sorted({int(r["concurrency"]) for r in rows})
    ctxs = sorted({int(r["input_len"]) for r in rows})
    grid = [
        [
            next(r["total_tok_per_s"] for r in rows if int(r["concurrency"]) == c and int(r["input_len"]) == x)
            for x in ctxs
        ]
        for c in concs
    ]

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")

    ax.set_xticks(range(len(ctxs)), [f"{x:,}" for x in ctxs])
    ax.set_yticks(range(len(concs)), [str(c) for c in concs])
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    peak = max(max(row) for row in grid)
    for i, c in enumerate(concs):
        for j, x in enumerate(ctxs):
            value = grid[i][j]
            # Ink flips on the dark end of the ramp so labels stay legible.
            colour = "#ffffff" if value > peak * 0.55 else INK
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", fontsize=9, color=colour)
            # 2px surface gap between cells.
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SURFACE, linewidth=2))
            # Mark the cells where concurrency exceeds what the KV cache holds resident.
            if c > KV_CACHE_TOKENS / x:
                ax.add_patch(
                    Rectangle(
                        (j - 0.42, i - 0.42),
                        0.84,
                        0.84,
                        fill=False,
                        edgecolor="#e34948",
                        linewidth=2,
                        linestyle=(0, (3, 2)),
                    )
                )

    ax.set_xlabel("context — input tokens per request", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_ylabel("concurrent users", color=INK_MUTED, fontsize=10, labelpad=8)
    ax.set_title("Gemma 4 E2B on TPU v5e-1 — total throughput (tokens/s)", color=INK, fontsize=13, pad=36, loc="left")
    ax.text(
        0,
        1.018,
        "dashed cells: concurrency exceeds the 321,376-token KV cache, so requests are preempted and re-prefilled",
        transform=ax.transAxes,
        color=INK_MUTED,
        fontsize=9,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)

    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def lines(rows, out):
    concs = sorted({int(r["concurrency"]) for r in rows})
    ctxs = sorted({int(r["input_len"]) for r in rows})
    panels = [
        ("total_tok_per_s", "Total throughput (tok/s)", False),
        ("mean_ttft_ms", "Mean time to first token (ms)", True),
        ("mean_tpot_ms", "Mean time per output token (ms)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.patch.set_facecolor(SURFACE)

    for ax, (metric, label, logy) in zip(axes, panels, strict=True):
        style_axes(ax)
        for idx, c in enumerate(concs):
            series = [
                next(r[metric] for r in rows if int(r["concurrency"]) == c and int(r["input_len"]) == x) for x in ctxs
            ]
            ax.plot(
                ctxs,
                series,
                color=ORDINAL_BLUE[idx],
                linewidth=2,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=f"{c} user" + ("s" if c > 1 else ""),
                zorder=3 + idx,
            )
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(ctxs, [f"{x // 1000}k" if x >= 1000 else str(x) for x in ctxs])
        ax.set_xlabel("context (input tokens)", color=INK_MUTED, fontsize=9.5)
        ax.set_title(label, color=INK, fontsize=11, loc="left", pad=8)

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper left")
    fig.suptitle(
        "Gemma 4 E2B on TPU v5e-1 — concurrency x context sweep", color=INK, fontsize=13, x=0.007, ha="left", y=0.99
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_results_v5e1.csv"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "sweep"
    rows = load(csv_path)
    print(f"✅ {heatmap(rows, f'{prefix}_heatmap_v5e1.png')}")
    print(f"✅ {lines(rows, f'{prefix}_lines_v5e1.png')}")


if __name__ == "__main__":
    main()

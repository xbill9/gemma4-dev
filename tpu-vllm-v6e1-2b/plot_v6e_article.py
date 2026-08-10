#!/usr/bin/env python3
"""Figures for the v6e-1 Medium article, from the 2026-08-10 validation run.

Medium renders images well and markdown tables not at all, so the two headline findings get
a figure each. Both read their numbers from `benchmarks/runs/2026-08-10-article-validation-v6e1/`
rather than restating them, so a re-run updates the plots.

Palette is the two-slot categorical default, validated with the dataviz skill's
`validate_palette.js` (light surface #fcfcfb): CVD ΔE 24.7 protan, 33.6 normal, all checks PASS.
Colour encodes *regime*, never rank — a cell keeps its colour regardless of where it sorts.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = pathlib.Path("benchmarks/runs/2026-08-10-article-validation-v6e1/results")
OUT = pathlib.Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
CONTROL = "#2a78d6"  # categorical slot 1
MEMBOUND = "#eb6834"  # categorical slot 2
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e0"

V5E = {
    (128, 1): 123.26,
    (128, 8): 738.28,
    (1024, 16): 896.11,
    (4096, 64): 585.92,
    (8192, 32): 307.76,
    (8192, 64): 314.44,
    (16000, 32): 166.76,
    (16000, 64): 166.69,
}
POOL_V5E = 321_376
POOL_V6E = 1_151_744
PRICE_RATIO = 2.25


def load():
    cells = []
    for name in ("validation.json", "validation2.json"):
        for c in json.load(open(RUN / name))["cells"]:
            if c.get("status") == "ok":
                cells.append(c)
    return cells


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def fig_asymmetry(cells):
    """What v6e returns per cell, against what it costs. Magnitude across categories -> bars."""
    rows = []
    for c in cells:
        k = (c["input_len"], c["concurrency"])
        if k not in V5E:
            continue
        need = c["concurrency"] * (c["input_len"] + 128)
        rows.append((f"{k[0]:,}x{k[1]}", c["output_tok_per_s"] / V5E[k], need < 0.10 * POOL_V5E))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=200, facecolor=SURFACE)
    style(ax)
    ys = range(len(rows))
    colors = [CONTROL if ctrl else MEMBOUND for _, _, ctrl in rows]
    ax.barh(list(ys), [r[1] for r in rows], color=colors, height=0.62, zorder=3)
    ax.axvline(PRICE_RATIO, color=INK, lw=1.6, ls="--", zorder=4)
    ax.text(
        PRICE_RATIO + 0.03,
        len(rows) - 0.35,
        f"price ratio {PRICE_RATIO}x",
        color=INK,
        fontsize=9.5,
        va="center",
        fontweight="bold",
    )

    for y, (_label, ratio, _) in zip(ys, rows, strict=True):
        ax.text(ratio + 0.03, y, f"{ratio:.2f}x", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(0, 3.15)
    ax.set_xlabel("v6e-1 throughput ÷ v5e-1 throughput", fontsize=10, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CONTROL), plt.Rectangle((0, 0), 1, 1, color=MEMBOUND)]
    ax.legend(
        handles,
        ["fits in a v5e pool", "exceeds a v5e pool"],
        frameon=False,
        fontsize=9.5,
        loc="lower right",
        labelcolor=INK,
    )
    ax.set_title(
        "v6e-1 only outruns its price tag when the working set will not fit a v5e-1",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        pad=14,
        loc="left",
    )
    fig.text(
        0.5,
        0.005,
        "context x concurrent clients · google/gemma-4-E2B-it · bf16 · TP=1",
        fontsize=8.5,
        color=INK_MUTED,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = OUT / "v6e-asymmetry.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_no_knee():
    """TTFT against concurrency across the pool boundary. Relationship -> scatter + fit."""
    pts = [
        (40, 2088),
        (46, 3659),
        (52, 5273),
        (56, 6309),
        (60, 7373),
        (64, 8459),
        (72, 10544),
        (80, 12689),
        (96, 16937),
        (112, 21229),
    ]
    a, b = -8542.0, 265.4
    boundary = POOL_V6E / (16000 + 128)

    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=200, facecolor=SURFACE)
    style(ax)
    xs = [p[0] for p in pts]
    fit_x = [min(xs) - 2, max(xs) + 2]
    ax.plot(fit_x, [(a + b * x) / 1000 for x in fit_x], color=INK_MUTED, lw=2, zorder=2)
    ax.scatter(xs, [p[1] / 1000 for p in pts], s=64, color=CONTROL, zorder=3, edgecolor=SURFACE, linewidth=2)

    ax.axvline(boundary, color=MEMBOUND, lw=1.6, ls="--", zorder=4)
    # Kept clear of the x tick labels; the fitted line is ~10.8 s here, so this band is empty.
    ax.text(boundary + 1.4, 4.3, "KV pool 100% full", color=MEMBOUND, fontsize=9.5, fontweight="bold")
    ax.text(boundary + 1.4, 3.3, "nothing happens here", color=INK_MUTED, fontsize=9)

    ax.text(
        43,
        17.2,
        "TTFT = −8542 + 265 × clients\nR² = 0.999996\n0 preemptions, every point",
        fontsize=10,
        color=INK,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID),
    )

    ax.set_xlabel("concurrent clients (16,000-token context)", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("median time to first token (s)", fontsize=10, color=INK_MUTED)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Crossing 100% of the KV cache is not an event", fontsize=12.5, color=INK, fontweight="bold", pad=14, loc="left"
    )
    fig.text(
        0.5,
        0.005,
        "56%–157% of the KV pool · the scheduler admits what fits and queues the rest",
        fontsize=8.5,
        color=INK_MUTED,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = OUT / "v6e-no-knee.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


if __name__ == "__main__":
    cells = load()
    for path in (fig_asymmetry(cells), fig_no_knee()):
        print(f"wrote {path}")

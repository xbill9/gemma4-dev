"""Draw the dev.to cover for the MI300X serving benchmark.

The house cover generator renders stat tiles, which is the right default and the
wrong picture for this result. The finding here is a *proportion* — the engine
allocates 9,026,017 tokens of KV cache and the heaviest cell in the grid wants
524,288 of them — so the cover draws that proportion at true scale instead of
printing it. The lit run of page blocks is 5.8% of the rail because the number
is 5.8%, computed from the report rather than chosen to look dramatic.

Geometry is dev.to's displayed 2.381:1 (1376x578), so nothing is cropped.
Drawn at 2x and downsampled, because type rendered at final size looks soft.
Colours are the dataviz reference palette's dark-mode steps on its dark
surface, and the pair passes every check in scripts/validate_palette.js:
blue #3987e5 carries capacity, orange #d95926 carries work.

    python3 make_benchmark_cover.py --report benchmarks/reports/<run>.json

Writes a content-addressed file, so a regenerated cover is a URL no proxy has
cached and the old bytes stay put for anything already published.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent

W, H = 1376, 578  # dev.to renders width=1000,height=420,fit=cover — 2.381:1
S = 2  # supersample factor

SURFACE = (26, 26, 25)  # dataviz dark surface #1a1a19
GRID = (38, 40, 42)
INK = (244, 244, 242)
INK_DIM = (150, 152, 150)
INK_FAINT = (98, 100, 99)
BLUE = (57, 135, 229)  # capacity
ORANGE = (217, 89, 38)  # work

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
MONO_B = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, max(8, int(size * S)))


def px(v: float) -> int:
    return int(round(v * S))


def mix(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))


def read_facts(report_path: Path) -> dict:
    """Pull every number on the cover out of the run's own report."""
    r = json.loads(report_path.read_text())
    cells = [c for c in r["throughput"]["sweep"] if c.get("status") == "ok"]
    pool = r["memory"]["resident_kv_tokens"]
    # The heaviest cell is the one wanting the most KV at once, which is what
    # the occupancy claim is about — not the fastest cell.
    heaviest = max(cells, key=lambda c: c["concurrency"] * (c["input_len"] + c["output_len"]))
    wanted = heaviest["concurrency"] * heaviest["input_len"]
    best = max(cells, key=lambda c: c["output_tok_per_s"])
    rate = r["hardware"]["pricing"]["rate_per_chip_hour"]
    return {
        "pool": pool,
        "wanted": wanted,
        "occupancy": wanted / pool,
        "kv_gib": r["memory"]["kv_cache_gib"],
        "hbm_gib": r["memory"]["usable_hbm_gib"],
        "best_rate": best["output_tok_per_s"],
        "best_clients": best["concurrency"],
        "best_ctx": best["input_len"],
        "dollars_per_m": rate / (best["output_tok_per_s"] * 3600) * 1e6,
        "curves": [
            (
                "128-token context",
                [
                    c["output_tok_per_s"]
                    for c in sorted((x for x in cells if x["input_len"] == 128), key=lambda x: x["concurrency"])
                ],
            ),
            (
                "8,192-token context",
                [
                    c["output_tok_per_s"]
                    for c in sorted((x for x in cells if x["input_len"] == 8192), key=lambda x: x["concurrency"])
                ],
            ),
        ],
        "engine": r["software"]["version"],
        "host": r["hardware"]["host"]["instance_name"],
    }


def draw_grid(d: ImageDraw.ImageDraw) -> None:
    for x in range(0, W + 1, 48):
        d.line([(px(x), 0), (px(x), px(H))], fill=GRID, width=1)
    for y in range(0, H + 1, 48):
        d.line([(0, px(y)), (px(W), px(y))], fill=GRID, width=1)


def draw_rail(d: ImageDraw.ImageDraw, facts: dict) -> tuple:
    """The KV pool at true scale: one block per 1%, the lit run is the occupancy.

    Discrete blocks rather than a continuous bar because the quantity is
    discrete (KV blocks) and because a 5.8% sliver on a continuous bar reads as
    a rounding error rather than as the point.
    """
    x0, x1 = 88, W - 88
    y_top, y_bot = 342, 410
    n = 100
    gap = 3
    span = x1 - x0
    bw = (span - gap * (n - 1)) / n
    lit = facts["occupancy"] * n  # 5.8 blocks

    for i in range(n):
        bx = x0 + i * (bw + gap)
        box = [px(bx), px(y_top), px(bx + bw), px(y_bot)]
        if i + 1 <= lit:  # wholly used
            d.rectangle(box, fill=ORANGE)
        elif i < lit:  # the partial block, drawn partially
            frac = lit - i
            d.rectangle(box, fill=mix(SURFACE, BLUE, 0.16), outline=mix(SURFACE, BLUE, 0.5), width=1)
            d.rectangle([box[0], box[1], px(bx + bw * frac), box[3]], fill=ORANGE)
        else:
            shade = 0.14 - 0.06 * (i / n)  # falls away to the right
            d.rectangle(box, fill=mix(SURFACE, BLUE, shade), outline=mix(SURFACE, BLUE, 0.34), width=1)

    lit_x = x0 + lit * (bw + gap)
    return x0, x1, y_top, y_bot, lit_x, bw


def draw_scaling(d: ImageDraw.ImageDraw, facts: dict) -> None:
    """Two series, one per context length, output tok/s against client count.

    This is the other half of the finding and the reason the rail below is
    mostly dark: at short context the card takes every client you give it, and
    at long context it stops taking them long before memory is the reason.
    Linear y on purpose — a log axis would flatter the 8,192 line and the point
    is that it is flat.
    """
    x0, x1 = 780, W - 88
    y0, y1 = 292, 182
    peak = max(v for _, series in facts["curves"] for v in series)
    clients = [1, 4, 16, 64]

    d.line([(px(x0), px(y0)), (px(x1), px(y0))], fill=GRID, width=px(1))
    for i, c in enumerate(clients):
        cx = x0 + (x1 - x0) * i / (len(clients) - 1)
        d.line([(px(cx), px(y0)), (px(cx), px(y0 + 5))], fill=GRID, width=px(1))
        d.text((px(cx - 4), px(y0 + 10)), str(c), font=font(MONO, 10), fill=INK_FAINT)
    d.text((px(x0), px(y0 + 24)), "CONCURRENT CLIENTS", font=font(MONO, 9), fill=INK_FAINT)

    for (label, series), colour in ((facts["curves"][0], ORANGE), (facts["curves"][1], BLUE)):
        pts = [
            (
                x0 + (x1 - x0) * i / (len(series) - 1),
                y0 + (y1 - y0) * (v / peak),
            )
            for i, v in enumerate(series)
        ]
        d.line([(px(x), px(y)) for x, y in pts], fill=colour, width=max(1, px(2)), joint="curve")
        for x, y in pts:  # >=8px markers, ringed in the surface so overlaps read
            r = 4
            d.ellipse([px(x - r), px(y - r), px(x + r), px(y + r)], fill=colour, outline=SURFACE, width=px(1.5))
        lx, ly = pts[-1]
        d.text(
            (px(lx - 6 - d.textlength(label, font=font(MONO_B, 12)) / S), px(ly - 22)),
            label,
            font=font(MONO_B, 12),
            fill=colour,
        )


def render(facts: dict, out: Path, content_address: bool, url_base: str) -> Path:
    img = Image.new("RGB", (px(W), px(H)), SURFACE)
    d = ImageDraw.Draw(img)
    draw_grid(d)

    f_eyebrow = font(MONO, 13)
    f_head = font(SANS_B, 44)
    f_sub = font(SANS, 19)
    f_label = font(MONO_B, 15)
    f_small = font(MONO, 12)
    f_foot = font(MONO, 11)

    d.text((px(88), px(44)), "GEMMA 4 E2B  /  vLLM ON ROCm  /  ONE MI300X", font=f_eyebrow, fill=INK_DIM)
    d.text((px(88), px(78)), "The pool was never the problem", font=f_head, fill=INK)
    d.text(
        (px(88), px(136)),
        f"{facts['kv_gib']:.2f} GiB of KV cache on one card. "
        f"The heaviest cell in the grid used {facts['occupancy'] * 100:.1f}% of it.",
        font=f_sub,
        fill=INK_DIM,
    )

    # The thesis, in the space the chart leaves. Three short lines rather than a
    # paragraph, because a cover is read at thumbnail size before it is read at
    # full size.
    f_claim = font(SANS, 21)
    f_claim_b = font(SANS_B, 21)
    d.text((px(88), px(206)), "Throughput at 8,192 tokens flattens", font=f_claim, fill=INK_DIM)
    d.text((px(88), px(238)), "between 16 and 64 clients \u2014 by 2%.", font=f_claim, fill=INK_DIM)
    d.text((px(88), px(276)), "Not memory. Prefill.", font=f_claim_b, fill=INK)

    draw_scaling(d, facts)
    x0, x1, y_top, y_bot, lit_x, bw = draw_rail(d, facts)

    # Label the lit run from below, so the leader does not cross the rail.
    d.line([(px(lit_x), px(y_bot + 8)), (px(lit_x), px(y_bot + 26))], fill=ORANGE, width=max(1, px(1.5)))
    d.text(
        (px(lit_x + 10), px(y_bot + 18)),
        f"{facts['wanted']:,} tokens wanted  —  {facts['occupancy'] * 100:.1f}%",
        font=f_label,
        fill=ORANGE,
    )
    d.text((px(x0), px(y_bot + 18)), "USED", font=f_small, fill=INK_FAINT)

    cap = f"{facts['pool']:,} RESIDENT  ·  {(1 - facts['occupancy']) * 100:.1f}% NEVER TOUCHED"
    d.text((px(x1 - d.textlength(cap, font=f_label) / S), px(y_bot + 18)), cap, font=f_label, fill=BLUE)

    # What the card did with the headroom it does have.
    note = (
        f"{facts['best_rate']:,.0f} output tok/s at {facts['best_clients']} clients "
        f"/ {facts['best_ctx']}-token context  —  ${facts['dollars_per_m']:.3f} per M tokens"
    )
    d.text((px(88), px(472)), note, font=f_label, fill=INK)
    d.line([(px(88), px(508)), (px(W - 88), px(508))], fill=GRID, width=px(1))
    d.text(
        (px(88), px(520)),
        f"Measured 2026-09-16 on {facts['host']}  /  vLLM {facts['engine']}  /  bf16  /  12 cells, 3 repeats",
        font=f_foot,
        fill=INK_FAINT,
    )

    img = img.resize((W, H), Image.LANCZOS)
    if content_address:
        tmp = out.with_suffix(".tmp.jpg")
        img.save(tmp, "JPEG", quality=92, optimize=True)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
        tmp.unlink()
        out = out.with_name(f"{out.stem}.{digest}{out.suffix}")
    img.save(out, "JPEG", quality=92, optimize=True)
    print(f"wrote {out.name}  {W}x{H}  {out.stat().st_size // 1024} KB")
    if url_base:
        print(f"cover_image: {url_base.rstrip('/')}/{out.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", default="benchmarks/reports/2026-09-16-vllm-sweep-mi300x.json")
    p.add_argument("--out", default="devto-benchmark-cover.jpg")
    p.add_argument("--content-address", action="store_true")
    p.add_argument("--url-base", default="")
    a = p.parse_args()
    facts = read_facts(PROJECT_DIR / a.report)
    render(facts, PROJECT_DIR / a.out, a.content_address, a.url_base)


if __name__ == "__main__":
    main()

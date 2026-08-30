#!/usr/bin/env python3
"""Render the cover for the pure-JAX Turing-vs-Ada article.

1600x900 to match the other covers in `docs/img/`, drawn at 2x and downsampled
so the type stays crisp. The output filename is CONTENT-ADDRESSED, because
Medium's importer caches by URL and ignores the query string -- a changed cover
has to arrive at a URL Medium has never seen.

This replaces `cover-medium-gde.jpg`, which showed a systolic-array die feeding a
GPU die. That image made a TPU claim the article no longer makes.

The cover is a STAT-TILE pair, not a chart: two numbers whose whole point is that
they disagree, measured on the same build. Colour is doing identity work only
(two categories, Turing vs Ada), so it uses categorical slots 1 and 2 from the
validated dark palette -- #3987e5 and #d95926 on surface #1a1a19, checked with
the palette validator rather than eyeballed:

    lightness band PASS · chroma floor PASS · CVD separation dE 26.8 protan
    · normal-vision dE 31.8 · contrast vs surface PASS

Numbers and labels are set in text tokens (white / #c3c2b7), never in the series
colour; the colour rides on a chip beside each tile so identity is never carried
by colour alone.

Sources for every figure on the cover:
  T4G  gpu-jax-g5g-2b/benchmarks/runs/2026-08-28-full-run-cached-g5g/summary.json
       convert 54.1% + gemv_fp32 32.8% = 86.9% -> "87%";  decode gauge 12.9 tok/s
  L4   gpu-jax-g6-2b/benchmarks/runs/2026-08-28-first-serve-g6/summary.json
       convert 0.0%;  decode gauge 48.4 tok/s (48.5/48.4/48.3 across the sweep)
  Both build id 51bc52c9e2e9, tpu_jax_weight_bytes 6,155,450,950.
"""

import hashlib
import pathlib

from PIL import Image, ImageDraw, ImageFont

S = 2  # supersample factor
W, H = 1600, 900

SURFACE = (26, 26, 25)  # #1a1a19
TILE = (32, 32, 31)
INK = (255, 255, 255)  # text-primary
INK_2 = (195, 194, 183)  # text-secondary
INK_3 = (128, 127, 120)  # muted
BLUE = (57, 135, 229)  # categorical slot 1 -- Ada / L4
ORANGE = (217, 89, 38)  # categorical slot 2 -- Turing / T4G
RULE = (58, 58, 55)

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
MONO_B = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"

OUT_DIR = pathlib.Path(__file__).parent / "img"


def f(path, size):
    return ImageFont.truetype(path, size * S)


def text(d, xy, s, font, fill, anchor="la"):
    d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor)


def main():
    img = Image.new("RGB", (W * S, H * S), SURFACE)
    d = ImageDraw.Draw(img)

    # A very subtle vignette so the flat surface does not read as a slide.
    for i in range(90):
        v = int(6 * (1 - i / 90))
        d.rectangle(
            [i * S, i * S, (W - i) * S, (H - i) * S], outline=(SURFACE[0] + v, SURFACE[1] + v, SURFACE[2] + v), width=S
        )

    # ---- eyebrow -----------------------------------------------------------
    text(d, (96, 96), "GEMMA 4 E2B   ·   PURE JAX 0.11.1   ·   AWS EC2", f(MONO, 21), INK_3)

    # ---- headline ----------------------------------------------------------
    text(d, (96, 146), "87% of decode went to", f(SANS_B, 66), INK)
    text(d, (96, 224), "converting numbers.", f(SANS_B, 66), INK)
    text(d, (96, 318), "One JAX port. A newer GPU made it 0.0%.", f(SANS, 40), INK_2)

    # ---- the two tiles -----------------------------------------------------
    # Vertical rhythm is explicit: the numeral needs its full line box before
    # the unit starts, and the claim has to sit clear at the tile foot.
    top, bot = 400, 744
    lx, rx = 96, 832
    tw = 672

    for x, chip, name, host, big, big_sub, claim in [
        (
            lx,
            ORANGE,
            "T4G  ·  Turing SM 7.5",
            "g5g.2xlarge  ·  aarch64",
            "87%",
            "of decode is dtype conversion",
            "12.9 tok/s decode",
        ),
        (
            rx,
            BLUE,
            "L4  ·  Ada SM 8.9",
            "g6.2xlarge  ·  x86_64",
            "0.0%",
            "storage and compute dtype agree",
            "48.4 tok/s decode",
        ),
    ]:
        d.rectangle([x * S, top * S, (x + tw) * S, bot * S], fill=TILE, outline=RULE, width=1 * S)
        # colour chip: identity lives here, not on the numerals or the labels
        d.rectangle([x * S, top * S, (x + 6) * S, bot * S], fill=chip)

        text(d, (x + 36, top + 30), name, f(MONO_B, 28), INK)
        text(d, (x + 36, top + 70), host, f(SANS, 21), INK_3)
        text(d, (x + 36, top + 108), big, f(SANS_B, 88), INK)
        text(d, (x + 36, top + 224), big_sub, f(SANS, 23), INK_2)
        # Claim: a colour swatch carries identity, the words stay in ink.
        d.rectangle([(x + 36) * S, (top + 288) * S, (x + 48) * S, (top + 300) * S], fill=chip)
        text(d, (x + 62, top + 285), claim, f(MONO_B, 21), INK_2)

    # ---- footer: the fact that makes the comparison fair -------------------
    d.line([96 * S, 812 * S, (W - 96) * S, 812 * S], fill=RULE, width=1 * S)
    text(
        d,
        (96, 838),
        "Same port, same build 51bc52c9e2e9, same 6,155,450,950 bytes of weights on both cards",
        f(MONO, 21),
        INK_3,
    )

    img = img.resize((W, H), Image.LANCZOS)

    OUT_DIR.mkdir(exist_ok=True)
    tmp = OUT_DIR / ".cover-gde.tmp.jpg"
    img.save(tmp, "JPEG", quality=92, optimize=True)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    out = OUT_DIR / f"cover-medium-gde-{digest}.jpg"
    tmp.replace(out)
    print(f"wrote {out}  {W}x{H}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

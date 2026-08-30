#!/usr/bin/env python3
"""Render devto-cover.jpg for the g4dn cost article.

1376x768 to match every other cover in this tree, drawn at 2x and downsampled so
the type is crisp on high-DPI screens.

The cover is a STAT-TILE pair, not a chart: two numbers whose whole point is that
they disagree about which instance is cheaper. Colour is doing identity work only
(two categories, Arm vs Intel), so it uses categorical slots 1 and 2 from the
validated dark palette -- #3987e5 and #d95926 on surface #1a1a19. That pair was
checked with the palette validator rather than eyeballed:

    lightness band PASS · chroma floor PASS · CVD separation dE 26.8 protan
    · normal-vision dE 31.8 · contrast vs surface PASS

Numbers and labels are set in text tokens (white / #c3c2b7), never in the series
colour; the colour rides on a chip beside each tile so identity is never carried
by colour alone.
"""

from PIL import Image, ImageDraw, ImageFont

S = 2  # supersample factor
W, H = 1376, 768

SURFACE = (26, 26, 25)          # #1a1a19
INK = (255, 255, 255)           # text-primary
INK_2 = (195, 194, 183)         # text-secondary
INK_3 = (128, 127, 120)         # muted
BLUE = (57, 135, 229)           # categorical slot 1 -- Intel / g4dn
ORANGE = (217, 89, 38)          # categorical slot 2 -- Arm / g5g
RULE = (58, 58, 55)

SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
MONO_B = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"


def f(path, size):
    return ImageFont.truetype(path, size * S)


def text(d, xy, s, font, fill, anchor="la", spacing=4):
    d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor,
           spacing=spacing * S)


def main():
    img = Image.new("RGB", (W * S, H * S), SURFACE)
    d = ImageDraw.Draw(img)

    # A very subtle vignette so the flat surface does not read as a slide.
    for i in range(80):
        v = int(6 * (1 - i / 80))
        d.rectangle([i * S, i * S, (W - i) * S, (H - i) * S],
                    outline=(SURFACE[0] + v, SURFACE[1] + v, SURFACE[2] + v), width=S)

    # ---- eyebrow -----------------------------------------------------------
    text(d, (88, 84), "GEMMA 4 E2B   ·   vLLM 0.28.0   ·   AWS EC2",
         f(MONO, 19), INK_3)

    # ---- headline ----------------------------------------------------------
    text(d, (88, 128), "The cheapest CUDA GPU", f(SANS_B, 62), INK)
    text(d, (88, 200), "on AWS has an Arm CPU.", f(SANS_B, 62), INK)
    text(d, (88, 284), "You probably want the Intel one.", f(SANS, 40), INK_2)

    # ---- the two tiles -----------------------------------------------------
    # Vertical rhythm is explicit because the first pass collided the claim
    # label with the unit line: the number needs its full line box before the
    # unit starts, and the claim needs to sit clear at the tile foot.
    top, bot = 372, 648
    lx, rx = 88, 726
    tw = 562

    for x, chip, name, host, big, big_sub, claim in [
        (lx, ORANGE, "g5g.xlarge", "Graviton2  ·  NVIDIA T4G",
         "$0.42", "per hour", "CHEAPEST PER HOUR"),
        (rx, BLUE, "g4dn.xlarge", "Intel  ·  NVIDIA T4",
         "$0.603", "per million output tokens", "CHEAPEST PER TOKEN"),
    ]:
        d.rectangle([x * S, top * S, (x + tw) * S, bot * S],
                    fill=(32, 32, 31), outline=RULE, width=1 * S)
        # colour chip: identity lives here, not on the numerals or the labels
        d.rectangle([x * S, top * S, (x + 6) * S, bot * S], fill=chip)

        text(d, (x + 34, top + 26), name, f(MONO_B, 27), INK)
        text(d, (x + 34, top + 62), host, f(SANS, 20), INK_3)
        text(d, (x + 34, top + 98), big, f(SANS_B, 72), INK)
        text(d, (x + 34, top + 190), big_sub, f(SANS, 22), INK_2)
        # Claim: a colour swatch carries identity, the words stay in ink.
        d.rectangle([(x + 34) * S, (top + 234) * S, (x + 46) * S, (top + 246) * S],
                    fill=chip)
        text(d, (x + 58, top + 231), claim, f(MONO_B, 19), INK_2)

    # ---- footer: the fact that makes the comparison fair -------------------
    d.line([88 * S, 686 * S, (W - 88) * S, 686 * S], fill=RULE, width=1 * S)
    text(d, (88, 710),
         "Same GPU either way:  SM 7.5  ·  15,360 MiB  ·  320.1 GB/s  ·  "
         "KV cache 329,579 tokens on both",
         f(MONO, 20), INK_3)

    img = img.resize((W, H), Image.LANCZOS)
    out = "devto-cover.jpg"
    img.save(out, "JPEG", quality=92, optimize=True)
    print(f"wrote {out}  {W}x{H}")


if __name__ == "__main__":
    main()

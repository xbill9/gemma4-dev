"""Cover art for devto-gemma4-t4-qat.md: Tux beside a low-profile T4-style card
whose CUDA cores glow green, blue wireframe racks behind, near-black ground.

House treatment of the repo's other covers (flat-vector, near-black, one warm light
source, cold blue wireframe), drawn in code so it is reproducible. No brand marks:
CUDA is carried by the grid of lit cores, never by a logo. Drawn at 2x and
downsampled to dev.to's displayed 1376x578 (2.381:1), then named by the first 8 hex
of its sha256, the kit's content-address rule.

    python3 docs/cover_art.py <out-dir>
"""

import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 2
W, H = 1376 * S, 578 * S
BG = (23, 22, 26)
GRID = (22, 52, 58)
BLUE = (57, 135, 229)
GREEN = (118, 185, 0)
ORANGE = (217, 89, 38)
INK = (242, 242, 240)
MUTED = (150, 150, 152)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Isometric projection for the card: x runs down-right, y down-left, z up.
ORIGIN = (640 * S, 250 * S)
UX = (0.866 * S, 0.5 * S)
UY = (-0.866 * S, 0.5 * S)


def P(x, y, z, origin=ORIGIN):
    return (origin[0] + x * UX[0] + y * UY[0], origin[1] + x * UX[1] + y * UY[1] - z * S)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def floor_and_racks(img):
    d = ImageDraw.Draw(img, "RGBA")
    horizon = int(H * 0.50)
    vp = (int(W * 0.60), horizon - 40 * S)
    for i in range(-24, 25):
        x = vp[0] + i * 150 * S
        d.line([vp, (x, H)], fill=GRID + (140,), width=S)
    y, step = horizon + 6 * S, 6 * S
    while y < H:
        d.line([(0, y), (W, y)], fill=GRID + (140,), width=S)
        step *= 1.28
        y += int(step)
    # Receding rows of rack outlines on both sides of the vanishing point.
    for side in (-1, 1):
        for k in range(9):
            t = k / 9
            scale = 1 - t * 0.85
            cx = vp[0] + side * (330 + 520 * scale) * S * 0.55
            w, h = 90 * S * scale, 230 * S * scale
            base = horizon + 40 * S * scale
            a = int(150 * scale + 30)
            box = [cx - w / 2, base - h, cx + w / 2, base]
            if box[0] < 640 * S:  # keep the text column clear
                continue
            d.rectangle(box, outline=BLUE + (a,), width=S)
            for r in range(1, 6):
                yy = base - h * r / 6
                d.line([(box[0], yy), (box[2], yy)], fill=BLUE + (a // 2,), width=1)


def card(img):
    L, Wd, Ht = 330, 120, 20  # low-profile, single slot: long, narrow, thin
    glow = Image.new("RGB", img.size, (0, 0, 0))
    g = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(img, "RGBA")

    # Pool of green light on the floor under the card.
    cx, cy = P(L / 2, Wd / 2, -40)
    g.ellipse([cx - 420 * S, cy - 70 * S, cx + 420 * S, cy + 90 * S], fill=mix((0, 0, 0), GREEN, 0.22))

    # Bracket at the far end, standing up.
    d.polygon([P(0, -6, -4), P(0, Wd + 6, -4), P(0, Wd + 6, Ht + 70), P(0, -6, Ht + 70)],
              fill=(88, 90, 96), outline=(130, 132, 138))
    # Faces.
    d.polygon([P(0, Wd, 0), P(L, Wd, 0), P(L, Wd, Ht), P(0, Wd, Ht)], fill=(34, 36, 40))
    d.polygon([P(L, 0, 0), P(L, Wd, 0), P(L, Wd, Ht), P(L, 0, Ht)], fill=(46, 48, 54))
    d.polygon([P(0, 0, Ht), P(L, 0, Ht), P(L, Wd, Ht), P(0, Wd, Ht)], fill=(62, 64, 70))
    # Gold fingers along the lower edge of the long face.
    for i in range(30):
        x0 = 60 + i * 5.2
        d.polygon([P(x0, Wd, 0), P(x0 + 3.2, Wd, 0), P(x0 + 3.2, Wd, 7), P(x0, Wd, 7)], fill=(201, 160, 72))
    # Heatsink fins on the top face, either side of the cut-away window.
    win = (95, 270, 18, Wd - 18)
    for j in range(8, Wd - 4, 9):
        for xa, xb in ((6, win[0] - 6), (win[1] + 6, L - 6)):
            d.line([P(xa, j, Ht), P(xb, j, Ht)], fill=(118, 122, 130), width=2 * S)
            d.line([P(xa, j + 3, Ht), P(xb, j + 3, Ht)], fill=(36, 38, 42), width=S)
    # The cut-away: a dark die holding a regular grid of lit cores.
    x0, x1, y0, y1 = win
    d.polygon([P(x0, y0, Ht), P(x1, y0, Ht), P(x1, y1, Ht), P(x0, y1, Ht)], fill=(12, 14, 10),
              outline=(150, 154, 160))
    n = 0
    for xi in range(int(x0) + 5, int(x1) - 4, 7):
        for yi in range(int(y0) + 5, int(y1) - 4, 7):
            n += 1
            bright = 1.0 if (xi * 7 + yi * 13) % 11 else 0.55
            col = mix((20, 30, 0), GREEN, bright)
            quad = [P(xi, yi, Ht), P(xi + 4.5, yi, Ht), P(xi + 4.5, yi + 4.5, Ht), P(xi, yi + 4.5, Ht)]
            d.polygon(quad, fill=col)
            g.polygon(quad, fill=col)
    # Green light spilling out between the fins.
    for j in range(8, Wd - 4, 9):
        g.line([P(win[1] + 6, j + 1.5, Ht), P(L - 6, j + 1.5, Ht)], fill=mix((0, 0, 0), GREEN, 0.45), width=2 * S)
    return glow, n


def tux(img, glow, fx, fy, h):
    """A flat-vector penguin standing with its feet at (fx, fy), h pixels tall."""
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    u = h / 100
    body = [fx - 30 * u, fy - 78 * u, fx + 30 * u, fy - 4 * u]
    head = [fx - 21 * u, fy - 100 * u, fx + 21 * u, fy - 62 * u]
    # Green rim light on the side facing the card.
    g.ellipse([body[0] - 4 * u, body[1] - 2 * u, body[2] - 20 * u, body[3]], fill=mix((0, 0, 0), GREEN, 0.9))
    g.ellipse([head[0] - 3 * u, head[1] - 2 * u, head[2] - 16 * u, head[3]], fill=mix((0, 0, 0), GREEN, 0.9))
    for foot in (-1, 1):
        d.ellipse([fx + foot * 14 * u - 13 * u, fy - 8 * u, fx + foot * 14 * u + 13 * u, fy + 2 * u], fill=ORANGE)
    d.ellipse([fx - 40 * u, fy - 62 * u, fx - 18 * u, fy - 18 * u], fill=(18, 18, 20))  # flipper
    d.ellipse([fx + 18 * u, fy - 62 * u, fx + 40 * u, fy - 18 * u], fill=(18, 18, 20))
    d.ellipse(body, fill=(18, 18, 20))
    d.ellipse([fx - 21 * u, fy - 66 * u, fx + 21 * u, fy - 8 * u], fill=(238, 236, 230))  # belly
    d.ellipse(head, fill=(18, 18, 20))
    for ex in (-8, 8):  # eyes, looking up and toward the card
        d.ellipse([fx + ex * u - 6 * u, fy - 91 * u, fx + ex * u + 6 * u, fy - 76 * u], fill=(245, 245, 245))
        d.ellipse([fx + ex * u - 5 * u, fy - 91 * u, fx + ex * u + 0.5 * u, fy - 83 * u], fill=(10, 10, 10))
    d.polygon([(fx - 9 * u, fy - 74 * u), (fx + 9 * u, fy - 74 * u), (fx - 2 * u, fy - 65 * u)], fill=ORANGE)


def text(img):
    d = ImageDraw.Draw(img)
    x = 72 * S
    d.text((x, 54 * S), "GEMMA 4 E2B · vLLM 0.29 · CUDA ON A TESLA T4", font=ImageFont.truetype(FONT_M, 17 * S), fill=MUTED)
    fb = ImageFont.truetype(FONT_B, 52 * S)
    d.text((x, 92 * S), "QAT decodes", font=fb, fill=INK)
    d.text((x, 154 * S), "1.79x faster", font=fb, fill=INK)
    fr = ImageFont.truetype(FONT_R, 22 * S)
    d.text((x, 228 * S), "Same T4, same 70 W.", font=fr, fill=MUTED)
    d.text((x, 260 * S), "72.31 vs 40.44 tok/s per stream", font=fr, fill=MUTED)
    d.line([(x, 520 * S), (W - 72 * S, 520 * S)], fill=(60, 60, 64), width=S)
    d.text((x, 534 * S), "github.com/xbill9/gemma4-dev · gpu-vllm-t4-2b", font=ImageFont.truetype(FONT_M, 16 * S), fill=MUTED)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    img = Image.new("RGB", (W, H), BG)
    floor_and_racks(img)
    glow, cores = card(img)
    tux(img, glow, fx=1215 * S, fy=470 * S, h=190 * S)
    # Additive glow: blurred copy of every lit element, then the sharp image on top.
    halo = glow.filter(ImageFilter.GaussianBlur(26 * S))
    px = [min(255, a + b) for a, b in zip(img.tobytes(), halo.tobytes())]
    img = Image.frombytes("RGB", img.size, bytes(px))
    # Re-draw the penguin sharp over its own halo.
    tux(img, Image.new("RGB", img.size), fx=1215 * S, fy=470 * S, h=190 * S)
    text(img)
    img = img.resize((W // S, H // S), Image.LANCZOS)
    tmp = out / "cover.tmp.jpg"
    img.save(tmp, "JPEG", quality=90, optimize=True)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    final = out / f"devto-t4-qat-cover.{digest}.jpg"
    tmp.rename(final)
    print(f"wrote {final.name}  {img.size[0]}x{img.size[1]}  {final.stat().st_size // 1024} KB  cores={cores}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Textless cover: five engines as light trails racing through a lambda arch toward one GPU card.

Drawn at 2x and downsampled; writes devto-cover.<hash>.jpg (1376x578) and builder-cover.<hash>.jpg (1200x675).
Palette from publishing-kit's make-cover.py (surface, paper, blue, orange) plus three trail colours.
"""

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
SURFACE = (26, 26, 25)
ORANGE = (255, 153, 0)
BLUE = (66, 133, 244)
TRAILS = [(255, 153, 0), (66, 133, 244), (52, 199, 160), (234, 84, 85), (250, 204, 21)]
# Relative trail lengths: the article's one-request medians, longest first (drawn, not labelled).
LENGTHS = [1.0, 0.81, 0.58, 0.57, 0.22]


def draw(w, h):
    S = 2
    W, H = w * S, h * S
    img = Image.new("RGB", (W, H), SURFACE)
    # Sky gradient toward a faint blue horizon.
    sky = ImageDraw.Draw(img)
    for y in range(H):
        k = max(0.0, 1 - abs(y / H - 0.55) * 2)
        sky.line([0, y, W, y], fill=tuple(int(SURFACE[i] * (1 - 0.35 * k) + BLUE[i] * 0.10 * k) for i in range(3)))
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(img)

    cy = H * 0.52
    left, right = W * 0.06, W * 0.80
    lane = H * 0.055
    # GPU card on the right: a board with a fan, green-free, blue rim.
    cx0, cy0 = W * 0.80, cy - H * 0.17
    cw, ch = W * 0.14, H * 0.34
    d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + ch], radius=14 * S, fill=(44, 44, 42), outline=BLUE, width=3 * S)
    fx, fy, fr = cx0 + cw * 0.5, cy0 + ch * 0.5, ch * 0.30
    d.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=(32, 32, 31), outline=(90, 90, 86), width=3 * S)
    for k in range(7):
        a = k * 2 * math.pi / 7
        d.line([fx, fy, fx + fr * 0.9 * math.cos(a), fy + fr * 0.9 * math.sin(a)], fill=(90, 90, 86), width=4 * S)
    d.ellipse([fx - fr * 0.25, fy - fr * 0.25, fx + fr * 0.25, fy + fr * 0.25], fill=(120, 120, 114))
    for k in range(9):  # gold contacts along the bottom edge
        x = cx0 + cw * (0.12 + k * 0.085)
        d.rectangle([x, cy0 + ch, x + cw * 0.05, cy0 + ch + 8 * S], fill=(200, 160, 60))
    g.rounded_rectangle([cx0 - 10 * S, cy0 - 10 * S, cx0 + cw + 10 * S, cy0 + ch + 10 * S], radius=20 * S, fill=(30, 60, 120))

    # Lambda arch, drawn as two strokes.
    lx, ltop, lbot = W * 0.40, cy - H * 0.36, cy + H * 0.33
    stroke = 18 * S
    top, foot = (lx - W * 0.03, ltop - H * 0.02), (lx + W * 0.085, lbot)
    t = 0.48  # the short stroke leaves the long one just below its middle, as in the letter
    join = (top[0] + (foot[0] - top[0]) * t, top[1] + (foot[1] - top[1]) * t)
    for (a, b) in [(top, foot), (join, (lx - W * 0.075, lbot))]:
        g.line([a, b], fill=(160, 90, 0), width=stroke * 3)
        d.line([a, b], fill=ORANGE, width=stroke)

    # Five trails from the left, ending at their length; a bright head on each.
    for i, (col, ln) in enumerate(zip(TRAILS, LENGTHS)):
        y = cy + (i - 2) * lane
        x_end = left + (right - left) * ln
        steps = 60
        for s in range(steps):
            t0, t1 = s / steps, (s + 1) / steps
            xa, xb = left + (x_end - left) * t0, left + (x_end - left) * t1
            a = t1 ** 2
            c = tuple(int(SURFACE[j] * (1 - a) + col[j] * a) for j in range(3))
            d.line([xa, y, xb, y], fill=c, width=int(7 * S * (0.4 + 0.6 * t1)))
        g.ellipse([x_end - 26 * S, y - 26 * S, x_end + 26 * S, y + 26 * S], fill=tuple(int(v * 0.8) for v in col))
        d.ellipse([x_end - 9 * S, y - 9 * S, x_end + 9 * S, y + 9 * S], fill=(255, 255, 255))

    glow = glow.filter(ImageFilter.GaussianBlur(28 * S))
    return add(img, glow).resize((w, h), Image.LANCZOS)


def add(a, b):
    """a + b per channel, clipped."""
    return Image.merge("RGB", [Image.eval(Image.blend(x, y, 0.5), lambda v: min(255, v * 2)) for x, y in zip(a.split(), b.split())])


for name, size in (("devto-cover", (1376, 578)), ("builder-cover", (1200, 675))):
    im = draw(*size)
    tmp = HERE / f"{name}.tmp.jpg"
    im.save(tmp, quality=90)
    h = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    dest = HERE / f"{name}.{h}.jpg"
    tmp.rename(dest)
    print(dest.name, size)

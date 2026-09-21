"""Cover art for devto-gemma4-t4-vm-deploy.md: the four things the article stacks,
drawn as a chain — a Compute Engine cloud holding a VM, Tux for the Debian guest,
a T4 card with its cores lit green, and vLLM's paged KV blocks.

Same house treatment as docs/cover_art.py (flat-vector, near-black ground, one warm
light source, cold blue wireframe) and the same rule: NO BRAND MARKS. Each vendor is
carried by its characteristic form and colour — a cloud outline, a penguin, a grid of
lit cores, a block table — never by a logo. Drawn at 2x and downsampled to dev.to's
displayed 1376x578 (2.381:1), then named by the first 8 hex of its sha256, the kit's
content-address rule.

    python3 docs/cover_art_vm.py <out-dir>
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
PANEL = (31, 31, 36)
EDGE = (58, 58, 66)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# The chain of four emblems occupies the right of the canvas.
TILE_Y = 196 * S          # top of each emblem panel
TILE_W, TILE_H = 132 * S, 132 * S
TILE_X = [688 * S, 848 * S, 1008 * S, 1168 * S]


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def font(path, pt):
    return ImageFont.truetype(path, int(pt * S))


def backdrop(img):
    """Cold blue wireframe receding to a vanishing point behind the chain."""
    d = ImageDraw.Draw(img, "RGBA")
    horizon = int(H * 0.56)
    vp = (int(W * 0.71), horizon - 30 * S)
    for i in range(-26, 27):
        d.line([vp, (vp[0] + i * 160 * S, H)], fill=GRID + (120,), width=S)
    y, step = horizon + 6 * S, 6 * S
    while y < H:
        d.line([(0, y), (W, y)], fill=GRID + (120,), width=S)
        step *= 1.30
        y += int(step)


def panel(d, x, caption):
    """The rounded plate every emblem sits on, with its caption underneath."""
    d.rounded_rectangle([x, TILE_Y, x + TILE_W, TILE_Y + TILE_H], radius=14 * S,
                        fill=PANEL, outline=EDGE, width=S)
    f = font(FONT_MB, 11)
    w = d.textlength(caption, font=f)
    d.text((x + (TILE_W - w) / 2, TILE_Y + TILE_H + 14 * S), caption, font=f, fill=MUTED)


def arrow(d, x0, x1, y):
    d.line([(x0, y), (x1 - 7 * S, y)], fill=EDGE, width=2 * S)
    d.polygon([(x1, y), (x1 - 9 * S, y - 5 * S), (x1 - 9 * S, y + 5 * S)], fill=EDGE)


def emblem_cloud(img, glow, x):
    """Compute Engine: a cloud outline with a VM's three racked units inside."""
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    cx, cy = x + TILE_W / 2, TILE_Y + TILE_H / 2
    # Cloud silhouette: three overlapping discs on a rounded base.
    lobes = [(-30, -6, 26), (0, -20, 32), (30, -4, 24)]
    for ox, oy, r in lobes:
        box = [cx + (ox - r) * S, cy + (oy - r) * S, cx + (ox + r) * S, cy + (oy + r) * S]
        d.ellipse(box, fill=mix(BG, BLUE, 0.16), outline=BLUE, width=2 * S)
        g.ellipse(box, fill=mix((0, 0, 0), BLUE, 0.28))
    base = [cx - 52 * S, cy + 2 * S, cx + 52 * S, cy + 30 * S]
    d.rounded_rectangle(base, radius=13 * S, fill=mix(BG, BLUE, 0.16), outline=BLUE, width=2 * S)
    g.rounded_rectangle(base, radius=13 * S, fill=mix((0, 0, 0), BLUE, 0.28))
    # Erase the interior seams so the lobes read as one cloud.
    d.ellipse([cx - 30 * S, cy - 26 * S, cx + 30 * S, cy + 20 * S], fill=mix(BG, BLUE, 0.16))
    d.rectangle([cx - 48 * S, cy + 2 * S, cx + 48 * S, cy + 16 * S], fill=mix(BG, BLUE, 0.16))
    # The VM inside: three racked units, the middle one lit.
    for i, ry in enumerate((-16, -3, 10)):
        unit = [cx - 26 * S, cy + ry * S, cx + 26 * S, cy + (ry + 10) * S]
        lit = i == 1
        d.rounded_rectangle(unit, radius=3 * S, fill=(16, 17, 20),
                            outline=BLUE if lit else EDGE, width=S)
        for k in range(3):
            dx = cx - 20 * S + k * 6 * S
            d.rectangle([dx, cy + (ry + 3) * S, dx + 3 * S, cy + (ry + 7) * S],
                        fill=BLUE if lit else (70, 72, 80))
        if lit:
            d.ellipse([cx + 17 * S, cy + (ry + 3) * S, cx + 21 * S, cy + (ry + 7) * S], fill=GREEN)
            g.ellipse([cx + 15 * S, cy + (ry + 1) * S, cx + 23 * S, cy + (ry + 9) * S],
                      fill=mix((0, 0, 0), GREEN, 0.8))


def emblem_tux(img, glow, x):
    """Debian guest: a flat-vector penguin, lit green from the card on its left."""
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    fx, fy, h = x + TILE_W / 2, TILE_Y + TILE_H - 20 * S, 96 * S
    u = h / 100
    body = [fx - 31 * u, fy - 76 * u, fx + 31 * u, fy - 4 * u]
    head = [fx - 22 * u, fy - 99 * u, fx + 22 * u, fy - 60 * u]
    g.ellipse([body[0] + 16 * u, body[1], body[2] + 5 * u, body[3]], fill=mix((0, 0, 0), GREEN, 0.75))
    g.ellipse([head[0] + 12 * u, head[1], head[2] + 4 * u, head[3]], fill=mix((0, 0, 0), GREEN, 0.75))
    for foot in (-1, 1):
        d.ellipse([fx + foot * 15 * u - 14 * u, fy - 8 * u, fx + foot * 15 * u + 14 * u, fy + 3 * u],
                  fill=ORANGE)
    d.ellipse([fx - 41 * u, fy - 60 * u, fx - 19 * u, fy - 16 * u], fill=(18, 18, 20))
    d.ellipse([fx + 19 * u, fy - 60 * u, fx + 41 * u, fy - 16 * u], fill=(18, 18, 20))
    d.ellipse(body, fill=(18, 18, 20))
    d.ellipse([fx - 22 * u, fy - 64 * u, fx + 22 * u, fy - 7 * u], fill=(238, 236, 230))
    d.ellipse(head, fill=(18, 18, 20))
    for ex in (-8, 8):
        d.ellipse([fx + ex * u - 6.5 * u, fy - 90 * u, fx + ex * u + 6.5 * u, fy - 74 * u],
                  fill=(245, 245, 245))
        d.ellipse([fx + ex * u - 5 * u, fy - 89 * u, fx + ex * u + 1 * u, fy - 80 * u], fill=(10, 10, 10))
    d.polygon([(fx - 9 * u, fy - 73 * u), (fx + 9 * u, fy - 73 * u), (fx - 1 * u, fy - 63 * u)],
              fill=ORANGE)


def emblem_card(img, glow, x):
    """The T4: a low-profile card face-on, its die a grid of lit cores. No logo."""
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    cx, cy = x + TILE_W / 2, TILE_Y + TILE_H / 2
    board = [cx - 50 * S, cy - 34 * S, cx + 50 * S, cy + 28 * S]
    d.rounded_rectangle(board, radius=5 * S, fill=(30, 32, 36), outline=(96, 100, 108), width=S)
    # Gold fingers along the bottom edge.
    for i in range(14):
        fx0 = cx - 40 * S + i * 5.8 * S
        d.rectangle([fx0, cy + 28 * S, fx0 + 3.6 * S, cy + 36 * S], fill=(201, 160, 72))
    # Mounting bracket on the left.
    d.rectangle([cx - 58 * S, cy - 40 * S, cx - 50 * S, cy + 30 * S], fill=(88, 90, 96))
    # Heatsink fins across the board.
    for i in range(9):
        fy0 = cy - 30 * S + i * 6.6 * S
        d.line([(cx - 44 * S, fy0), (cx - 20 * S, fy0)], fill=(112, 116, 124), width=2 * S)
        d.line([(cx + 20 * S, fy0), (cx + 44 * S, fy0)], fill=(112, 116, 124), width=2 * S)
    # The die: a regular grid of cores, most lit.
    die = [cx - 17 * S, cy - 21 * S, cx + 17 * S, cy + 15 * S]
    d.rectangle(die, fill=(12, 14, 10), outline=(150, 154, 160), width=S)
    n = 0
    step = 5.6 * S
    yy = die[1] + 3 * S
    while yy + 4 * S <= die[3] - 1 * S:
        xx = die[0] + 3 * S
        while xx + 4 * S <= die[2] - 1 * S:
            n += 1
            bright = 1.0 if (int(xx) * 7 + int(yy) * 13) % 11 else 0.5
            col = mix((20, 30, 0), GREEN, bright)
            d.rectangle([xx, yy, xx + 3.6 * S, yy + 3.6 * S], fill=col)
            g.rectangle([xx, yy, xx + 3.6 * S, yy + 3.6 * S], fill=col)
            xx += step
        yy += step
    g.rectangle([die[0] - 3 * S, die[1] - 3 * S, die[2] + 3 * S, die[3] + 3 * S],
                fill=mix((0, 0, 0), GREEN, 0.35))
    return n


def emblem_paged(img, glow, x):
    """vLLM: the paged KV block table, filled blocks warm, free blocks cold."""
    d = ImageDraw.Draw(img, "RGBA")
    g = ImageDraw.Draw(glow)
    cx, cy = x + TILE_W / 2, TILE_Y + TILE_H / 2
    cols, rows = 5, 4
    bw, bh, gap = 16 * S, 12 * S, 4 * S
    gw = cols * bw + (cols - 1) * gap
    gh = rows * bh + (rows - 1) * gap
    x0, y0 = cx - gw / 2, cy - gh / 2 - 6 * S
    filled = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (3, 0), (2, 1)}
    for r in range(rows):
        for c in range(cols):
            bx, by = x0 + c * (bw + gap), y0 + r * (bh + gap)
            box = [bx, by, bx + bw, by + bh]
            if (c, r) in filled:
                d.rounded_rectangle(box, radius=2 * S, fill=mix(BG, ORANGE, 0.85))
                g.rounded_rectangle(box, radius=2 * S, fill=mix((0, 0, 0), ORANGE, 0.55))
            elif r == rows - 1 and c < 2:
                d.rounded_rectangle(box, radius=2 * S, fill=mix(BG, BLUE, 0.7))
                g.rounded_rectangle(box, radius=2 * S, fill=mix((0, 0, 0), BLUE, 0.4))
            else:
                d.rounded_rectangle(box, radius=2 * S, fill=(38, 39, 45), outline=EDGE, width=S)
    f = font(FONT_MB, 15)
    w = d.textlength("vLLM", font=f)
    d.text((cx - w / 2, y0 + gh + 9 * S), "vLLM", font=f, fill=INK)


def text(img, d):
    x = 72 * S
    d.text((x, 150 * S), "GEMMA 4 E2B · ONE TESLA T4 ON COMPUTE ENGINE",
           font=font(FONT_M, 17), fill=MUTED)
    fb = font(FONT_B, 52)
    d.text((x, 188 * S), "The minimum", font=fb, fill=INK)
    d.text((x, 250 * S), "GCE VM", font=fb, fill=INK)
    fr = font(FONT_R, 22)
    d.text((x, 324 * S), "n1-standard-2 · Debian 13 · no driver preinstalled", font=fr, fill=MUTED)
    d.text((x, 356 * S), "362 s from start to a healthy endpoint", font=fr, fill=MUTED)
    d.line([(x, 500 * S), (W - 72 * S, 500 * S)], fill=(60, 60, 64), width=S)
    d.text((x, 516 * S), "github.com/xbill9/gemma4-dev · gpu-vllm-t4-2b · part 2",
           font=font(FONT_M, 16), fill=MUTED)


def draw_chain(img, glow):
    d = ImageDraw.Draw(img, "RGBA")
    for cx_ in TILE_X:
        panel(d, cx_, "")
    emblem_cloud(img, glow, TILE_X[0])
    emblem_tux(img, glow, TILE_X[1])
    cores = emblem_card(img, glow, TILE_X[2])
    emblem_paged(img, glow, TILE_X[3])
    mid = TILE_Y + TILE_H / 2
    for i in range(3):
        arrow(d, TILE_X[i] + TILE_W + 6 * S, TILE_X[i + 1] - 6 * S, mid)
    for cx_, cap in zip(TILE_X, ("COMPUTE ENGINE", "DEBIAN 13", "TESLA T4", "vLLM 0.29")):
        f = font(FONT_MB, 11)
        w = d.textlength(cap, font=f)
        d.text((cx_ + (TILE_W - w) / 2, TILE_Y + TILE_H + 14 * S), cap, font=f, fill=MUTED)
    return cores


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    img = Image.new("RGB", (W, H), BG)
    backdrop(img)
    glow = Image.new("RGB", img.size, (0, 0, 0))
    cores = draw_chain(img, glow)
    # Additive glow: blurred copy of every lit element, then the sharp art on top.
    halo = glow.filter(ImageFilter.GaussianBlur(22 * S))
    px = [min(255, a + b) for a, b in zip(img.tobytes(), halo.tobytes())]
    img = Image.frombytes("RGB", img.size, bytes(px))
    draw_chain(img, Image.new("RGB", img.size))
    text(img, ImageDraw.Draw(img))
    img = img.resize((W // S, H // S), Image.LANCZOS)
    tmp = out / "cover.vm.tmp.jpg"
    img.save(tmp, "JPEG", quality=90, optimize=True)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    final = out / f"devto-t4-vm-cover.{digest}.jpg"
    tmp.rename(final)
    print(f"wrote {final.name}  {img.size[0]}x{img.size[1]}  "
          f"{final.stat().st_size // 1024} KB  cores={cores}")


if __name__ == "__main__":
    main()

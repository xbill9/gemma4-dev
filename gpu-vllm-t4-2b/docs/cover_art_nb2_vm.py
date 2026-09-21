"""Cover art for devto-gemma4-t4-vm-deploy.md from an nb2lite-generated illustration.

Companion to cover_art_nb2.py, which does the same job for part 1, and to
cover_art_vm.py, which draws this article's cover entirely in PIL. This one takes
a generated 16:9 illustration and does the parts an image model cannot be trusted
with: exact geometry, legible type, and a reproducible name.

The model is asked for art only, never for text -- it renders lettering
unreliably, and the kit's rule is that a cover's numerals wear ink tokens rather
than riding on generated pixels. The prompt that produced the illustration lives
in docs/cover_art_nb2_vm.prompt.txt so the image is reproducible.

Three things happen here:

  1. The illustration is fitted to dev.to's DISPLAYED 1376x578 (2.381:1). dev.to
     renders width=1000,height=420,fit=cover, so anything authored at another
     ratio loses a band off the top and bottom.
  2. A left-hand scrim darkens the text column, so the type reads whatever the
     model drew there. The prompt reserves that space; the scrim is the guard for
     when it does not honour it.
  3. The file is named by the first 8 hex of its sha256. A cover URL is a mutable
     name that dev.to proxies and caches, so a regenerated cover needs a URL
     nothing has seen. Identical bytes reproduce the identical name.

    python3 docs/cover_art_nb2_vm.py <generated.png> [out-dir]
"""

import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 1376 * S, 578 * S
BG = (23, 22, 26)
INK = (242, 242, 240)
MUTED = (150, 150, 152)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

EYEBROW = "GEMMA 4 E2B · n1-standard-2 · ONE TESLA T4 · DEBIAN 13"
HEAD = ("2 vCPU, 7.8 GB,", "one T4")
SUB = ("Debian 13 ships no driver.", "16 GB of swap, or the loader is killed.")
FOOTER = "github.com/xbill9/gemma4-dev · gpu-vllm-t4-2b"

COLUMN = int(W * 0.55)  # the scrimmed text column; type must not cross it


def fit(src):
    """Cover-fit the illustration to WxH, cropping the long axis symmetrically."""
    img = Image.open(src).convert("RGB")
    scale = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((left, top, left + W, top + H))


def scrim(img):
    """Fade the left 55% to near-black so the text column reads over any art."""
    veil = Image.new("RGBA", img.size, BG + (0,))
    d = ImageDraw.Draw(veil)
    edge = int(W * 0.55)
    for x in range(edge):
        t = 1 - (x / edge) ** 1.6  # opaque at the left margin, clear by the edge
        d.line([(x, 0), (x, H)], fill=BG + (int(238 * t),))
    return Image.alpha_composite(img.convert("RGBA"), veil).convert("RGB")


def text(img):
    """The same positions cover_art_nb2.py sets, with this article's wording."""
    d = ImageDraw.Draw(img)
    x = 72 * S
    widest = 0

    def put(xy, s, font, fill):
        nonlocal widest
        d.text(xy, s, font=font, fill=fill)
        widest = max(widest, xy[0] + d.textlength(s, font=font))

    put((x, 54 * S), EYEBROW, ImageFont.truetype(FONT_M, 15 * S), MUTED)
    fb = ImageFont.truetype(FONT_B, 52 * S)
    put((x, 92 * S), HEAD[0], fb, INK)
    put((x, 154 * S), HEAD[1], fb, INK)
    fr = ImageFont.truetype(FONT_R, 22 * S)
    put((x, 228 * S), SUB[0], fr, MUTED)
    put((x, 260 * S), SUB[1], fr, MUTED)
    d.line([(x, 520 * S), (W - 72 * S, 520 * S)], fill=(60, 60, 64), width=S)
    put((x, 534 * S), FOOTER, ImageFont.truetype(FONT_M, 16 * S), MUTED)
    return widest


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.rstrip().splitlines()[-1].strip())
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else src.parent)
    img = scrim(fit(src))
    widest = text(img)
    img = img.resize((W // S, H // S), Image.LANCZOS)
    tmp = out / "cover.nb2vm.tmp.jpg"
    img.save(tmp, "JPEG", quality=90, optimize=True)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    final = out / f"devto-t4-vm-cover.{digest}.jpg"
    tmp.rename(final)
    print(f"wrote {final.name}  {img.size[0]}x{img.size[1]}  {final.stat().st_size // 1024} KB  from {src.name}")
    print(f"widest line ends at {widest / S:.0f}px of the {COLUMN / S:.0f}px text column")
    if widest > COLUMN:
        print("WARNING: type crosses the scrim edge and will sit on the illustration")


if __name__ == "__main__":
    main()

"""Cover art for devto-gemma4-t4-qat.md from an nb2lite-generated illustration.

Companion to cover_art.py, which draws the same cover entirely in PIL. This one
takes a generated 16:9 illustration and does the parts an image model cannot be
trusted with: exact geometry, legible type, and a reproducible name.

The model is asked for art only, never for text -- it renders lettering
unreliably, and the kit's rule is that a cover's numerals wear ink tokens rather
than riding on generated pixels. The prompt that produced the illustration lives
in docs/cover_art_nb2.prompt.txt so the image is reproducible.

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

    python3 docs/cover_art_nb2.py <generated.png> [out-dir]
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
    """The same wording cover_art.py sets, at the same positions."""
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
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.rstrip().splitlines()[-1].strip())
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else src.parent)
    img = scrim(fit(src))
    text(img)
    img = img.resize((W // S, H // S), Image.LANCZOS)
    tmp = out / "cover.nb2.tmp.jpg"
    img.save(tmp, "JPEG", quality=90, optimize=True)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    final = out / f"devto-t4-qat-cover.{digest}.jpg"
    tmp.rename(final)
    print(f"wrote {final.name}  {img.size[0]}x{img.size[1]}  {final.stat().st_size // 1024} KB  from {src.name}")


if __name__ == "__main__":
    main()

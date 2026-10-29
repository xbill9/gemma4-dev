import json, sys
H = "h76c0f9b50a7270c1_1_"
R = []          # requests in order
ORDER = []      # final slide order
PT = lambda v: {"magnitude": v, "unit": "PT"}
def rgb(h): h = h.lstrip("#"); return {"red": int(h[0:2], 16)/255, "green": int(h[2:4], 16)/255, "blue": int(h[4:6], 16)/255}
INK, GREY = "#1e1e1e", "#5f6368"
YEL, BLU, PNK, GRN = "#ffe7a5", "#c3ecf6", "#f7d8d8", "#ccf6c5"
BBLU, BGRN, BPNK, BYEL, ORG, GBLUE = "#57caff", "#5cdb6c", "#ff7dae", "#ffd427", "#f9ab00", "#4285f4"
GS, MONO = "Google Sans", "Roboto Mono"

def style(oid, size, bold=False, color=INK, font=GS, rng=None, cell=None):
    r = {"objectId": oid, "style": {"fontFamily": font, "fontSize": PT(size), "bold": bold,
         "foregroundColor": {"opaqueColor": {"rgbColor": rgb(color)}}},
         "fields": "fontFamily,fontSize,bold,foregroundColor",
         "textRange": rng or {"type": "ALL"}}
    if cell: r["cellLocation"] = cell
    R.append({"updateTextStyle": r})

def rng(a, b): return {"type": "FIXED_RANGE", "startIndex": a, "endIndex": b}

def shape(sid, oid, kind, x, y, w, h, fill=None, line=INK, weight=1):
    R.append({"createShape": {"objectId": oid, "shapeType": kind, "elementProperties": {"pageObjectId": sid,
        "size": {"width": PT(w), "height": PT(h)}, "transform": {"scaleX": 1, "scaleY": 1, "translateX": x, "translateY": y, "unit": "PT"}}}})
    props, fields = {}, []
    if fill:
        props["shapeBackgroundFill"] = {"solidFill": {"color": {"rgbColor": rgb(fill)}}}; fields.append("shapeBackgroundFill")
    if line:
        props["outline"] = {"outlineFill": {"solidFill": {"color": {"rgbColor": rgb(line)}}}, "weight": PT(weight)}; fields.append("outline")
    else:
        props["outline"] = {"propertyState": "NOT_RENDERED"}; fields.append("outline")
    R.append({"updateShapeProperties": {"objectId": oid, "shapeProperties": props, "fields": ",".join(fields)}})

def text(sid, oid, x, y, w, h, t, size=13, bold=False, color=INK, font=GS, fill=None, line=None, kind="TEXT_BOX",
         bullets=False, align=None, valign=None):
    shape(sid, oid, kind, x, y, w, h, fill, line)
    R.append({"insertText": {"objectId": oid, "text": t}})
    style(oid, size, bold, color, font)
    if bullets:
        R.append({"createParagraphBullets": {"objectId": oid, "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE", "textRange": {"type": "ALL"}}})
    if align:
        R.append({"updateParagraphStyle": {"objectId": oid, "style": {"alignment": align}, "fields": "alignment", "textRange": {"type": "ALL"}}})
    if valign:
        R.append({"updateShapeProperties": {"objectId": oid, "shapeProperties": {"contentAlignment": valign}, "fields": "contentAlignment"}})

def card(sid, oid, x, y, w, h, t, fill="#ffffff", size=13, head=None, bullets=False, valign="TOP", center=False):
    """Rounded, outlined card; optional bold first line as a heading."""
    text(sid, oid, x, y, w, h, t, size=size, fill=fill, line=INK, kind="ROUND_RECTANGLE", valign=valign, align="CENTER" if center else "START")
    if bullets:
        start = len(head) + 1 if head else 0
        R.append({"createParagraphBullets": {"objectId": oid, "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                  "textRange": rng(start, len(t)) if head else {"type": "ALL"}}})
    if head:
        style(oid, size + 3, True, rng=rng(0, len(head)))

def content(sid, title):
    ORDER.append(sid)
    R.append({"duplicateObject": {"objectId": "base0", "objectIds": {"base0": sid, "base0_logo": sid + "_logo"}}})
    size = 22 if len(title) <= 40 else 18
    w = min(530, 40 + len(title) * size * 0.55)
    text(sid, sid + "_t", 24, 20, w, 42, title, size=size, fill="#ffffff", line=INK, kind="ROUND_RECTANGLE", valign="MIDDLE")

def source(sid, t, y=376):
    text(sid, sid + "_src", 24, y, 672, 24, t, size=8.5, color=GREY)

def table(sid, oid, rows, x=24, y=76, w=672, size=10.5, widths=None, hl=(), mono=(), boldcol0=False):
    nr, nc = len(rows), len(rows[0])
    R.append({"createTable": {"objectId": oid, "rows": nr, "columns": nc, "elementProperties": {"pageObjectId": sid,
        "size": {"width": PT(w), "height": PT(20 * nr)}, "transform": {"scaleX": 1, "scaleY": 1, "translateX": x, "translateY": y, "unit": "PT"}}}})
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            if not t: continue
            loc = {"rowIndex": r, "columnIndex": c}
            R.append({"insertText": {"objectId": oid, "cellLocation": loc, "text": t}})
            style(oid, size, bold=(r == 0 or (boldcol0 and c == 0)), font=MONO if (r > 0 and c in mono) else GS, cell=loc)
    def fill(r, color):
        R.append({"updateTableCellProperties": {"objectId": oid, "tableRange": {"location": {"rowIndex": r, "columnIndex": 0}, "rowSpan": 1, "columnSpan": nc},
            "tableCellProperties": {"tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": rgb(color)}}}}, "fields": "tableCellBackgroundFill"}})
    fill(0, YEL)
    for r in hl: fill(r, BLU)
    R.append({"updateTableBorderProperties": {"objectId": oid, "borderPosition": "ALL",
        "tableBorderProperties": {"tableBorderFill": {"solidFill": {"color": {"rgbColor": rgb(INK)}}}, "weight": PT(0.75), "dashStyle": "SOLID"},
        "fields": "tableBorderFill,weight,dashStyle"}})
    if widths:
        for c, wd in enumerate(widths):
            R.append({"updateTableColumnProperties": {"objectId": oid, "columnIndices": [c], "tableColumnProperties": {"columnWidth": PT(wd)}, "fields": "columnWidth"}})

def section(sid, tmpl, txt_id, tag_id, label, part):
    ORDER.append(sid)
    R.append({"duplicateObject": {"objectId": H + tmpl, "objectIds": {H + tmpl: sid, H + txt_id: sid + "_txt", H + tag_id: sid + "_tag"}}})
    for oid, t, size, bold in ((sid + "_txt", label, 40, True), (sid + "_tag", part, 9.5, False)):
        R.append({"deleteText": {"objectId": oid, "textRange": {"type": "ALL"}}})
        R.append({"insertText": {"objectId": oid, "text": t}})
        style(oid, size, bold)

def arrow(sid, oid, x1, y1, x2, y2):
    R.append({"createLine": {"objectId": oid, "category": "STRAIGHT", "elementProperties": {"pageObjectId": sid,
        "size": {"width": PT(max(x2 - x1, 0.01)), "height": PT(max(y2 - y1, 0.01))},
        "transform": {"scaleX": 1, "scaleY": 1, "translateX": x1, "translateY": y1, "unit": "PT"}}}})
    R.append({"updateLineProperties": {"objectId": oid, "lineProperties": {"endArrow": "FILL_ARROW", "weight": PT(1.5),
        "lineFill": {"solidFill": {"color": {"rgbColor": rgb(INK)}}}}, "fields": "endArrow,weight,lineFill"}})

def bars(sid, pre, items, x0, y0, wmax, bh, vmax, unit="", colors=None, labelw=110):
    for i, (lab, v) in enumerate(items):
        y = y0 + i * (bh + 8)
        text(sid, f"{pre}l{i}", x0, y, labelw, bh, lab, size=11, align="END", valign="MIDDLE")
        w = max(4, wmax * v / vmax)
        shape(sid, f"{pre}b{i}", "RECTANGLE", x0 + labelw + 8, y, w, bh, fill=(colors or [YEL])[i % len(colors or [YEL])])
        text(sid, f"{pre}v{i}", x0 + labelw + 14 + w, y, 70, bh, f"{v:,}{unit}", size=11, bold=True, valign="MIDDLE")

# ---------- base: logo-only content slide ----------
R.append({"duplicateObject": {"objectId": H + "178", "objectIds": {H + "178": "base0", H + "181": "base0_logo",
    H + "179": "base0_x1", H + "180": "base0_x2", H + "182": "base0_x3", H + "183": "base0_x4"}}})
for k in range(1, 5): R.append({"deleteObject": {"objectId": f"base0_x{k}"}})

# ---------- 1 title ----------
def title_slide(sid, lines):
    ORDER.append(sid)
    R.append({"duplicateObject": {"objectId": H + "0", "objectIds": {H + "0": sid, H + "2": sid + "_card"}}})
    oid = sid + "_card"; t = "\n".join(l for l, _, _, _ in lines)
    R.append({"deleteText": {"objectId": oid, "textRange": {"type": "ALL"}}})
    R.append({"insertText": {"objectId": oid, "text": t}})
    i = 0
    for l, size, bold, font in lines:
        style(oid, size, bold, font=font, rng=rng(i, i + len(l) + (1 if i + len(l) < len(t) else 0)))
        i += len(l) + 1
title_slide("t_title", [("Intro to Cloud TPU", 30, False, GS),
                        ("Gemma 4 on a v5e chip, managed over MCP", 13, False, GS),
                        ("William McLean (xbill) · GDE Cloud + AI", 11, False, MONO)])

# ---------- 2 about me ----------
s = "slide02"; content(s, "About Me")
bio_head = "William McLean (xbill)"
card(s, s + "_bio", 24, 78, 340, 250, bio_head + "\nAWS Community Builder 2026, AI\nGDE (Google Developer Expert), Cloud and AI/ML\nCTO and co-founder, Glitnir Ticketing\n2026 goal: multi-cloud agents, tools and models",
     size=14, head=bio_head, bullets=True)
links_head = "Where to Find Me"
card(s, s + "_lnk", 380, 78, 316, 250, links_head + "\nLinkedIn  linkedin.com/in/xbill\nDev.to  dev.to/xbill\nMedium  medium.com/@xbill999\nGitHub  github.com/xbill9\nHugging Face  huggingface.co/xbill9",
     fill=BLU, size=13, head=links_head)

# ---------- 3 agenda ----------
s = "slide03"; content(s, "Agenda")
items = [("01", "Ten years of TPUs", YEL), ("02", "The chips, and what they compute natively", BLU),
         ("03", "Getting a chip: control planes, quota and price", PNK), ("04", "Live demo: Gemma 4 on a v5e-1 over MCP", GRN),
         ("05", "So, which one?", "#ffffff")]
for i, (n, t, c) in enumerate(items):
    y = 82 + i * 56
    text(s, f"{s}_n{i}", 56, y - 2, 52, 52, n, size=14, bold=True, fill=c, line=INK, kind="ELLIPSE", align="CENTER", valign="MIDDLE")
    text(s, f"{s}_i{i}", 122, y + 3, 480, 40, t, size=17, fill="#ffffff", line=INK, kind="ROUND_RECTANGLE", valign="MIDDLE")

# ---------- 4 section 01 ----------
section("sec01", "35", "39", "38", "Ten Years\nof TPUs", "Part 01")

# ---------- 5 history ----------
s = "slide05"; content(s, "The First TPU Was Built for Inference")
table(s, s + "_tb", [["Year", "Chip", "What changed"],
    ["2015", "TPU v1", "Inference only, 8-bit, 92 TOPS; announced 2016"],
    ["2017", "TPU v2", "bf16 training, 180 TFLOPs per device, first pods"],
    ["2019", "TPU v3 pods", "Liquid cooled, 1,024 chips, public beta"],
    ["2021", "TPU v4", "Optical circuit switches, 4,096 chips"],
    ["2023", "v5e and v5p", "Cost-optimized v5e; v5p with 95 GiB HBM"],
    ["2024", "v6e Trillium", "4.7x v5e peak compute"],
    ["2025", "TPU7x Ironwood", "fp8, 192 GiB HBM; “first TPU for the age of inference”; GA March 2026"],
    ["2026", "TPU 8t and 8i", "Announced April 2026; 8i is the inference part"]],
    widths=[70, 130, 472], hl=(1, 7, 8), boldcol0=True)
source(s, "Blue rows: the inference-first chips. Sources: Google Cloud and Google blogs 2016 to 2026; Jouppi et al., arXiv 1704.04760.")

# ---------- 6 stack ----------
s = "slide06"; content(s, "vLLM Reaches the TPU Through tpu-inference")
layers = [("Client", "OpenAI API", YEL), ("vLLM", "batching, KV paging", PNK), ("tpu-inference", "JAX model code", BLU),
          ("XLA", "one program per shape", GRN), ("TPU VM", "1 to 8 chips per host", ORG)]
for i, (a, b, c) in enumerate(layers):
    t = f"{a}\n{b}"
    text(s, f"{s}_L{i}", 24, 78 + i * 56, 250, 48, t, size=11, fill=c, line=INK, kind="ROUND_RECTANGLE", align="CENTER", valign="MIDDLE")
    style(f"{s}_L{i}", 14, True, rng=rng(0, len(a)))
card(s, s + "_k", 296, 78, 400, 272,
     "tpu-inference v0.31.0 shipped 2026-10-08; v5e, v6e and v7x recommended, v5p experimental\n"
     "Gemma 4 E2B, E4B, 26B-A4B and 31B pass its release matrix\n"
     "XLA compile is 81 to 91% of engine start for E2B on v6e and v5e\n"
     "Pin an image digest: the nightly tag broke two 4-bit builds on 2026-10-02", size=13, bullets=True, valign="MIDDLE")
source(s, "Sources: github.com/vllm-project/tpu-inference; gemma4-dev tpu-jax-v5e1-2b and tpu-vllm-v5e1-2b reports.")

# ---------- 7 section 02 ----------
section("sec02", "83", "87", "86", "The Chips", "Part 02")

# ---------- 8a TPU vs GPU vs NPU ----------
def vs_slide(s):
    table(s, s + "_tb", [["", "TPU", "GPU", "NPU"],
        ["Built for", "Large matrix operations in ML", "Graphics, plus parallel compute", "Deep learning math on the device"],
        ["Inside", "MXU: a systolic array of 128x128 multiply-accumulators (256x256 on v6e, TPU7x)",
         "Per Ada SM: 128 CUDA cores, 1 RT core, 4 Tensor Cores, 4 texture units", "Low-bit integer math such as int8"],
        ["Where", "Google Cloud: v5e, v6e, TPU7x", "Cloud and desktop; the NVIDIA L4 is Ada", "AI PCs: Snapdragon X Elite, Ryzen AI 300, Core Ultra 200V"]],
        widths=[100, 190, 190, 192], size=11, boldcol0=True)
    card(s, s + "_k", 24, 262, 672, 92,
         "TPU: a custom ASIC built around the systolic matrix unit, with HBM on the chip\n"
         "GPU: tensor cores share each SM with the shader, ray-tracing and texture hardware that draws graphics\n"
         "NPU: low power and on the device; Copilot+ PCs need 40+ TOPS for translation and image generation",
         fill=BLU, size=12, bullets=True, valign="MIDDLE")
    source(s, "Sources: Google Cloud TPU docs (Introduction to Cloud TPU; TPU system architecture); NVIDIA Ada GPU Architecture whitepaper; Microsoft Learn, Copilot+ PCs developer guide. Read 2026-10-29.")
s = "slide08a"; content(s, "TPU vs GPU vs NPU: Who Does the Matrix Math"); vs_slide(s)

# ---------- 8 chips ----------
s = "slide08"; content(s, "TPU7x: About 13x v5e's Memory at 10x the Price")
table(s, s + "_tb", [["Chip", "HBM", "Peak bf16", "Native low precision", "Max pod", "On-demand"],
    ["v5e", "16 GB", "197 TFLOPs", "int8, 393 TOPs", "256", "$1.20/hr"],
    ["v6e Trillium", "32 GB", "918 TFLOPs", "int8, 1,836 TOPs", "256", "$2.70/hr"],
    ["v5p", "95 GiB", "459 TFLOPs", "int8; fp8 at bf16 rate", "8,960", "$4.20/hr"],
    ["TPU7x Ironwood", "192 GiB", "2,307 TFLOPs", "fp8, 4,614 TFLOPs", "9,216", "$12.00/hr"]],
    widths=[130, 80, 110, 172, 80, 100], size=12, hl=(1,), boldcol0=True)
card(s, s + "_k", 24, 234, 672, 80,
     "Single-host shapes used here: v5litepod-1 and -4, v6e-1, -4 and -8\nv6e on-demand varies by region: $2.70 us-east, $2.97 europe-west4",
     fill=YEL, size=13, bullets=True, valign="MIDDLE")
source(s, "Sources: docs.cloud.google.com/tpu/docs and cloud.google.com/tpu/pricing, read 2026-10-12; regional rates from the Billing Catalog. Blue row: today's demo chip.")

# ---------- 9 bandwidth ----------
s = "slide09"; content(s, "One-Stream Decode Follows Memory Bandwidth")
table(s, s + "_tb", [["Chip", "Memory", "Bandwidth", "E2B bf16 tok/s, 1 stream", "tok/s per TB/s"],
    ["TPU v5e", "16 GB", "859 GB/s¹", "120", "140"],
    ["TPU v6e", "32 GB", "1,638 GB/s", "199", "121"],
    ["TPU v5p", "95 GiB", "2,765 GB/s", "not measured", "—"],
    ["TPU7x", "192 GiB", "7,380 GB/s", "not measured", "—"],
    ["NVIDIA L4", "24 GB", "300 GB/s", "46", "153"]],
    widths=[130, 100, 120, 182, 140], size=12, hl=(1,), boldcol0=True)
card(s, s + "_k", 24, 256, 672, 70,
     "Each token reads every weight once, so bandwidth sets the ceiling for one request\nThe three measured chips land at 121 to 153 tok/s per TB/s of bandwidth",
     fill=GRN, size=13, bullets=True, valign="MIDDLE")
source(s, "¹ Google quotes v5e as 800 GiBps. Different vLLM builds per platform. Sources: gemma4-dev HARDWARE.md, MI300X dollar-hour article; Google TPU docs.")

# ---------- 10 precision ----------
s = "slide10"; content(s, "Only int8, and fp8 on TPU7x, Buy Compute")
table(s, s + "_tb", [["Format", "v5e", "v6e", "v5p", "TPU7x"],
    ["bf16", "native", "native", "native", "native"],
    ["int8", "2x bf16", "2x bf16", "native", "native"],
    ["fp8", "storage", "storage", "bf16 rate", "2x bf16"],
    ["int4", "storage", "storage", "storage", "storage"]],
    widths=[120, 138, 138, 138, 138], size=12, boldcol0=True)
card(s, s + "_k", 24, 230, 672, 100,
     "2x bf16: a native matmul path at double the bf16 peak\nstorage: saves HBM and bandwidth, then widens to bf16 before every matmul\nfp8 KV cache on v6e logs fp8 and allocates bf16",
     fill=PNK, size=13, bullets=True, valign="MIDDLE")
source(s, "Sources: gemma4-dev HARDWARE.md, QUANTIZATION.md; Google TPU7x docs (not measured here).")

# ---------- 11 names ----------
s = "slide11"; content(s, "“v5e-1” Works in Prose, Never in a gcloud Flag")
table(s, s + "_tb", [["What you mean", "What gcloud wants", "Where it goes"],
    ["v5e, one chip", "v5litepod-1", "--accelerator-type (Cloud TPU API)"],
    ["v5e runtime", "v2-alpha-tpuv5-lite", "--runtime-version (Cloud TPU API)"],
    ["v5e machine", "ct5lp-hightpu-1t", "listed in the Compute Engine catalog; create is refused"],
    ["v6e, one chip", "v6e-1  /  ct6e-standard-1t", "Cloud TPU API  /  Compute Engine"],
    ["v6e runtime", "v2-alpha-tpuv6e", "--runtime-version (Cloud TPU API)"]],
    widths=[140, 230, 302], size=12, mono=(1,), boldcol0=True)

# ---------- 12 section 03 ----------
section("sec03", "132", "136", "135", "Getting\na Chip", "Part 03")

# ---------- 13 two control planes ----------
s = "slide13"; content(s, "Two Control Planes")
h1, h2 = "Cloud TPU API", "Compute Engine"
card(s, s + "_l", 24, 78, 328, 190, h1 + "\nQueued resources produce a node named <id>-node\ngcloud alpha compute tpus queued-resources create\nBug fixes and security updates only\nThe only path for v5e outside GKE",
     fill=YEL, size=12, head=h1, bullets=True)
card(s, s + "_r", 368, 78, 328, 190, h2 + "\nThe instance is the node: one object to manage\ngcloud compute instances create --machine-type=ct6e-standard-1t\nSupports TPU7x, v6e and v5p\nTPU7x (Ironwood) onward: Compute Engine or GKE only",
     fill=BLU, size=12, head=h2, bullets=True)
text(s, s + "_q", 24, 284, 672, 64,
     "“The Cloud TPU API is no longer under active development … New hardware generations, starting with TPU7x (Ironwood), are supported only through Compute Engine or Google Kubernetes Engine (GKE).”",
     size=12, fill="#ffffff", line=INK, kind="ROUND_RECTANGLE", valign="MIDDLE", align="START")
source(s, "Quote: Cloud TPU resources in Compute Engine, Google Cloud docs.")

# ---------- 14 flag mapping ----------
s = "slide14"; content(s, "Moving a v6e-1 Across: the Flag Mapping")
table(s, s + "_tb", [["Cloud TPU API", "Compute Engine"],
    ["--accelerator-type=v6e-1", "--machine-type=ct6e-standard-1t"],
    ["--runtime-version=v2-alpha-tpuv6e", "--image-family=ubuntu-accel-… + --image-project"],
    ["--valid-until-duration", "--request-valid-for-duration"],
    ["--provisioning-model=flex-start", "--provisioning-model=FLEX_START"],
    ["queued resource produces <id>-node", "the instance is the node"],
    ["gcloud compute tpus tpu-vm list", "gcloud compute instances list"],
    ["gcloud compute tpus tpu-vm ssh", "gcloud compute ssh"]],
    widths=[336, 336], size=10.5, mono=(0, 1))
card(s, s + "_k", 24, 312, 672, 46,
     "Same chip, same checkpoint, same serving flags: the KV cache came out at 1,151,744 tokens on both control planes.",
     fill=GRN, size=12, valign="MIDDLE", center=True)
source(s, "Source: The unofficial TPU migration guide, dev.to/gde, 2026-08-11.", y=372)

# ---------- 15 v5e no CE path ----------
s = "slide15"; content(s, "v5e Has No Compute Engine Create Path")
text(s, s + "_c", 24, 78, 672, 92,
     "$ gcloud compute machine-types list --filter=\"name~ct5lp\"\nct5lp-hightpu-1t  ct5lp-hightpu-4t  ct5lp-hightpu-8t   (26 zones)\n\n"
     "$ gcloud compute instances create … --machine-type=ct5lp-hightpu-1t\nERROR: This user agent is not allowed to use the machine type [ct5lp-hightpu-1t].",
     size=10.5, font=MONO, fill="#ffffff", line=INK, kind="ROUND_RECTANGLE", valign="MIDDLE", align="START")
card(s, s + "_b", 24, 186, 672, 168,
     "The catalog lists the machine type; GKE uses those names for its v5e node pools\n"
     "Google's TPU machines page names TPU7x, v6e and v5p for Compute Engine, and no v5e\n"
     "The v5e page: supported using Google Kubernetes Engine and the Cloud TPU API\n"
     "To test any chip, try a create in a zone where your quota is zero: a refusal costs nothing\n"
     "Today's demo runs on a Cloud TPU API queued resource", fill=YEL, size=13, bullets=True, valign="MIDDLE")

# ---------- 16 provisioning models ----------
s = "slide16"; content(s, "Four Ways to Ask for a Chip")
table(s, s + "_tb", [["Model", "Capacity", "Max run", "Ends", "Quota spent", "v6e, europe-west4"],
    ["STANDARD", "now, if available", "unlimited", "when you delete it", "standard", "$2.97 / chip-hr"],
    ["SPOT", "now, if available", "unlimited", "preempted at any time", "preemptible → standard", "$1.78 / chip-hr"],
    ["FLEX_START", "queues up to 2h", "10 min – 7 days", "at --max-run-duration", "preemptible → standard", "$1.35 / chip-hr"],
    ["RESERVATION_BOUND", "reserved ahead, if approved", "up to 90 days", "when the reservation ends", "with the reservation", "no list rate"]],
    widths=[134, 110, 88, 120, 120, 100], size=10, mono=(0,), hl=(3,))
card(s, s + "_k", 24, 268, 672, 92,
     "Flex-start is the cheapest of the three priced models on v6e, under half of on-demand\nOn v5e in us-west4: flex-start $0.60, spot $0.607 per chip-hr\nThe demo uses flex-start: it queues for capacity and deletes itself after 4 hours",
     fill=GRN, size=12, bullets=True, valign="MIDDLE")
source(s, "Behaviour: Compute Engine provisioning models docs. Prices: Cloud Billing Catalog, read 2026-08-11.")

# ---------- 17 pricing ----------
s = "slide17"; content(s, "Flex-Start Halves the On-Demand Rate")
table(s, s + "_tb", [["$ per chip-hour", "On-demand", "Flex-start", "Calendar", "1-yr CUD", "3-yr CUD", "Spot read"],
    ["v5e", "$1.20", "$0.60", "$0.84", "$0.84", "$0.54", "$0.58"],
    ["v6e Trillium", "$2.70", "$1.35", "$1.89", "$1.89", "$1.22", "$1.40"],
    ["v5p", "$4.20", "$2.10", "$2.94", "$2.94", "$1.89", "—"],
    ["TPU7x Ironwood", "$12.00", "$6.00", "$8.40", "$8.40", "$5.40", "—"]],
    widths=[132, 90, 90, 90, 90, 90, 90], size=12, hl=(1,), boldcol0=True)
card(s, s + "_k", 24, 234, 672, 100,
     "Flex-start can undercut spot: v6e in us-east5 is $1.35 flex-start against $1.40 spot\nFlex-start runs 10 min to 7 days and bills until deleted\nTPU7x spot, flex-start and calendar are behind an allowlist",
     fill=YEL, size=13, bullets=True, valign="MIDDLE")
source(s, "List rates: cloud.google.com/tpu/pricing, read 2026-10-12. Spot: Billing Catalog reads, Aug 2026.")

# ---------- 18 quota funnel ----------
s = "slide18"; content(s, "Quota Does Not Mean Availability")
steps = [("1. Hardware", "in the zone", YEL), ("2. Provisioning", "model offered", BLU), ("3. Capacity", "right now", PNK)]
for i, (a, b, c) in enumerate(steps):
    x = 24 + i * 232
    text(s, f"{s}_f{i}", x, 80, 200, 70, f"{a}\n{b}", size=13, fill=c, line=INK, kind="ROUND_RECTANGLE", align="CENTER", valign="MIDDLE")
    style(f"{s}_f{i}", 17, True, rng=rng(0, len(a)))
    if i < 2: arrow(s, f"{s}_a{i}", x + 202, 115, x + 230, 115.01)
card(s, s + "_k", 24, 170, 672, 186,
     "37 zones report v6e quota; 18 offer v6e-1 hardware, and Google's docs list 8\n"
     "3 of 5 zones holding 1,536 chips of quota returned stockout\n"
     "us-central1-b has hardware, quota and a flex-start price, and rejects flex-start\n"
     "Probe with spot: it fails fast with “reason: stockout”; flex-start with no quota queues as PENDING, like a stockout\n"
     "v5e-1 flex-start: accepted in us-west4-a, rejected in europe-west4-a and -b",
     size=13, bullets=True, valign="MIDDLE")
source(s, "Source: gemma4-dev tpu_zones_status.md (2026-08-10), TPU.md, request-quota.sh; tpu-vllm-v5e1-2b CLAUDE.md (2026-08-04).")

# ---------- 19 fails quietly ----------
s = "slide19"; content(s, "The Control Plane Fails Quietly")
cells = [("Cloud TPU API is frozen", "Bug fixes only. TPU7x runs on Compute Engine or GKE.", YEL),
         ("v5e has no Compute Engine path", "Listed in 26 zones; create refused.", BLU),
         ("Two separate quota pools", "us-east5: 512 v6e chips via the TPU API, 0 via Compute Engine.", PNK),
         ("RUNNING means booted", "A GKE node is Ready about 10 min before the model serves.", GRN),
         ("Boot defaults break vLLM", "10 GB disk, no Docker on PATH; a missing scope spins 30 min.", YEL),
         ("Nothing stops billing", "GKE pools and old queued resources bill until deleted.", BLU)]
for i, (a, b, c) in enumerate(cells):
    x, y = 24 + (i % 3) * 228, 80 + (i // 3) * 140
    card(s, f"{s}_g{i}", x, y, 216, 124, f"{a}\n{b}", fill=c, size=12, head=a, valign="MIDDLE", center=True)
source(s, "Sources: Google TPU docs; gemma4-dev TPU.md, request-quota.sh, gke-vllm-v6e1-2b.")

# ---------- 20 section 04 ----------
section("sec04", "35", "39", "38", "Live\nDemo", "Part 04")

# ---------- 21 architecture ----------
s = "slide21"; content(s, "What the Agent Drives")
boxes = [("MCP client", "Claude Code or\nAntigravity CLI", YEL), ("server.py", "tpu-vllm-v5e1-2b\n31 tools, stdio", PNK),
         ("Queued resource", "v5litepod-1\nus-west4-a, flex-start", BLU), ("TPU VM node", "vllm/vllm-tpu\nport 8000", GRN),
         ("Gemma 4 E2B-it", "bf16, one chip", ORG)]
for i, (a, b, c) in enumerate(boxes):
    x = 24 + i * 136
    text(s, f"{s}_b{i}", x, 84, 118, 96, f"{a}\n{b}", size=10.5, fill=c, line=INK, kind="ROUND_RECTANGLE", align="CENTER", valign="MIDDLE")
    style(f"{s}_b{i}", 12.5, True, rng=rng(0, len(a)))
    if i < 4: arrow(s, f"{s}_a{i}", x + 119, 132, x + 135, 132.01)
card(s, s + "_k", 24, 200, 672, 150,
     "The client talks to server.py over stdio; its tools wrap gcloud to provision, find the node's IP and start the container\n"
     "Queries go to vLLM's OpenAI-compatible /v1/chat/completions\n"
     "analyze_cloud_logging sends the TPU's own error logs to the model it hosts\n"
     "Project, zone, model and provisioning model come from one file, tpu.env",
     size=13, bullets=True, valign="MIDDLE")

# ---------- 22 demo script ----------
s = "slide22"; content(s, "Demo Script")
table(s, s + "_tb", [["#", "Ask the agent", "Tool it calls"],
    ["1", "What's running in the zone?", "get_system_status"],
    ["2", "Which zones have v5e quota?", "get_zones_with_available_quota"],
    ["3", "Create the v5e-1, flex-start", "create_tpu_queued_resource"],
    ["4", "Is it up yet? Show the container log", "describe_queued_resource, get_vllm_docker_logs"],
    ["5", "Is the model healthy?", "verify_model_health"],
    ["6", "Explain a TPU in two sentences, with speed", "query_queued_gemma4_with_stats"],
    ["7", "Any TPU errors in the last hour?", "analyze_cloud_logging"],
    ["8", "Tear it down", "destroy_queued_resource"]],
    widths=[36, 290, 346], size=11.5, mono=(2,), y=74)
text(s, s + "_n", 24, 346, 672, 36,
     "In a shared zone, provision with create_tpu_queued_resource: manage_queued_resource deletes every other queued resource in the zone.",
     size=11, fill=PNK, line=INK, kind="ROUND_RECTANGLE", valign="MIDDLE")

# ---------- 23 cold start ----------
s = "slide23"; content(s, "XLA Compile Is Most of a Cold Start: Cache It")
text(s, s + "_h", 24, 76, 672, 20, "Seconds from container start to healthy, Gemma 4 E2B", size=12, bold=True)
bars(s, s + "_", [("v6e-1", 510), ("v5e-1", 986)], 24, 102, 440, 34, 986, unit=" s", colors=[BLU, YEL], labelw=70)
card(s, s + "_k", 24, 200, 672, 150,
     "XLA compile is 81 to 91% of engine start\n"
     "A mounted compile cache cut a v5e restart 42%, 857 to 497 s\n"
     "31B weights load in 5.6 min from in-region GCS; 60 to 90 min from Hugging Face\n"
     "GKE from nothing to first token: about 28 min\n"
     "Each v6e cold start costs $0.19 to $0.24 at flex-start rates",
     fill=GRN, size=13, bullets=True, valign="MIDDLE")
source(s, "Source: gemma4-dev tpu-jax-v5e1-2b and tpu-vllm-v5e1-2b reports.")

# ---------- 24 int8 bars ----------
s = "slide24"; content(s, "On v5e, int8 Serves E2B 1.7x Faster Than bf16")
text(s, s + "_h", 24, 74, 420, 20, "Output tok/s, one stream, v5e-1 (E2B)", size=12, bold=True)
bars(s, s + "_", [("bf16", 144), ("4-bit QAT", 150), ("fp8", 182), ("fp8 + 4-bit emb.", 203), ("int8 W8A8", 220), ("int8 + 4-bit emb.", 243)],
     24, 100, 230, 30, 243, colors=[PNK, YEL, BLU, BLU, GRN, GRN], labelw=120)
card(s, s + "_k", 470, 100, 226, 228,
     "Accuracy holds: 0.683 bf16, 0.677 int8 + 4-bit embeddings\nfp8 trails int8 at the same file size: 518 against 624 tok/s on 12B\n12B int8 (11.31 GiB) is the largest model served on one 16 GB v5e chip",
     fill=YEL, size=12, bullets=True, valign="MIDDLE")
source(s, "One stream, v5e-1. Source: gemma4-dev QUANTIZATION.md, measured 2026-09-29/30.")

# ---------- 25 measured ----------
s = "slide25"; content(s, "Measured on One v5e Chip")
table(s, s + "_tb", [["Gemma 4 E2B-it, bf16, v5litepod-1", "Result"],
    ["GSM8K, all 1,319 problems, 2,048-token limit", "0.910"],
    ["GSM8K, 768-token limit", "0.894"],
    ["BFCL v3 simple, 400 single-tool calls", "0.928"],
    ["~1,000-token prompts, output tok/s at 1 / 16 requests", "139 / 1,521"],
    ["~3,600-token prompts, output tok/s at 1 / 16 requests", "130 / 906"],
    ["~3,600-token prompts, time to first token at 16 requests", "1.10 s"]],
    widths=[512, 160], size=13, hl=(4,))
card(s, s + "_k", 24, 300, 672, 50, "One chip, 16 requests, ~1,000-token prompts: 1,521 output tok/s", fill=GRN, size=15, valign="MIDDLE", center=True)
source(s, "Measured 2026-09-30. GSM8K and BFCL greedy. Each long prompt has its own prefix, so the prefix cache serves none of it. Source: tpu-vllm-v5e1-2b README.")

# ---------- 26 which one ----------
s = "slide26"; content(s, "So, Which One?")
table(s, s + "_tb", [["Workload", "Pick", "Why"],
    ["E2B or E4B, cost first", "v5e-1, int8, one engine per chip", "1.7x bf16 speed; lowest $/M at spot or flex-start"],
    ["12B to 31B on one chip", "v6e-1, 4-bit QAT W4A16", "26B at 1,283 tok/s; 31B fits in 21.67 GiB"],
    ["bf16 26B or 31B, long context", "v6e-4, TP=4", "about 2,000 tok/s"],
    ["fp8 math, 192 GiB per chip", "TPU7x", "allowlisted; not measured here"],
    ["Stock software, fast boot", "L4, or Cloud Run L4", "26B 4-bit loads unpatched; 334 s to healthy"]],
    widths=[190, 220, 262], size=11.5, hl=(1,), boldcol0=True)
card(s, s + "_k", 24, 262, 672, 60,
     "Then: probe capacity with spot, ask for the smallest slice, pin the image digest, mount the compile cache.",
     fill=GRN, size=13, valign="MIDDLE", center=True)

# ---------- 27 takeaways ----------
s = "slide27"; content(s, "Takeaways")
tk = [("One v5e chip serves Gemma 4 E2B at about 1,500 output tok/s across 16 requests, for $0.60 an hour on flex-start", YEL),
      ("int8 serves E2B 1.7x faster than bf16 on v5e, with accuracy within a point", BLU),
      ("New TPU generations ship on Compute Engine and GKE; v5e stays on the Cloud TPU API", PNK),
      ("Probe capacity with a spot create; read quota by metric name", GRN),
      ("Health-check the port: RUNNING and ACTIVE only say the machine is up", YEL),
      ("An MCP server turns the whole lifecycle into prompts, including teardown", BLU)]
for i, (t, c) in enumerate(tk):
    x, y = 24 + (i % 2) * 342, 80 + (i // 2) * 96
    card(s, f"{s}_g{i}", x, y, 330, 84, t, fill=c, size=13, valign="MIDDLE", center=True)

# ---------- 28 links ----------
s = "slide28"; content(s, "Links")
card(s, s + "_a", 24, 80, 400, 270, "Rigs and results\ngithub.com/xbill9/gemma4-dev\n\ntpu-inference\ngithub.com/vllm-project/tpu-inference\n\n"
     "The unofficial TPU migration guide\ndev.to/gde/the-unofficial-tpu-migration-guide-cloud-tpu-api-to-compute-engine-2co7\n\n"
     "TPU docs and pricing\ndocs.cloud.google.com/tpu/docs\ncloud.google.com/tpu/pricing", size=11, valign="MIDDLE")
card(s, s + "_b", 440, 80, 256, 270, "Where to Find Me\nLinkedIn  linkedin.com/in/xbill\nDev.to  dev.to/xbill\nMedium  medium.com/@xbill999\nGitHub  github.com/xbill9\nHugging Face  huggingface.co/xbill9",
     fill=BLU, size=12, head="Where to Find Me", valign="MIDDLE")

# ---------- 29 questions ----------
title_slide("t_qna", [("Questions?", 34, False, GS), ("github.com/xbill9/gemma4-dev", 12, False, MONO)])

# ---------- order and cleanup ----------
for i, sid in enumerate(ORDER): R.append({"updateSlidesPosition": {"slideObjectIds": [sid], "insertionIndex": i}})
for k in ["0", "26", "35", "43", "53", "64", "73", "83", "91", "97", "115", "132", "140", "150", "162", "178", "186", "210"]:
    R.append({"deleteObject": {"objectId": H + k}})
R.append({"deleteObject": {"objectId": "base0"}})
json.dump(R, open(sys.argv[1], "w"), separators=(",", ":"), ensure_ascii=False)
print(len(R), len(ORDER), len(json.dumps(R)))

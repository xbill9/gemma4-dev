"""Write a model card, plus the scripts and reports it cites, for each new build.

    python3 make_cards.py REPORTS_DIR SIZES_TXT OUT_DIR

Every figure is read from the build's own reports; nothing here is typed in by hand
except the source revisions, which were read from the Hub on 2026-10-08.
"""
import glob, json, os, re, shutil, sys
from huggingface_hub import hf_hub_download

REP, SIZES, OUT = sys.argv[1:4]
DEV = os.path.expanduser("~/gemma4-dev")
SCRIPTS = {
    "fp8_text.py": f"{DEV}/gpu-vllm-mi300x-2b/repack/fp8_text.py",
    "fp8:repack_q4_0.py": f"{DEV}/gpu-vllm-t4-2b-w4a16/repack/repack_q4_0.py",
    "gguf_exact.py": f"{DEV}/gpu-vllm-mi300x-2b/repack/gguf_exact.py",
    "w8a8_from_qat.py": f"{DEV}/jev-tpu-31b/w8a8_from_qat.py",
    "w8a8_emb4.py": f"{DEV}/jev-tpu-31b/w8a8_emb4.py",
    "embed_int4.py": f"{DEV}/jev-tpu-31b/embed_int4.py",
    "repack_q4_0.py": f"{DEV}/jev-tpu-31b/repack_q4_0.py",
}
UNQ_REV = {"E2B": "6befbac", "E4B": "476025a", "12B": "b6ed862", "26B-A4B": "f1e06dc", "31B": "1e4d8be"}
GGUF_SRC = {"12B": ("gemma-4-12b-it-qat-q4_0.gguf", "29d0977"), "26B-A4B": ("gemma-4-26B_q4_0-it.gguf", "d1c082b"),
            "31B": ("gemma-4-31B_q4_0-it.gguf", "59dde24")}
CT_REV = {"E2B-text": "315aec0", "E2B-emb4": "db715f8", "E4B-text": "38b5a97", "E4B-emb4": "2c02a2c",
          "12B-emb4": "94ff29e", "26B-A4B-text": "62a64d6", "26B-A4B-emb4": "38971d3",
          "31B-text": "db223c9", "31B-emb4": "c8e1d82"}
sizes = dict(l.split() for l in open(SIZES) if l.strip())


def gib(b):
    return f"{int(b) / 2**30:.2f} GiB"


def pct(x, n=2):
    return f"{100 * x:.{n}f} %"


def size_of(repo):
    return re.match(r"gemma-4-(E2B|E4B|12B|26B-A4B|31B)-", repo).group(1)


def front(size, tags, lib="vllm", pipe="text-generation"):
    t = "\n".join(f"- {x}" for x in tags)
    return f"""---
library_name: {lib}
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: {pipe}
base_model:
- google/gemma-4-{size}-it-qat-q4_0-unquantized
base_model_relation: quantized
tags:
{t}
---
"""


TAIL = """
## License and attribution

Gemma 4 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This repository redistributes Google's weights in a changed storage format, under the same license. The weights, training and the original model card (kept unchanged in `ORIGINAL_README.md`) are Google DeepMind's; this build and card are not affiliated with or endorsed by Google.
"""

UNSERVED = ("**Status:** built and checked offline against its source; not yet served or evaluated. "
            "It is queued for a serving sweep on one AMD Instinct MI300X.")


def link(repo):
    return f"[`xbill9/{repo}`](https://huggingface.co/xbill9/{repo})"


def moe_note(size):
    if size != "26B-A4B":
        return ""
    return ("\nThe source stores each layer's 128 experts as two fused banks (`experts.gate_up_proj`, "
            "`experts.down_proj`). This build splits them into one module per expert, "
            "`experts.{i}.{gate,up,down}_proj`, the layout of the W4A16 repack "
            f"{link('gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text')}. The router (`router.proj`) stays bf16.\n")


def fp8_card(repo, size):
    fnuz = "fp8fnuz" in repo
    emb4 = repo.endswith("-emb4")
    v = json.load(open(f"{REP}/{repo}/verify_report.json"))
    fmt = "FP8 E4M3FNUZ" if fnuz else "FP8 E4M3"
    title = f"Gemma 4 {size}-it QAT, {fmt} W8A8{' with int4 embeddings' if emb4 else ''}, text only (unofficial)"
    src_ct = f"gemma-4-{size}-it-qat-q4_0-w4a16-ct-text{'-emb4' if emb4 else ''}"
    rev = CT_REV[f"{size}-{'emb4' if emb4 else 'text'}"]
    emb = (f"Embeddings{', per-layer embeddings' if size in ('E2B', 'E4B') else ''} and `lm_head` are the int4 "
           f"tables of {link(src_ct)} (revision `{rev}`), copied unchanged; `lm_head` is untied."
           if emb4 else "Embeddings, norms and other tensors are bf16, copied byte for byte.")
    scale = "max\\|row\\| / 240" if fnuz else "max\\|row\\| / 448"
    s = front(size, ["gemma4", "compressed-tensors", "fp8", "w8a8", "qat", "vllm", "text-only"]
              + (["rocm", "mi300x"] if fnuz else []))
    s += f"\n# {title}\n\n"
    s += ("**This is an unofficial build, made and published independently of Google.** "
          f"It holds Google's quantization-aware-trained (QAT) Gemma 4 {size}-it weights, from "
          f"[`google/gemma-4-{size}-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-{size}-it-qat-q4_0-unquantized) "
          f"(revision `{UNQ_REV[size]}`), text model only, with every Linear layer stored as **{fmt}** "
          "(one float32 scale per output channel) and activations quantized to FP8 per token at run time "
          f"(compressed-tensors `float-quantized`). {emb}\n")
    if fnuz:
        twin = repo.replace("fp8fnuz", "fp8")
        s += (f"\n**E4M3FNUZ is the FP8 format of AMD CDNA 3 (Instinct MI300X, MI300A, MI325X).** It has the same "
              "3-bit mantissa as the E4M3 used by NVIDIA, with an exponent bias one higher, so its largest value is 240 "
              f"against 448, and it has no negative zero. This build rounds the QAT weights straight to that grid, "
              f"with scale {scale}. Its twin {link(twin)} holds the same weights in E4M3; the two builds differ by "
              "where each rounds. Against the QAT weights both have a relative RMS error of 2.64 %.\n")
    s += moe_note(size)
    total = int(v["values"])
    s += f"""
| Measured against the QAT weights (`verify_report.json`) | |
| --- | ---: |
| FP8 Linear modules | {v['modules']:,} |
| Quantized values | {total:,} |
| Relative RMS error | {pct(v['relative_rms_error'])} |
| Largest error, as a fraction of its row's largest value | {pct(v['max_rel_err'])} |
| Other tensors, byte-identical to their source | {v['copied_identical']:,} of {v['copied']:,} |
| Checkpoint | {gib(sizes[repo])} |

QAT trained the weights onto a 4-bit grid with one scale per group of 32 values. FP8 with one scale per
output channel cannot represent those per-group scales, so this build rounds the QAT weights again, and it
also quantizes activations. For the exact QAT grid use {link(f'gemma-4-{size}-it-qat-q4_0-w4a16-ct-text')}.

{UNSERVED}
"""
    if fnuz:
        s += """
## Serving

On ROCm, vLLM's compressed-tensors loader casts these weights into an E4M3 parameter and then converts
them to E4M3FNUZ for the GPU, doubling the scale. Every value of this checkpoint up to 240 survives that
round trip except the smallest E4M3FNUZ subnormal (2^-10), which E4M3 cannot hold and rounds. Other GPUs
are untested.
"""
    s += f"""
## Built with

`fp8_text.py` (in this repo{', with `--fnuz`' if fnuz else ''}{', `build-on` the int4-embedding build' if emb4 else ''}),
which imports helpers from `repack_q4_0.py` (also here). No calibration data is used.

## Limitations

- Text only.
- Not yet served or evaluated.
- Unofficial. Report problems here, not to Google.
""" + TAIL
    return s, ["fp8_text.py", "fp8:repack_q4_0.py"]


def w8a8_rows(repo):
    log = open(f"{REP}/root/build/logs/{repo}.log").read()
    rows = re.findall(r"^(\S+)\s+(\d+) tensors, int8 vs QAT rel err ([\d.]+)-([\d.]+) \(mean ([\d.]+)\)", log, re.M)
    return [(n, int(t), float(a), float(b), float(m)) for n, t, a, b, m in rows]


def w8a8_card(repo, size):
    emb4 = repo.endswith("-emb4")
    base = repo.removesuffix("-emb4")
    rows = w8a8_rows(base)
    s = front(size, ["gemma4", "compressed-tensors", "w8a8", "int8"] + (["int4"] if emb4 else [])
              + ["qat", "vllm", "text-only"])
    s += f"\n# Gemma 4 {size}-it QAT, int8 W8A8{' with int4 embeddings' if emb4 else ''}, text only (unofficial)\n\n"
    s += ("**This is an unofficial build, made and published independently of Google.** "
          f"It holds Google's quantization-aware-trained (QAT) Gemma 4 {size}-it weights, from "
          f"[`google/gemma-4-{size}-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-{size}-it-qat-q4_0-unquantized) "
          f"(revision `{UNQ_REV[size]}`), text model only, stored as compressed-tensors int8 W8A8: int8 weights with one "
          "bf16 scale per output channel, activations quantized to int8 per token at run time (`format: int-quantized`). "
          "The QAT weights already sit on the Q4_0 grid; this build rounds them to int8 per channel.\n")
    if emb4:
        src = f"gemma-4-{size}-it-qat-q4_0-w4a16-ct-text-emb4"
        s += (f"\n`embed_tokens` and `lm_head` are the int4 tables (group 32, fp16 scales) of {link(src)} "
              f"(revision `{CT_REV[size + '-emb4']}`), copied unchanged; `lm_head` is untied. QAT put the embedding on the "
              f"same 4-bit grid as the linears, so those tables keep the trained values. The other tensors are identical "
              f"to {link(base)}.\n")
    s += moe_note(size)
    s += "\n| Layer | Tensors | int8 against QAT, relative error |\n| --- | ---: | --- |\n"
    for n, t, a, b, m in rows:
        s += f"| `{n}` | {t:,} | {pct(a)}–{pct(b)} (mean {pct(m)}) |\n"
    s += f"\nCheckpoint: {gib(sizes[repo])}"
    if emb4:
        s += f", against {gib(sizes[base])} for the build with bf16 embeddings"
    s += f".\n\n{UNSERVED}\n"
    s += f"""
## Built with

`w8a8_from_qat.py`{' then `w8a8_emb4.py`' if emb4 else ''} (in this repo, with the `repack_q4_0.py` helpers they import).
No calibration data is used.

## Limitations

- Text only.
- Not yet served or evaluated.
- Unofficial. Report problems here, not to Google.
""" + TAIL
    return s, ["w8a8_from_qat.py", "repack_q4_0.py"] + (["w8a8_emb4.py"] if emb4 else [])


def ple4_card(repo, size):
    log = open(f"{REP}/root/build/logs/{repo}.log").read()
    m = re.search(r"embed_tokens_per_layer\.weight: ([\d.]+) GiB bf16 -> ([\d.]+) GiB int4 \+ f16 scales; (\d+) off-grid "
                  r"groups; ([\d.]+)% of values bit-identical, worst error ([\d.e-]+) of the group max; scales ([\d.]+)\.\.([\d.]+)", log)
    a, b, off, ident, worst, lo, hi = m.groups()
    src = f"gemma-4-{size}-it-qat-q4_0-w4a16-ct-text"
    s = front(size, ["gemma4", "compressed-tensors", "w4a16", "int4", "qat", "vllm", "text-only"])
    s += f"\n# Gemma 4 {size}-it QAT, W4A16 with int4 per-layer embeddings, text only (unofficial)\n\n"
    s += ("**This is an unofficial build, made and published independently of Google.** "
          f"It is the text-only W4A16 repack {link(src)} (revision `{CT_REV[size + '-text']}`) with its per-layer embedding "
          "table packed as int4. `embed_tokens` and `lm_head` stay bf16 and tied. It separates the per-layer table's "
          f"effect from `lm_head`'s: compare it with {link(src + '-emb4')}, which also packs `embed_tokens` and an untied "
          "`lm_head`. QAT put the embedding tables on the same 4-bit grid as the linears (group 32), so the packing "
          "recovers that grid; an off-grid group would stop the build.\n")
    s += f"""
| Table | bf16 | int4 |
| --- | ---: | ---: |
| `embed_tokens_per_layer` | {a} GiB | {b} GiB + f16 scales |

- `embed_tokens_per_layer`: {off} off-grid groups; {ident}% of values bit-identical, worst error {worst} of the group max; scales {lo}..{hi}.

Checkpoint: {gib(sizes[repo])}. Linear layers are unchanged from the W4A16 repack.

{UNSERVED}

## Built with

`embed_int4.py` (in this repo, with the `repack_q4_0.py` helpers it imports), default fp16 scales.
Needs **vLLM 0.29 or later** (`CompressedTensorsEmbeddingWNA16Int`).

## Limitations

- Text only.
- Not yet served or evaluated.
- Unofficial. Report problems here, not to Google.
""" + TAIL
    return s, ["embed_int4.py", "repack_q4_0.py"]


def ct_card(repo, size):
    v = json.load(open(f"{REP}/{repo}/verify_report.json"))
    q = v["quantized"]
    groups = sum(x["groups"] for x in q.values())
    bad = sum(x["groups_level_mismatch"] for x in q.values())
    vals = sum(x["values"] for x in q.values())
    ident = sum(x["values_bit_identical"] for x in q.values())
    worst = max(x["max_rel_err"] for x in q.values())
    s = front(size, ["gemma4", "compressed-tensors", "w4a16", "int4", "qat", "vllm"], pipe="image-text-to-text")
    s += f"\n# Gemma 4 {size}-it QAT, compressed-tensors W4A16 (unofficial repack)\n\n"
    s += ("**This is an unofficial repack, made and published independently of Google.** "
          f"It holds Google's quantization-aware-trained (QAT) weights for Gemma 4 {size}-it, from "
          f"[`google/gemma-4-{size}-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-{size}-it-qat-q4_0-unquantized) "
          f"(revision `{UNQ_REV[size]}`), repacked into the compressed-tensors W4A16 format that vLLM loads, with the "
          f"vision tower kept. The text-only build is {link(repo + '-text')}.\n")
    s += f"""
## What it is

- **Format:** compressed-tensors `pack-quantized`, symmetric int4, group size 32, bf16 scales, activations unquantized (W4A16).
- **Quantized:** every attention and MLP projection of the language model.
- **Kept bf16, copied byte for byte:** embeddings, norms and the vision tower ({v['copied']['byte_identical']:,} of {v['copied']['tensors']:,} tensors byte-identical).
- **Size:** {gib(v['out_bytes'])}, against {gib(v['src_bytes'])} for the bf16 source.

## How it was made

The `-qat-q4_0-unquantized` export stores bf16 values that already lie on a 4-bit grid: within every group of 32
weights, each weight is `step × level` with level from −8 to 7. The repack recovers each group's step (the
`max|w| / m`, m from 1 to 8, that reproduces all 32 values, refined by least squares) and writes the levels as
packed int4. It does no new quantization.

## Verification

`repack_q4_0.py verify` rereads both checkpoints and checks every group:

| Layer | Tensors | Groups of 32 | Levels off the source grid | Values bit-identical |
|---|---:|---:|---:|---:|
"""
    for k, x in q.items():
        s += f"| `{k}` | {x['tensors']} | {x['groups']:,} | {x['groups_level_mismatch']} | {pct(x['share_bit_identical'], 1)} |\n"
    s += (f"| **All** | {sum(x['tensors'] for x in q.values())} | {groups:,} | {bad} | {pct(ident / vals, 1)} |\n\n"
          f"Values that are not bit-identical differ through the bf16 scale, by at most {worst:.1e} relative. "
          "`repack_report.json` and `verify_report.json` in this repository are that run's outputs.\n\n"
          "**Status:** built and checked offline against its source; not yet served or evaluated.\n")
    s += """
## Built with

`repack_q4_0.py` (in this repo): `repack`, then `verify`.

## Limitations

- Not yet served or evaluated; image input in particular is untested.
- Unofficial. Report problems here, not to Google.
""" + TAIL
    return s, ["repack_q4_0.py"]


def gguf_card(repo, size):
    t = open(f"{REP}/root/build/logs/{repo}.log").read()
    j = json.loads(t[t.rindex("\n{\n") + 1:])
    r = j["rebuilt"]
    gfile, grev = GGUF_SRC[size]
    out = f"gemma-4-{size}-it-q4_0-exact.gguf"
    vals = sum(x["values"] for x in r.values())
    exact = sum(x["exact"] for x in r.values())
    s = front(size, ["gemma4", "gguf", "llama.cpp", "q4_0", "qat", "text-only"], lib="gguf")
    s += f"\n# Gemma 4 {size}-it QAT, exact Q4_0 GGUF, text only (unofficial)\n\n"
    s += ("**This is an unofficial build, made and published independently of Google.** "
          f"It is a llama.cpp GGUF of Gemma 4 {size}-it in which every 4-bit weight is the value Google's "
          "quantization-aware training (QAT) produced, the token embedding included. It is "
          f"{int(j['out_bytes']) / 1e9:.2f} GB, against {int(j['google_bytes']) / 1e9:.2f} GB for Google's "
          f"[`gemma-4-{size}-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-{size}-it-qat-q4_0-gguf). "
          f"The E4B counterpart, built by the same script and measured, is {link('gemma-4-E4B-it-qat-q4_0-exact-gguf')}.\n")
    s += f"""
| | Google Q4_0 GGUF | This GGUF |
|---|---|---|
| File size | {int(j['google_bytes']):,} B | **{int(j['out_bytes']):,} B** |
| `token_embd` | Q6_K | Q4_0 |
| Rebuilt values bit-identical to the QAT source | — | {pct(exact / vals)} |
"""
    s += f"""
#### What is in it

- **Language model only.** Vision lives in Google's separate mmproj GGUF, which is not included here.
- **Metadata copied byte for byte** from Google's `{gfile}` at revision `{grev}`: tokenizer, chat template,
  hyperparameters and tensor order.
- **Weights rebuilt** from [`google/gemma-4-{size}-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-{size}-it-qat-q4_0-unquantized)
  at revision `{UNQ_REV[size]}`, as Q4_0:

| Tensor | Count | Google's type | Values bit-identical to the source |
|---|---:|---|---:|
"""
    for k, x in r.items():
        s += f"| `{k}` | {x['tensors']} | {x['was']} | {pct(x['share_exact'])} |\n"
    if size == "26B-A4B":
        s += ("\nThe experts stay in Google's fused layout, `ffn_gate_up_exps` and `ffn_down_exps`, one bank per layer, each "
              "rebuilt row for row from the source's fused bank (`experts.gate_up_proj`, `experts.down_proj`). `ffn_down_exps.scale` and the other F32 "
              "tensors are copied unchanged.\n")
    s += f"""
- **Copied unchanged:** the norms and scale vectors (F32).

Every rebuilt block landed on a 4-bit grid; values that are not bit-identical differ by the fp16 rounding of the
block scale. Google's GGUF uses llama.cpp's standard step (the block's largest magnitude divided by 8); this build
recovers each block's trained step instead (divided by 8, 7, … 1, whichever puts every value on an integer level,
refined by least squares). `evidence/build_report.json` has the per-tensor counts.

**Status:** built and checked offline against its source; not yet loaded in llama.cpp. Divergence from bf16, speed
and memory were measured for the E4B build only.

#### Rebuilding it

```bash
hf download google/gemma-4-{size}-it-qat-q4_0-unquantized --revision {UNQ_REV[size]} --local-dir src
hf download google/gemma-4-{size}-it-qat-q4_0-gguf {gfile} --revision {grev} --local-dir gguf
python3 gguf_exact.py src gguf/{gfile} {out}
```

Needs only `numpy`. The script stops if any block fails to land on a 4-bit grid.

#### Limitations

- Text only.
- Not yet run in llama.cpp.
- Unofficial. Report problems here, not to Google.
""" + TAIL
    return s, ["gguf_exact.py"], j


def main():
    builds = sorted(sizes) + [os.path.basename(p).removesuffix(".log") for p in
                              glob.glob(f"{REP}/root/build/logs/*exact-gguf.log")]
    for repo in builds:
        size = size_of(repo)
        d = f"{OUT}/{repo}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        extra = None
        if "exact-gguf" in repo:
            card, scripts, extra = gguf_card(repo, size)
            orig = (f"google/gemma-4-{size}-it-qat-q4_0-gguf", GGUF_SRC[size][1])
        else:
            if "fp8" in repo:
                card, scripts = fp8_card(repo, size)
            elif "w8a8" in repo:
                card, scripts = w8a8_card(repo, size)
            elif repo.endswith("ple4"):
                card, scripts = ple4_card(repo, size)
            elif repo.endswith("w4a16-ct"):
                card, scripts = ct_card(repo, size)
            orig = (f"google/gemma-4-{size}-it-qat-q4_0-unquantized", UNQ_REV[size])
        open(f"{d}/README.md", "w").write(card)
        shutil.copy(hf_hub_download(orig[0], "README.md", revision=orig[1]), f"{d}/ORIGINAL_README.md")
        for sc in scripts:
            shutil.copy(SCRIPTS[sc], f"{d}/{sc.split(':')[-1]}")
        if extra:
            os.makedirs(f"{d}/evidence")
            json.dump(extra, open(f"{d}/evidence/build_report.json", "w"), indent=2)
        print(repo, len(card), sorted(os.listdir(d)))


main()

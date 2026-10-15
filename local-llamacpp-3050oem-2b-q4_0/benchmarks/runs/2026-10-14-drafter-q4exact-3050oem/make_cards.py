"""Hugging Face model cards for the exact Q4_0 drafters.

    python3 make_cards.py OUT_DIR

Writes OUT_DIR/{E2B,E4B}/README.md, ORIGINAL_README.md (Google's card, from the snapshot
`make draft` used to download into ~/models/hf) and evidence/ for
xbill9/gemma-4-E{2,4}B-it-qat-assistant-q4_0-exact-gguf. Every figure comes from the runs named
below. Published 2026-10-14 (E2B 6188294, E4B a946809); 1650 Ti section added the same day
(E2B 4090fee, E4B ae5013c) with huggingface_hub's create_commit of README.md and evidence/.
"""
import sys
import json, os, shutil
RUN = "/home/xbill/gemma4-dev/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem"
R1650 = "/home/xbill/gemma4-dev/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs"
# GTX 1650 Ti: 2026-10-14-drafter-formats-1650ti (precisions at the serving draft length) and
# 2026-10-14-drafter-1650ti (draft lengths, this drafter fetched from this repo).
T1650 = {
 "E2B": dict(n=1, none=78.9, q8=(92.1, 1.167, 0.614), q4=(92.7, 1.175, 0.619), bf=(84.6, 1.072, 0.614),
             vram=(1485, 1595, 1557, 1663), lengths=(1.180, 1.109, 1.019), x3050="1.40x at 2 tokens", best="best on this card"),
 "E4B": dict(n=2, none=38.4, q8=(53.8, 1.401, 0.493), q4=(54.5, 1.419, 0.491), bf=(49.8, 1.296, 0.502),
             vram=(2853, 2965, 2927, 3035), lengths=(1.432, 1.430, 1.321), x3050="1.72x at 2 tokens", best="level with 1 token on this card"),
}
M = {
 "E2B": dict(rev="ebc7e1a211354561464cb82ed6d886792138dcb6", size=59236000, q8=97836704, sha="060f5a0158f8120d25137866b90e04e21cd0d2d23de1db7a9cc3b7f00e60bd11",
             backbone=1536, target="xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf", gtarget="google/gemma-4-E2B-it-qat-q4_0-gguf",
             none=108.2, q8tps=150.1, q4tps=151.1, bf=133.1, q8x=1.388, q4x=1.397, bfx=1.231, acc8=0.475, acc4=0.475, accb=0.475,
             vram=(2795, 2915, 2875), rel8=0.00494, rel4=0.00138, ex8=11.7, ex4=29.4,
             gsm=(0.8976, 0.8992, 0.9022), bfcl=(0.9250, 0.9225, 0.9225), gacc=(0.7974, 0.7972)),
 "E4B": dict(rev="27f8d204f09f2be353d6ff0bf0d012792b13c79f", size=59678368, q8=98672288, sha="dc965cb963e56b86fbc04db8efec83163886085e88514574650a57fa55a2ee24",
             backbone=2560, target="xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf", gtarget="google/gemma-4-E4B-it-qat-q4_0-gguf",
             none=58.2, q8tps=100.4, q4tps=100.0, bf=92.3, q8x=1.723, q4x=1.717, bfx=1.585, acc8=0.487, acc4=0.481, accb=0.484,
             vram=(4415, 4537, 4499), rel8=0.00486, rel4=0.00138, ex8=11.4, ex4=28.7, gsm=(0.9325, 0.9310, 0.9325), bfcl=(0.9100, 0.9150, 0.9150), gacc=(0.8374, 0.8382)),
}
for k, v in M.items():
    repo = f"xbill9/gemma-4-{k}-it-qat-assistant-q4_0-exact-gguf"
    fn = f"gemma-4-{k}-it-qat-assistant-q4_0-exact.gguf"
    src = f"google/gemma-4-{k}-it-qat-q4_0-unquantized-assistant"
    acc = ""
    if v["gsm"]:
        acc = f"""
#### Answer quality with the drafter serving

The target model checks every drafted token, so the drafter changes speed and leaves the target's
choices to the target, apart from rounding in the batched check. {k} target, llama.cpp
`77c4540ce`, 8 requests at once, greedy, 2 drafted tokens:

| | no drafter | Q8_0 drafter | this drafter |
|---|---:|---:|---:|
| GSM8K, 1,319 problems | {v['gsm'][0]:.4f} | {v['gsm'][1]:.4f} | {v['gsm'][2]:.4f} |
| BFCL simple, 400 tool calls | {v['bfcl'][0]:.4f} | {v['bfcl'][1]:.4f} | {v['bfcl'][2]:.4f} |
| Drafted tokens accepted over both | — | {v['gacc'][0]:.4f} | {v['gacc'][1]:.4f} |

Paired on the same problems, every drafted arm's difference from no drafter has a 95% bootstrap
range that includes zero.
"""
    s = T1650[k]
    t1650 = f"""
#### On a GTX 1650 Ti (4 GB laptop GPU)

{k} target `xbill9/gemma-4-{k}-it-qat-q4_0-exact-gguf`, llama.cpp `f95b0d9`, same prompts and method,
{s['n']} drafted token{'s' if s['n'] > 1 else ''} per step ({s['best']}):

| | Q8_0 (llama.cpp converter) | This GGUF | bf16 (Google's weights) |
|---|---:|---:|---:|
| Generation with drafter | {s['q8'][0]} tok/s ({s['q8'][1]:.2f}x) | **{s['q4'][0]} tok/s ({s['q4'][1]:.2f}x)** | {s['bf'][0]} tok/s ({s['bf'][1]:.2f}x) |
| Drafted tokens accepted | {s['q8'][2]:.3f} | {s['q4'][2]:.3f} | {s['bf'][2]:.3f} |
| VRAM, target + drafter, `-c 8192` | {s['vram'][1]:,} MiB | **{s['vram'][2]:,} MiB** | {s['vram'][3]:,} MiB |

Without a drafter the target runs at {s['none']} tok/s in {s['vram'][0]:,} MiB. With this drafter at
1 / 2 / 3 drafted tokens: {s['lengths'][0]:.2f}x / {s['lengths'][1]:.2f}x / {s['lengths'][2]:.2f}x
(RTX 3050: {s['x3050']}). Acceptance is about the same on both cards, so the smaller gain likely
comes from each drafted step costing a larger share of a target step on this card (inferred, not
measured). Use `--spec-draft-n-max {s['n']}` on a card of this class.
"""
    card = f"""---
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- {src}
base_model_relation: quantized
tags:
- gemma4
- gguf
- llama.cpp
- q4_0
- qat
- speculative-decoding
- mtp
- cuda
- text-only
---

# Gemma 4 {k}-it QAT assistant (drafter), exact Q4_0 GGUF (unofficial)

**This is an unofficial build, made and published independently of Google.** It is a llama.cpp
GGUF of Google's speculative-decoding drafter for Gemma 4 {k}-it QAT,
[`{src}`]({"https://huggingface.co/" + src}), with every weight matrix stored at 4 bits on the
grid quantization-aware training produced. It is {v['size']:,} B, against {v['q8']:,} B for the
Q8_0 file llama.cpp's converter writes, and follows Google's weights more closely. Google's model
card is kept unchanged in `ORIGINAL_README.md`.

Use it with a Gemma 4 {k}-it **QAT** target, for example
[`{v['target']}`](https://huggingface.co/{v['target']}) or
[`{v['gtarget']}`](https://huggingface.co/{v['gtarget']}). Google's card says the drafter must
match the target's QAT precision.

```
llama-server -m <{k} QAT Q4_0 target>.gguf -ngl 99 -fa 1 \\
  -md {fn} --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

Needs a llama.cpp with Gemma 4 MTP support (`#23398`, June 2026). Measured on builds `77c4540ce`\n(RTX 3050) and `f95b0d9` (GTX 1650 Ti).

| | Q8_0 (llama.cpp converter) | This GGUF |
|---|---:|---:|
| File size | {v['q8']:,} B | **{v['size']:,} B** |
| Relative error against Google's bf16 weights | {v['rel8']:.5f} | **{v['rel4']:.5f}** |
| Weights bit-identical (dequantized, fp32) | {v['ex8']}% | **{v['ex4']}%** |
| Generation with drafter, {k} target, RTX 3050 | {v['q8tps']} tok/s ({v['q8x']:.2f}x) | **{v['q4tps']} tok/s ({v['q4x']:.2f}x)** |
| Drafted tokens accepted | {v['acc8']:.3f} | {v['acc4']:.3f} |
| VRAM, target + drafter, `-c 8192` | {v['vram'][1]} MiB | **{v['vram'][2]} MiB** |

Speed is level with the Q8_0 file and acceptance is the same. The gain is size, memory and
fidelity. Google's bf16 weights, served unchanged, accept the same share of tokens
({v['accb']:.3f}) and run slower ({v['bf']} tok/s, {v['bfx']:.2f}x), so drafter precision does not
limit acceptance. Without a drafter the target runs at {v['none']} tok/s.

Measured on an RTX 3050 OEM (8 GB), llama.cpp `77c4540ce`, target
`{v['target']}`, flash attention, f16 KV cache: six prompts (code, chat, story, explanation,
facts, summary), 256 tokens, greedy, median of three runs, geometric mean over prompts; 2 drafted
tokens per step. Every drafter precision produced the same greedy replies.
{t1650}{acc}
#### What is in it

- **Built from** [`{src}`]({"https://huggingface.co/" + src}) at revision `{v['rev'][:7]}`:
  4 transformer layers of width 256 (3 sliding-window, 1 full attention), with queries only;
  keys and values come from the target's cache. Two projections join it to the target's
  {v['backbone']}-wide hidden state. The output head is the tied 262,144-token embedding.
- **Every weight matrix as Q4_0** (the embedding, 4 x attention q/o, 4 x MLP gate/up/down, both
  projections), each block's trained step recovered: largest magnitude divided by 8, 7, … 1,
  whichever puts all 32 values on an integer level, refined by least squares and stored as the
  fp16 block scale. No block was off the grid. 96.5–97.1% of values per tensor reproduce the
  bf16 source exactly once rounded back to bf16.
- **Metadata** (hyperparameters, tokenizer, chat template) is llama.cpp's converter output,
  `general.file_type` set to Q4_0. Norms and scales stay F32.
- **Left out by llama.cpp's converter:** the checkpoint's `masked_embedding` centroids and token
  ordering, a clustered shortcut for the output head. llama.cpp scores the full vocabulary instead.
- **Text only.** The drafter has no vision or audio parts.

#### Files

| file | bytes | SHA-256 |
|---|---:|---|
| `{fn}` | {v['size']:,} | `{v['sha']}` |

`evidence/` holds the build script (`gguf_exact.py`), its per-tensor report, the RTX 3050 speed
run's per-prompt table (`speed-summary.txt`) and the GTX 1650 Ti tables
(`speed-summary-1650ti-precision.txt`, `speed-summary-1650ti-draft-length.txt`).

#### License

Apache 2.0, as Google's source. Gemma 4 terms: https://ai.google.dev/gemma/docs/gemma_4_license
"""
    d = os.path.join(sys.argv[1], k); os.makedirs(d + "/evidence", exist_ok=True)
    open(d + "/README.md", "w").write(card)
    shutil.copy(f"/home/xbill/models/hf/gemma-4-{k}-it-qat-q4_0-unquantized-assistant/README.md", d + "/ORIGINAL_README.md")
    shutil.copy(RUN + "/gguf_exact.py", d + "/evidence/")
    shutil.copy(RUN + f"/build-{k}.json", d + "/evidence/build-report.json")
    shutil.copy(RUN + f"/{k.lower()}/summary.txt", d + "/evidence/speed-summary.txt")
    shutil.copy(R1650 + f"/2026-10-14-drafter-formats-1650ti/{k.lower()}/summary.txt", d + "/evidence/speed-summary-1650ti-precision.txt")
    shutil.copy(R1650 + f"/2026-10-14-drafter-1650ti/{k.lower()}/summary.txt", d + "/evidence/speed-summary-1650ti-draft-length.txt")
    print(repo, "ok")

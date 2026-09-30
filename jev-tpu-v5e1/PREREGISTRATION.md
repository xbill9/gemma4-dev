# Preregistration: the 2026-09-30 gap runs on one v5e chip

Written before any of these runs started. Four runs fill the gaps left by the 2026-09-27 to 09-30 sweep
(README, "Every QAT build"). All four use bundle `gaps-<hash>.tgz` (the 2026-09-29 bundle
`lowmem-v5e-0aae4c38.tgz` plus `gen_eval.py`, `build_gen_set.py`, the GSM8K and BFCL records, the
`loadlong` mode in `tpu/w4a16_client.py` and per-arm flags in `tpu/run_quant.sh`), the pinned image
`vllm/vllm-tpu@sha256:19a1a052…`, patches `kvshare.diff wna16.diff moe.diff lowmem.diff textonly.diff` (metadata `jev-patches`)
(lowmem.diff sha256 `27e40c1b…`, as every 2026-09-30 run), `GMM_V2_TILE_VMEM_FRACTION=0.85
MIN_TOKEN_BUCKET=64`, `--max-num-batched-tokens 512`, 40 GB swap, flex-start `v5litepod-1` in `us-west4-a`.

## 1. `2026-09-30-gspeedb-v5e1` and `2026-09-30-g12speedb-v5e1`: throughput of Google's 4-bit exports

Question: are Google's `-qat-w4a16-ct` exports faster or slower than the repacks, which beat them on accuracy?
Arms, load mode at 1 / 4 / 16 requests (26–444 token prompts, 256 output tokens, `ignore_eos`, three passes):

| run | arms | per-arm flags |
|---|---|---|
| gspeedb | E2B Google, E2B repack, E4B Google, E4B repack | `gmu=0.80` |
| g12speedb | 12B Google, 12B repack | `override,gmu=0.85,blocks=40` |

Each Google export is compared with the repack **served on the same VM in the same run**, so the
comparison does not depend on the earlier runs' patch set. The earlier repack numbers (E2B 136 / 532 /
1,906, E4B 75 / 291 / 1,012, 12B 33 / 127 / 388) are a consistency check on the new ones.

## 2. `2026-09-30-longb-v5e1` and `2026-09-30-long12b-v5e1`: long prompts, and generation and tool calling

**Long prompts.** `loadlong` at prompts of about 1,024 and 3,584 tokens, 256 output tokens, 1 / 4 / 16
requests, a unique prefix on every request of every pass, streamed. Reported: output tok/s, total
(prompt + output) tok/s, median time to first token. `--max-model-len 4096` on every arm.

**Generation and tool calling** (`gen_eval.py`, greedy):
- GSM8K test, all 1,319 problems, zero-shot chain of thought, 768 tokens; right when the final number equals the reference.
- BFCL v3 simple, all 400 records, one tool offered, `tool_choice` auto, served with `--tool-call-parser gemma4`;
  right when exactly one call has the right name and accepted arguments (BFCL's AST rules, reimplemented and
  checked against all 400 reference answers in `tests/test_gen_eval.py`).

| run | arms (all `long+gen`, `tools,mml=4096`) |
|---|---|
| longb | E2B bf16 (`google/gemma-4-E2B-it`), E2B W8A8 from QAT, E2B W8A8 + int4 tables, E4B W8A8 + int4 tables, all `gmu=0.80` |
| long12b | 12B W8A8 + int4 tables (`gmu=0.92`), 12B int4 + int4 tables (`gmu=0.72`) |

Comparisons, paired record for record with a 95% bootstrap range (10,000 resamples of records):
E2B W8A8 and E2B W8A8 + int4 tables against E2B bf16, and against each other; 12B W8A8 + int4 tables
against 12B int4 + int4 tables. No bf16 12B or E4B arm fits this chip, so their scores stand alone here.
A difference counts as a loss or gain only when its 95% range excludes zero.

## What would change the recommendation

- A Google export faster than its repack at every concurrency: the README's accuracy-only preference for
  the repack gets a speed caveat.
- With long prompts, a build whose output tok/s ranking against the others inverts: the recommendation
  becomes workload-dependent, and says so.
- A recommended build losing on GSM8K or BFCL (range excluding zero) against its bf16 or 4-bit comparison.

## 3. Completeness fills (added before these runs started)

Every quantized build of a size that fits one v5e chip, and every repack variant, gets at least a suite
score on this chip. Four runs, same bundle and image; patch set and settings as above unless listed.

| run | arm | mode | per-arm flags | fills |
|---|---|---|---|---|
| fillA | 12B fp8 (`-qat-q4_0-fp8-text`) | read+load+gen | `tools,gmu=0.92,mml=6144` | fp8 at 12B |
| fillA | 12B W8A8 rounded from bf16 (glenic, tied head removed) | read+load+gen | `override,tools,gmu=0.92,mml=6144` | QAT vs rounding at 12B |
| fillB | E4B W8A8 rounded from bf16 (glenic, tied head removed) | read+load | `override,gmu=0.80` | QAT vs rounding at E4B |
| fillB | E2B repack, text only (`xbill9/…-w4a16-ct-text`) | read | `gmu=0.80` | the `-text` repacks load |
| fillB | E2B W8A8 + int4 tables as published (`xbill9/…-w8a8-ct-text-emb4`) | read | `gmu=0.80` | the published copy equals the scored one |
| fillC | 26B repack + int4 tables (`xbill9/…-text-emb4`) | read+load | `gmu=0.97,blocks=17` | 26B int4 tables |
| fillC | the same | load | `gmu=0.97,mml=8192` | 26B context with int4 tables |
| fillD | 26B repack (`xbill9/…-w4a16-ct`) | load | `override,gmu=0.97,blocks=17` | 26B throughput |

fillC and fillD use the 26B settings of `2026-09-28-lowmem7-v5e1` (`W4A16_MOE_BF16_SCALES=1 W4A16_MOE_NO_PAD=1
SKIP_IDENTITY_MODEL_JIT=1`, fillD also `EMBED_INT8_GROUP=32`; no `--max-num-batched-tokens`).

**The glenic builds** store `lm_head.weight` beside a tied `embed_tokens`. vLLM's JAX path loads both, which
puts the 12B at about 13.9 GiB of weights, over what the chip can hold with a compiled program.
`strip_tied_head.py` copies each build without the stored head only after checking that the config ties
the embeddings and that the head is byte-identical (sha256) to `embed_tokens`. So every served value is
glenic's. The source revision and both hashes are in each copy's `STRIPPED.json`.

Comparisons: 12B and E4B W8A8 from QAT against glenic's, paired on the suite (as E2B's +0.015); 12B fp8
against 12B W8A8 from QAT on the suite; the two E2B `-text` arms against their scored counterparts.

## 4. fillE: generation for the 4-bit builds (added before the run started)

Added after section 2's first results showed E2B's int8 builds losing about 2 points on GSM8K against bf16
while matching it on the suite. Same bundle and settings as section 2, all `gen`, `tools,mml=4096,gmu=0.80`:
E2B QAT bf16 (`google/gemma-4-E2B-it-qat-q4_0-unquantized`), E2B 4-bit repack, E4B 4-bit repack.
Comparisons, paired as in section 2: each E2B arm against E2B bf16 and against E2B W8A8 from QAT; the E4B
repack against E4B W8A8 + int4 tables.

## 5. fillF: fp8 KV cache for the 12B recommended build (added before the run started)

Section 2's long-prompt read shows 12B W8A8 + int4 tables bound by its 9,728-token bf16 KV cache (95 output
tok/s and 22 s to first token at 16 requests of about 3,600 tokens). One arm, 12B W8A8 + int4 tables,
`read+long+gen`, `tools,mml=4096,gmu=0.92`, run-wide `--kv-cache-dtype fp8`. Reported: KV tokens from the
boot log, long-prompt throughput against section 2's arm, and suite, GSM8K and BFCL paired against the bf16-KV
arms of the same checkpoint (suite: `2026-09-29-12b-v5e1`; GSM8K and BFCL: `2026-09-30-long12b-v5e1`).
v5e has no fp8 compute (`../HARDWARE.md`), so the cache is storage only.

## 6. gen2048: GSM8K with a 2,048-token answer limit (added before the runs started)

At 768 tokens GSM8K truncates builds unequally (E2B: bf16 53, QAT bf16 86, W8A8 78), so part of every GSM8K
difference so far is answer length. Every arm that ran `gen` runs it again with metadata
`jev-gen-max-tokens=2048`, same flags otherwise, bundle `gaps2-bd571dd4.tgz` (`gen_eval.py` reads the limit
from `JEV_GEN_MAX_TOKENS` and records it in every record). BFCL runs again unchanged (its 512 limit truncated
nothing).

| run | arms |
|---|---|
| gen2048a | E2B bf16, E2B QAT bf16, E2B W8A8 from QAT, E2B W8A8 + int4 tables, E2B 4-bit repack |
| gen2048b (rerun as gen2048b2: the first left no log) | E4B 4-bit repack, E4B W8A8 + int4 tables, 12B fp8, 12B W8A8 rounded from bf16 |
| gen2048c | 12B W8A8 + int4 tables, 12B int4 + int4 tables |

The 2,048-token results replace the 768-token ones as the GSM8K headline; the 768-token results stay in the
record as the reading at the shorter limit. Comparisons are the pairings of sections 2 and 4, plus 12B fp8 and
12B rounded against 12B W8A8 + int4 tables. A pairing also reports how many records either arm truncated at
2,048; if that exceeds 1% of records the limit is reported as still binding.

## 7. gen2048-v6e1: the bf16 references for GSM8K and BFCL (added before the run started)

bf16 E4B and 12B do not fit one v5e chip, so their GSM8K and BFCL references run on one v6e chip (Compute
Engine, flex-start, `europe-west4-a`, `ct6e-standard-1t`), bundle `gaps3-3852bad5.tgz` (section 6's plus
`jev-self-delete=instance`), same patch list, `jev-gen-max-tokens=2048`, `--max-num-batched-tokens 512`.
Arms, all `gen` with `tools,mml=4096`: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12B-it`
(`override`, as the v6e 12B bf16 suite run). E2B bf16 is the chip check: it pairs against the v5e E2B bf16 arm
of section 6, and a difference whose range excludes zero means the cross-chip pairings below are not clean.
Pairings: every v5e E4B and 12B arm of section 6 against its v6e bf16 reference.

## 8. fillG: 26B context with int4 tables, rerun (added before the run started)

fillC's context arm failed downloading from Hugging Face (HTTP 429) and measured nothing. Rerun from a bucket copy
(`gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4`, revision `38971d33`, every file checked against Hugging Face's
sha256), fillC's settings, `load` mode, `gmu=0.97` with no block cap, at `--max-model-len` 8192, then 4096, then
3072. Reported: KV tokens from each boot log, and throughput for whichever lengths serve.

fillG's 8,192-token arm was refused before compiling: 1.72 GiB of KV needed against 1.51 GiB available inside the
0.97 cap (vLLM's estimate: 7,168 tokens). That estimate leaves no room for the compiled model (about 1.26 GiB,
`2026-09-28-lowmem7`), and without a block cap vLLM gives the KV pool all of it, so the two remaining uncapped
arms could only fail at compile. fillG was stopped and replaced by **fillG2**: same checkpoint and settings,
`load`, `gmu=0.97` with an explicit cap: 24 blocks at `--max-model-len 3072`, then 20 blocks at 2560. The largest
that serves is the 26B's context with int4 tables on this chip; if neither serves, 2,176 tokens (fillC) stands.

Section 7 was not run: the flex-start request waited 1 h 22 min without capacity and was cancelled on
2026-09-30 at the maintainer's request. E4B and 12B still have no bf16 reference for GSM8K or BFCL.

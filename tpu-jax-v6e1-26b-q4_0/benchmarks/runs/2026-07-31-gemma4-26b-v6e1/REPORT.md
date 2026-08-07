# gemma-4-26B-A4B on the pure-JAX engine — port report

**Date:** 2026-07-31
**Checkpoint:** `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`
**Target:** TPU v6e-1 (33.55 GB HBM)

Status: **ported and verified against the HF reference on CPU. Not yet run on a
TPU.** Everything below with a number attached was measured; the TPU sections say
so explicitly where they are projections. `ports/gemma4/jax_26b_port.py` is the
script that turns the projections into measurements.

---

## 1. Why this size needed its own port

Two things separate the 26B from every Gemma 4 already running here. Everything
else — proportional RoPE on full-attention layers, `attention_k_eq_v`,
`global_head_dim` 512, logit softcapping, no PLE, no KV sharing — is shared with
the 31B and needed no new code.

### 1.1 There is no `-w4a16-ct` checkpoint

Enumerated from the Hub, 2026-07-31:

| Size | `-qat-w4a16-ct` | `-qat-q4_0-unquantized` | `-qat-q4_0-gguf` | mobile |
| :--- | :---: | :---: | :---: | :---: |
| E2B  | ✅ | ✅ | ✅ | ✅ |
| E4B  | ✅ | ✅ | ✅ | ✅ |
| 12B  | ✅ | ✅ | ✅ | — |
| **26B A4B** | **❌** | ✅ | ✅ | — |
| 31B  | ✅ | ✅ | ✅ | — |

`google/gemma-4-26B-A4B-it-qat-w4a16-ct` does not exist. GGUF targets llama.cpp,
so the only usable export is 51.61 GB of BF16 against 33.55 GB of HBM.

### 1.2 It is sparse

From `config.json`: `num_experts: 128`, `top_k_experts: 8`,
`moe_intermediate_size: 704`, `enable_moe_block: true`, 30 layers (25 sliding /
5 full), `hidden_size: 2816`.

The expert bank runs **alongside** the dense MLP on every layer and the two sum
before a shared post-norm — it does not replace it. The engine had no router and
no expert gather before this port.

---

## 2. The weights are already quantized (measured)

The `-q4_0-unquantized` export is QAT-quantized data in an unquantized container.
Probed by range-reading tensors straight out of the safetensors shards, before
any code was written:

| Tensor | groups of 32 | on a 4-bit grid |
| :--- | ---: | ---: |
| `layers.0.experts.gate_up_proj` | 256 | **256** |
| `layers.0.experts.down_proj` | 256 | **256** |
| `layers.0.self_attn.q_proj` | 256 | **256** |
| `layers.0.mlp.down_proj` | 256 | **256** |
| `layers.0.router.proj` | 256 | **256** |
| `embed_tokens` | 256 | **256** |

Group size **64** fails the same test (3/128), which is what pins the group size
at 32 rather than leaving it assumed.

### 2.1 The trap: `d = amax / 8` is wrong here

The textbook Q4_0 rule assumes each block's largest magnitude sits at level ±8.
Plenty of blocks in this checkpoint peak lower. When they do, the derived step is
a fraction of the true one, `round(x/d)` lands between grid points, and the block
is re-quantized onto a grid that does not contain its own values:

| step rule | median relative error on `experts.gate_up_proj` |
| :--- | ---: |
| `d = amax/8` (textbook) | **4.9e-2** |
| `d = amax/m`, m searched over 1..8 | 0 for 78.7% of values |
| ... plus least-squares refinement | **0 for 93.1% of values** |

Nothing raises in the first case. The model loads, generates fluent text, and is
5% wrong in every expert weight. `quantize_q4_0` therefore *searches* for the
level the peak occupies and refines the step by least squares over the group;
`_recover_step` returns a count of groups it could not place, and the loader
raises on any nonzero count rather than logging it.

### 2.2 Round-trip fidelity, against the checkpoint bytes

Ten tensors spanning every kind in the model (expert banks, attention, MLP,
router, embeddings, a sliding and a full-attention layer):

```
0 groups unrepresentable out of 52,096
89-93% of values reconstruct BIT-IDENTICALLY
worst-case relative error 6.4e-3  (~1.6 BF16 ULP)
```

The residue is scale precision, not level assignment — the integer levels are
recovered exactly (refining the step moves zero levels). Q4_0 carries an fp16
block scale; this format stores BF16, which is three mantissa bits shorter. An
fp16 scale would reach 97% bit-exact at the *same* two bytes per group, but only
if the dequant multiplies in f32, which puts an f32 multiply on the decode hot
path. Not taken — revisit with a benchmark, not an argument.

This is the same precision regime the shipped `-w4a16-ct` checkpoints already run
in on this engine.

### 2.3 Resident size

Projected from `quantized_bytes`, cross-checked against the real config:

| | packed |
| :--- | ---: |
| expert bank / layer | 0.428 GB |
| dense MLP / layer | 0.010 GB |
| attention / layer | 0.020 GB (sliding), 0.028 GB (full) |
| `embed_tokens` (left BF16 — a gather, not a matmul) | 1.476 GB |
| **total** | **15.25 GB** |

Against 51.61 GB as shipped, and 33.55 GB of HBM: ~18 GB free for KV and
activations, versus 14.25 GB free for the dense 31B.

**Unverified on hardware.** The load stage of the port script prints measured
resident bytes next to this projection.

---

## 3. Correctness

Verified against `transformers.models.gemma4` (v5.12.1) in float32 on both
sides — BF16 would put a 1e-2 noise floor under everything and hide exactly the
kind of error being looked for. `tests/test_moe_parity.py`, 8 tests:

* router — top-k **indices exact**, weights < 2e-4
* experts — both dispatch paths < 2e-4 against the reference `index_add`
* full model (embeddings → attention → MoE → norms → tied lm_head → softcap) —
  < 5e-3, argmax identical
* gather vs dense agree to 1e-5

### 3.1 The bug this caught

The reference routes on the **raw** post-attention residual
(`self.router(hidden_states_flat)`) but feeds the experts that residual after
`pre_feedforward_layernorm_2`. Two different tensors. The first implementation
passed the normalized one to both — the obvious reading.

Every unit test still passed. Router parity passed. Expert parity passed. The
full model was off by **0.36 relative** and would have generated perfectly
fluent, subtly wrong text. It is caught only by comparing the whole stack against
the reference, which is why that test exists and why it runs in f32.

Guarded now by two dedicated regression tests, since the merge is a natural
"simplification" for a future reader to make.

Not a mistake in the same class, but worth recording: the first version of the
parity harness passed a **boolean** causal mask to an engine that does
`scores = scores + mask`, so every allowed position got +1.0 and every forbidden
one +0.0. That failed against a *correct* engine and briefly looked like an
engine bug.

---

## 4. Known cost: prefill does 16× the expert FLOPs

Decode gathers only the 8 selected expert banks per token. Prefill cannot — at
T tokens it would re-read T×8 banks — so it dequantizes the whole 128-expert bank
once and masks, which is optimal in *bytes moved* and 128/8 = **16× wasteful in
FLOPs**. That ratio is arithmetic from E and K, not a measurement.

The fix is expert-sorted dispatch: bucket the T×K (token, expert) pairs by expert
and do one batched matmul per expert over just its own rows, via a
capacity-padded gather or `lax.ragged_dot`. Deliberately not attempted in this
port, because capacity-padded dispatch **drops tokens** when an expert is
oversubscribed, and a silent drop is precisely the failure mode this engine keeps
producing. Land it behind a parity test against `moe_experts_dense`, with the
overflow count returned and asserted rather than inferred.

The `moe` stage of the port script measures where gather and dense actually
cross, rather than trusting the T=16 threshold the byte-counting argument
predicts. This engine has been wrong before about bytes predicting time — the
fused W4A16 Pallas kernel was a 0.59× "optimization" because decode was not
bandwidth-bound.

---

## 5. Reproducing

```bash
# anywhere (reads config.json only)
python3 ports/gemma4/jax_26b_port.py --stages config

# on a v6e-1
python3 ports/gemma4/jax_26b_port.py --stages config,load,parity,moe,sweep

# correctness, CPU, needs torch + transformers
python3 -m pytest tests/test_moe_parity.py tests/test_q4_0_requant.py \
                  tests/test_moe_engine_smoke.py -q
```

## 6. Open items

1. Run the port script on a v6e-1: confirm 15.25 GB resident, get real
   prefill/decode numbers, and check output quality on the templated prompt.
2. Measure the gather/dense crossover and correct `MOE_GATHER_MAX_TOKENS` if the
   byte-counting threshold does not survive contact.
3. Expert-sorted prefill dispatch (§4) — the largest performance item.
4. Decide whether `embed_tokens` should be packed too (saves 1.06 GB, costs a
   dequant in the lm_head and the embedding gather). Not needed to fit.

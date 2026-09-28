#!/usr/bin/env python3
"""Gemma 4 26B-A4B expert matmuls on one TPU chip: the served layout against the low-memory one.

Runs inside the patched vLLM TPU image. For 128 experts, top-8 routing and a few token counts,
computes the MoE MLP (gmm1 -> act(gate) * up -> gmm2) two ways with the same int4 weights:
  served  intermediate dim padded 704 -> 768, float32 scales, activation fused into gmm1
  lowmem  unpadded, bfloat16 scales, gmm1 unfused and the activation in JAX
            (W4A16_MOE_NO_PAD + W4A16_MOE_BF16_SCALES, lowmem.diff and tokamax-bf16-scale.diff)
and checks each against an XLA reference that dequantizes every expert and uses jnp.dot. Every
case runs on its own, so one failing case does not stop the rest. Prints one JSON line per case
and a summary; every figure is computed here.
"""
import json
import statistics
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.gmm_v2.gmm_v2 import gmm_v2

from tpu_inference.layers.common.fused_moe_gmm import _gate_up_act

E, TOPK, D, F, PAD_F, GROUP = 128, 8, 2816, 704, 768, 32
ACT = "gelu"  # what Gemma4MoE passes as hidden_act
TOKENS = [1, 16, 128, 1024]
ITERS, REPS = 20, 5


def timed(fn, *args):
    out = fn(*args)
    out.block_until_ready()
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            last = fn(*args)
        last.block_until_ready()
        samples.append((time.perf_counter() - t0) / ITERS)
    return out, statistics.median(samples)


def gmm(lhs, rhs, scale, sizes, fuse_act=None, out_dtype=None):
    return gmm_v2(lhs=lhs, rhs=rhs, rhs_scale=scale, rhs_bias=None, group_sizes=sizes,
                  group_offset=jnp.array(0, jnp.int32), zero_initialize=False,
                  fuse_act=fuse_act, preferred_element_type=out_dtype or lhs.dtype)


@jax.jit
def served(x, w1, s1, w2, s2, sizes):
    h = gmm(x, w1, s1, sizes, fuse_act=ACT)[:, :F]
    return gmm(h, w2, s2, sizes)


@jax.jit
def lowmem(x, w1, s1, w2, s2, sizes):
    h = _gate_up_act(gmm(x, w1, s1, sizes, out_dtype=jnp.float32), ACT, x.dtype)
    return gmm(h, w2, s2, sizes)


@jax.jit
def _expert(x, gate, up, down):
    g = jnp.dot(x, gate, preferred_element_type=jnp.float32)
    u = jnp.dot(x, up, preferred_element_type=jnp.float32)
    h = (jax.nn.gelu(g) * u).astype(x.dtype)
    return jnp.dot(h, down, preferred_element_type=jnp.float32).astype(x.dtype)


def reference(x, gate, up, down, sizes):
    """Dequantized bf16 weights [E, in, out]; each expert's rows through plain XLA matmuls."""
    out, start = [], 0
    for e, n in enumerate(np.asarray(sizes)):
        if n:
            out.append(_expert(x[start:start + n], gate[e], up[e], down[e]))
        start += n
    return jnp.concatenate(out)


def main():
    print(json.dumps({"device": str(jax.devices()[0]), "jax": jax.__version__}), flush=True)
    rng = np.random.default_rng(0)
    q_gate = rng.integers(-8, 8, (E, D, F), dtype=np.int8)
    q_up = rng.integers(-8, 8, (E, D, F), dtype=np.int8)
    q_down = rng.integers(-8, 8, (E, F, D), dtype=np.int8)
    s_gate = jnp.asarray(rng.uniform(0.001, 0.02, (E, D // GROUP, F)), jnp.bfloat16)
    s_up = jnp.asarray(rng.uniform(0.001, 0.02, (E, D // GROUP, F)), jnp.bfloat16)
    s_down = jnp.asarray(rng.uniform(0.001, 0.02, (E, F // GROUP, D)), jnp.bfloat16)

    def deq(q, s):
        e, k, n = q.shape
        return (jnp.asarray(q, jnp.float32).reshape(e, k // GROUP, GROUP, n)
                * s.astype(jnp.float32)[:, :, None, :]).reshape(e, k, n).astype(jnp.bfloat16)

    ref_w = (deq(q_gate, s_gate), deq(q_up, s_up), deq(q_down, s_down))
    padn = ((0, 0), (0, 0), (0, PAD_F - F))
    served_w = (
        jnp.asarray(np.concatenate([np.pad(q_gate, padn), np.pad(q_up, padn)], -1)).astype(jnp.int4),
        jnp.concatenate([jnp.pad(s_gate, padn), jnp.pad(s_up, padn)], -1).astype(jnp.float32)[:, :, None, :],
        jnp.asarray(q_down).astype(jnp.int4),
        s_down.astype(jnp.float32)[:, :, None, :])
    low_w = (
        jnp.asarray(np.concatenate([q_gate, q_up], -1)).astype(jnp.int4),
        jnp.concatenate([s_gate, s_up], -1)[:, :, None, :],
        jnp.asarray(q_down).astype(jnp.int4),
        s_down[:, :, None, :])
    for name, w in (("served", served_w), ("lowmem", low_w)):
        print(json.dumps({"layout": name, "expert_bytes_per_layer": int(sum(a.nbytes for a in w)),
                          "scale_dtype": str(w[1].dtype), "gmm1_n": int(w[0].shape[-1])}), flush=True)

    rows = []
    for t in TOKENS:
        m = t * TOPK
        sizes = jnp.asarray(rng.multinomial(m, np.full(E, 1 / E)), jnp.int32)
        x = jnp.asarray(rng.standard_normal((m, D)) * 0.5, jnp.bfloat16)
        b = np.asarray(reference(x, *ref_w, sizes), np.float32)
        for name, fn, w in (("served", served, served_w), ("lowmem", lowmem, low_w)):
            row = {"layout": name, "tokens": t, "rows": m}
            try:
                y, dt = timed(fn, x, *w, sizes)
                a = np.asarray(y, np.float32)
                row.update(ok=True, us=round(dt * 1e6, 1),
                           rel_err_vs_ref=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                           nonfinite=int((~np.isfinite(a)).sum()))
            except Exception as e:  # noqa: BLE001 -- record the failure and keep going
                row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:600]}")
                traceback.print_exc()
            rows.append(row)
            print(json.dumps(row), flush=True)

    print("\n| layout | tokens | us | rel err vs ref | non-finite | error |")
    print("|---|---:|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r['layout']} | {r['tokens']} | {r['us']} | {r['rel_err_vs_ref']:.1e} | {r['nonfinite']} | |"
              if r["ok"] else f"| {r['layout']} | {r['tokens']} | | | | {r['error'][:120]} |")
    ok = [r for r in rows if r["ok"]]
    worst = {n: max((r["rel_err_vs_ref"] for r in ok if r["layout"] == n), default=None)
             for n in ("served", "lowmem")}
    print(f"\n{len(ok)} of {len(rows)} cases ran; worst rel err vs ref: served {worst['served']}, "
          f"lowmem {worst['lowmem']}; non-finite outputs {sum(r['nonfinite'] for r in ok)}")


if __name__ == "__main__":
    main()

# Bisecting the W4A16 miscomputation on Inferentia2

**Run:** 2026-08-03 (same session as `REPORT.md`, after the instance was rebuilt)
**Instance:** `inf2.8xlarge` **spot** ($0.3872/hr), `us-east-2a`, `i-0c5f2238ab1d1f6f0`
**AMI:** `ami-09e1477ba5140fe3e`, Ubuntu 24.04, Neuron SDK 2.31.0
**Stack:** jax / jaxlib 0.9.2, system `python3` (no venv), pytest 9.1.1
**Checkpoint:** `google/gemma-4-E2B-it-qat-w4a16-ct`
**Reference:** `artifacts/ref.json` — PyTorch/CPU/float32 oracle, 4 prompts, 24 tokens

> `REPORT.md` closed by narrowing the fault to two features and named `w4a16`
> the last suspect. This run confirms that and then takes it two levels deeper:
> the defect is **Neuron-specific**, is **not** the unpack primitive, and is
> **not** the dequant math or the loader.

## Where the fault is not

### 1. Not the int4 unpack primitive — bit-exact on the NeuronCore

The leading hypothesis was that `qat_w4a16_unpack_dequant_jax`
(`ports/gemma4/jax_e_model.py:180-184`) miscomputes on device, because its first
step is signed-int32 bit manipulation:

```python
shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
q = ((packed[:, :, None] >> shifts) & jnp.int32(0xF))
```

That is precisely the class of op `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1`
forces onto the host, which would explain why the workaround "fixes" correctness.

`jax_neuron/bisect_w4a16.py` decomposes the function into five stages and
compares each against a plain-numpy host reference on identical inputs. It needs
no model weights and compiles in seconds. Inputs span the full int32 range, so
negative words exercise sign extension in `>>`.

Run on the NeuronCore with **`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=0`**:

| stage | op | result |
|---|---|---|
| `identity` | int32 round trip (control) | PASS, err 0.0 |
| `shift` | signed int32 arithmetic right shift | PASS, err 0.0 |
| `shift_mask` | `>>` then `& 0xF` → nibble value | PASS, err 0.0 |
| `to_bf16` | nibble → bf16, minus zero-point 8 | PASS, err 0.0 |
| `full` | complete unpack + group scale | PASS, err 0.0018 |

Artifact: `artifacts/bisect_unpack_off.json`. The `full` stage's 0.0018 is the
bf16 scale round trip, inside a 5% relative tolerance; the four integer stages
are exact.

**The hypothesis is wrong.** The unpack arithmetic is correct on the NeuronCore
in the exact configuration that garbles the server.

*Caveat:* this is a small isolated graph (256×32 packed). The same ops inside the
full decode graph may be fused and scheduled differently by `neuronx-cc`, so this
clears the arithmetic, not necessarily the op in situ.

### 2. Not the dequant math or the loader — CPU produces coherent text

Same checkpoint, same reference, same engine code, `--subject-platform cpu`:

| platform | first divergence | agreement | output |
|---|---|---|---|
| **CPU** | tokens 15 and 22 (2 of 4 exact) | 100 / 93.8 / 100 / 91.7 % | coherent |
| **Neuron** | **token 0**, all 4 prompts | **0.0** | `Atha Atha Atha`, `nonprofits nonprofits` |

Artifacts: `artifacts/parity_w4a16_cpu.json`, `artifacts/parity_w4a16.json`.

The CPU run's two mismatches are ordinary 4-bit quantization drift against an
unquantized reference — they diverge late and differ by a single word choice
(`shorter wavelengths` vs `shorter blue wavelengths`), with everything before
identical. That is what a correct W4A16 path looks like. Its `passed: false` is
an artifact of the cross-checkpoint comparison (below), not a defect.

## A methodological correction to REPORT.md

**The failing w4a16 comparison was cross-checkpoint.** `ref.json` was built from
`google/gemma-4-E2B-it-qat-q4_0-unquantized`; the w4a16 subject ran
`google/gemma-4-E2B-it-qat-w4a16-ct`. Different weights. The three *passing*
runs (`parity_off`, `parity_on`, `parity_int8kv`) all used the unquantized
checkpoint on both sides, so they were like-for-like — the w4a16 run was not.

This does not explain away the failure: quantization drift does not produce one
token repeated 24 times at 0.0 agreement. But it means REPORT.md's framing did
not distinguish "w4a16 weights miscompute" from "the w4a16 checkpoint is loaded
wrong," and the CPU control above was needed to separate them. It also means
`passed: false` is the wrong headline for the CPU run.

**Fix for future runs:** build a w4a16 reference for w4a16 subjects, or treat
agreement rate rather than exact match as the criterion when the subject and
reference checkpoints differ in quantization.

## Where the fault is — and the fix

**The fault is the IN-GRAPH W4A16 dequant.** The same arithmetic performed on
the host is correct on the NeuronCore, with the workaround off, at full speed.

`parity.py --dequant-at-load`, `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=0`,
subject `neuron`, same checkpoint and reference as every run above:

| configuration | output | agreement | speed |
|---|---|---|---|
| in-graph dequant, workaround off | `Atha Atha Atha` | 0.0, diverges at token 0 | 5.4 tok/s of garbage |
| workaround `=1` (REPORT.md) | `Paris.` — correct | — | 5 tokens in 77–84 s |
| **host dequant, workaround off** | **coherent** | 1.0 / 0.938 / 1.0 / 0.75 | 16 tokens in 1.4 s |

Artifact: `artifacts/parity_w4a16_hostdequant.json`. Two of four prompts match
the unquantized reference exactly; the other two diverge late (tokens 15 and 18)
on single word choices — the same signature as the CPU run, i.e. ordinary 4-bit
drift, not a defect.

So the residual defect is `neuronx-cc`'s handling of the fused
dequant-and-matmul at real shapes — fusion, scheduling, or layout — and not any
single operation's semantics. That remains open upstream;
`--dequant-at-load` routes around it entirely.

**Cost of the fix:** dense BF16 weights, 9.26 GB against 6.56 GB packed, and a
28.8 s load. On any inf2 host that is free; it is the trade
`dequantize_params_to_dense` was written for (`jax_e_model.py:1113`), which notes
E2B is 3.7 GiB dense against 31.24 GiB of HBM.

### Shipped

- `jax_openai_server.py` — `--dequant-at-load` plumbed through `load_engine`.
- `deployments/aws-inf2/neuron_entrypoint.py` — the
  `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU` default flipped `1` → `0`, with the
  evidence inline so the 65x workaround is not reinstated by reflex.
- `deployments/aws-inf2/user_data.sh` — the service passes `--dequant-at-load`,
  and the env file sets the variable **explicitly**. The entrypoint only
  `setdefault()`s it, so an env file still carrying `1` would silently reimpose
  the cost — which is exactly how the old host was configured.

## End-to-end validation through the HTTP server

The parity harness drives `JaxGemmaEngine` in-process; the server adds FastAPI,
the entrypoint's env vars, and `neuron_entrypoint.py`. Validated separately, on
the live service with `--dequant-at-load` and
`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=0`:

```
0.16s   5 tok  30.4 tok/s  'The capital of France is'  -> 'Paris.'
0.37s  16 tok  43.3 tok/s  'Q: What is 17 * 23?\nA:'   -> '17 * 23 = 391'
0.37s  16 tok  43.4 tok/s  'why the sky is blue'       -> "...sunlight is scattered by the Earth's"
0.37s  16 tok  43.3 tok/s  'def fibonacci(n):'         -> 'if n <= 1:\n        return n\n    else:'
```

`tpu_jax_requests_total` — 7 success, **0 failed**. `/health` reports
`weights: bf16` (was `w4_int4`), which is how you confirm the flag took effect.

**These are warm, second-pass numbers and they are not engine capability.**
Batch 1, greedy, 16 tokens, one bucket. Quoting them as throughput would violate
the measurement rule in `CLAUDE.md`. Their only load-bearing use here is the
comparison against the *same* configuration's alternatives:

| | correct? | observed |
|---|---|---|
| workaround `=1`, in-graph dequant | yes | 5 tokens in 77–84 s |
| workaround `=0`, in-graph dequant | **no** | garbage at 5.4 tok/s |
| workaround `=0`, **host dequant** | **yes** | 16 tokens in 0.37 s |

The first pass over the last two prompts ran at 1.74 tok/s — first touch of a new
bucket, not steady state. That is the same trap REPORT.md names: a slow first
request is `neuronx-cc` compiling, not a hang. Check `tpu_jax_requests_total` and
`pgrep neuronx-cc` before concluding anything from a single timing.

### Validated across a restart, from the persisted configuration

Hand-editing a running host proves nothing about what a fresh launch does, so the
service was restarted and re-tested with no manual intervention. Three warmth
states, distinguished rather than averaged — REPORT.md's closing caveat was that
it had quoted three different warmth states as though they were comparable:

| pass | condition | rate |
|---|---|---|
| 1st, cold process | fresh process, NEFFs reloading from cache | 0.82 → 17.4 tok/s |
| 2nd, settled | same process, all buckets resident | **43.1 – 43.7 tok/s** |
| 2nd, settled (pre-restart) | independent process, same buckets | **42.9 – 43.4 tok/s** |

Output was byte-identical across all of them, and `/health` reported
`weights: bf16` after the restart, confirming the flag came from
`/usr/local/bin/gemma4-jax-inf2-run` and `/etc/gemma4-inf2.env` rather than from
anything typed by hand. `tpu_jax_requests_total`: 4 success, 0 failed.

The settled figure reproduces to within 2% across two independent processes,
which is what makes it a measurement rather than an anecdote — but it is still
batch 1, greedy, 16 tokens, one bucket, and must not be quoted as engine
capability.

## Regression-test gap — closed

The repo suite passes completely on the Inf2 host — **175 passed, 1 skipped, 10
subtests, 0 failures in 154 s** (15 of 17 modules; `test_inf2_mcp_jax.py` and
`test_server.py` need `mcp` and `google.cloud`, neither installable-relevant on
Inf2). That includes `test_ple_quantization.py` and `test_quantized_kv.py`.

**Nothing in `tests/` covered this defect.** A passing suite was not evidence for
the W4A16 path on Neuron — the suite went 175/175 on the very host that was
serving garbage. That is exactly the failure mode `CLAUDE.md` warns about: code
that runs, reports success, and computes the wrong thing.

`tests/test_w4a16_host_dequant.py` (7 tests, all passing) closes what can be
closed off-device. The device-level defect cannot be reproduced without a
NeuronCore, so the tests guard the invariants that survive off it:

1. the numpy twin agrees with the JAX implementation — if it drifts, the fix
   silently serves different weights than were validated;
2. hand-decoded nibbles, independent of both implementations, so a wrong layout
   convention fails even when numpy and JAX agree with each other;
3. `on_host=True` and `on_host=False` agree over a params tree — the flag is a
   placement choice, never a semantic one;
4. the deployment does not reinstate `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1`,
   and the service still passes `--dequant-at-load`.

(4) is asserted against the deployment files rather than library code on purpose:
both regressions are silent. Re-enabling the workaround costs ~65x while output
stays *correct*, so no output check would catch it; dropping the flag makes
output wrong while every unit test still passes. The test caught a real instance
of the second case immediately — the deployed tree on the live host still had the
pre-fix `user_data.sh`.

## Tooling added

- `jax_neuron/bisect_w4a16.py` — stage-by-stage device-vs-host comparison of the
  unpack path. No weights, seconds to compile.
- `jax_neuron/parity.py --dequant-at-load` — materializes packed weights to dense
  BF16 on the **host** before `device_put` and runs the dense path. Identical
  arithmetic; only *where* the dequant executes changes. `dequant_at_load`
  existed on `JaxGemmaEngine` (`jax_engine.py:324`) but was not reachable from
  the parity harness.

## Operational notes

- **Only one process can hold the NeuronCores.** The serving unit must be
  stopped (`systemctl stop gemma4-jax-inf2`) before any parity or bisect run, or
  JAX fails at backend init with `NRT_FAILURE status_code=1`.
- **Cold rebuild is cheap.** A fresh spot instance with an empty cache volume was
  serving-ready in ~7 minutes: 7.8 GB checkpoint pulled in ~3 min, and the
  `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1` path needs only 51 NEFF modules
  (1.3 MB) against the 145 modules / 102 MB the `=0` path built. Retaining a
  200 GiB cache volume across sessions is not obviously worth its $16/mo.
- **inf2 spot capacity is scarce but not absent.** Placement scores were 1/10
  across us-east-1 and us-east-2 and 3/10 in us-west-2; us-east-2c failed with
  `InsufficientInstanceCapacity`, us-east-2a succeeded at an 80% discount.
  EBS is AZ-locked, so a cache volume pins the AZ and costs you the retry.

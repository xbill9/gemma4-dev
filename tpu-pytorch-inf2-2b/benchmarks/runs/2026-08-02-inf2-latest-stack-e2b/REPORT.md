# Does the latest Neuron/JAX stack dissolve the 2700x workaround?

**Run:** 2026-08-02
**Instance:** `inf2.xlarge` on-demand, `us-east-2c`, `i-0022c706bb9e89263`
**AMI:** `ami-09e1477ba5140fe3e`, Ubuntu 24.04, Neuron SDK 2.31.0, `aws-neuronx-runtime-lib` 2.33.10.0
**Stack:** jax / jaxlib 0.9.2, jax-neuronx 0.10.0.1.0.10466, libneuronxla 3.0.3854.0,
neuronx-cc 2.26.6360.0, transformers 5.14.1
**Checkpoint:** `google/gemma-4-E2B-it-qat-w4a16-ct` (W4A16, int8 KV, `--max-model-len 4096`)

> **Headline, after the parity section below:** the upgrade did not dissolve the
> fault, but parity localized it. The engine is token-exact on Neuron *with the
> workaround off* at ~14 tok/s, for both windowed and unwindowed KV. The fault
> lives in the `w4a16` + `int8`-KV serving configuration, not in the engine.

## Result: no. The fast path is 65x faster and still computes garbage.

`benchmarks/runs/2026-07-31-inf2-serving-perf/REPORT.md` closed on a single
question — whether the op that miscomputes on the NeuronCore, and forces
`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1` as a correctness workaround, was a bug
that a newer compiler or JAX would fix. It is not. Every component moved forward
by a major step and the fault survived intact.

Same host, same checkpoint, same request, varying only the workaround. All
timings are warm — the NEFFs were compiled and cached first, and the numbers
below are steady-state, not first-call:

| | `=1` (workaround on) | `=0` (workaround off) |
|---|---|---|
| warm request, 5 tokens | **77–84 s** | **1.18 / 1.18 / 1.52 s** |
| output | `Paris.` | `Atha Atha Atha Atha Atha` |
| correct | yes | **no** |
| prefill | — | 240.4 ms |
| decode | — | 5.4 tok/s |

The workaround costs **~65x** on this stack, against the 2700x measured on jax
0.6.2 / neuronx-cc 2.24. That ratio narrowed because the *slow* path got faster
(77–84 s against ~126 s), not because the fast path changed.

**The symptom changed, and that is the new information.** On the old stack,
unsetting the variable produced an empty completion — EOS on the first sampled
token, zero completion tokens, ~0.01 s. Here it produces five deterministic
garbage tokens at full speed. Same root cause or not, the failure now emits
tokens instead of terminating, which is a different observable and a better
handle: a repeated single token points at the logits path or the sampler rather
than at premature termination.

## Parity: the engine is correct on Neuron, and the workaround is not needed

Run later the same session, against a PyTorch/CPU/float32 oracle built on the
same host (`artifacts/ref.json`, 4 prompts, 24 tokens, 536.7 s). Both runs
execute with `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU` **unset** — `parity.py`
never sets it — so this is the configuration that garbles the HTTP server:

| subject | result | decode |
|---|---|---|
| `--window-kv off`, fp16 weights, bf16 KV | **PASS 4/4** | 24 tok in 10.3 s |
| `--window-kv on`, fp16 weights, bf16 KV | **PASS 4/4** | 24 tok in 1.7 s |

Artifacts: `artifacts/parity_off.json`, `artifacts/parity_on.json`, both
`"passed": true`.

Three conclusions, and they substantially rewrite this report's premise:

- **The workaround is not required by the engine.** Token-exact output against an
  independent implementation, at ~14 tok/s, in the exact configuration that
  produces `Atha Atha Atha` through the server. The attention stack, the decode
  loop, the sampler, and the logits path are all correct on Neuron.
- **Windowed KV is validated on Neuron.** Both settings pass, so the
  `take_along_axis` gather introduced by the 7/31 windowed-KV fix compiles *and*
  computes correctly. This closes the open item that report left when spot
  reclaimed the instance mid-run — and it retires the leading hypothesis, since
  the windowed path was the prime suspect going in.
- **The fault is confined to the quantized serving path.** Parity ran fp16
  weights with bf16 KV; the failing server runs `w4a16` weights with `int8` KV.
  Those are the only remaining differences. The suspect list went from "some op
  in the decode graph" to two specific, independently testable features.

Note the load-time asymmetry, unexplained: 147.7 s for the first subject load
against 22.4 s for the second, both 9.26 GB. Probably NEFF cache warmth across
the two runs rather than anything about windowing.

## Three defects fixed on the way, all in the deployment rather than the engine

**1. `libneuronxla==2.2.*` was obsolete, not load-bearing.** The pin existed
because jax-neuronx 0.6.2.1.0 declared an unbounded `libneuronxla>=2.2.12677.0`,
so pip took 3.0.3854.0, which targets an NRT 3.0 runtime that the SDK-2.29.1 AMI
did not ship — surfacing much later as `undefined symbol:
nrta_event_register_xu_completion, version NRT_3.0.0`. Pinning jax-neuronx to
the current line and dropping the libneuronxla pin entirely resolves
**3.0.3854.0** — the exact build the pin existed to avoid — and it works, because
SDK 2.31.0 ships NRT 2.33.10.0. The pin was pure carried cost from the moment the
AMI moved.

**2. `JAX_COMPILATION_CACHE_DIR` is a hard crash on jax 0.9.2.** Collateral
finding #4 of the 7/31 report noted that `user_data.sh` set it while
`ports/gemma4/backend.py:134` declares `persistent_compilation_cache=False` for
Neuron, and could not say which side was wrong. **The capability table was
right.** On jax 0.6.2 the contradiction was silently tolerated (3.9 MB of entries
written, no symptom). On jax 0.9.2 the service crash-loops at startup before the
model loads:

```
RET_CHECK failure (xla/hlo/ir/hlo_module.cc:822)
proto.has_host_program_shape()  No program shape found in the proto
```

Unsetting it is the entire fix. This is the upgrade doing useful work: it
converted a latent misconfiguration into an immediate, legible failure.

**3. `--cache_dir` is not a `neuronx-cc` argument.** Collateral finding #3 held
that the NEFF cache never populated because `NEURON_COMPILE_CACHE_URL` is a
torch-neuronx variable and neuronx-cc wants `--cache_dir` in `NEURON_CC_FLAGS`.
Adding it makes **every compile fail**:

```
[NCC_EARG002] Illegal argument(s) - the following argument(s) are unrecognized:
--cache_dir=/opt/gemma4/cache/neuron
```

The cache does populate on this stack — `$CACHE_ROOT/neuron/neuronxcc-2.26.6360.0+.../MODULE_*/`
with `model.hlo_module.pb` and `compile_flags.json` — via `NEURON_COMPILE_CACHE_URL`,
the variable finding #3 dismissed. It reached **102 MB across 145 `MODULE_*`
directories** once the workaround-off path forced real compilation, against 3.6 MB
beforehand, so this is genuine compiler output and not a stub. That finding's
conclusion should be treated as inverted for this stack. Note this run did **not**
isolate the mechanism by A/B-ing the variable; it observed the cache populating
with the variable set and the flag absent.

This also settles the practical question finding #3 raised — whether a
build-on-one-box / serve-on-another split can work. The artifact survives on the
cache volume, and with the phase markers (`neuron-probe`, `os-packages`,
`python-deps`) on the root volume plus the 7.8 GB checkpoint on the cache volume,
a stop/start skips the download, the pip install, and all 145 compiles.

## Other measurements

- **Parameter load: 156.5 s** for 6.56 GB onto one NeuronCore, against 263.0 s
  (1 core) and 68.8 s (2 cores) on the 7/31 stack. Not a controlled comparison —
  different AMI, different jax, different core count.
- **First request with the workaround off exceeded 900 s** and returned nothing
  to `curl`, while `/metrics` recorded it as a success. That is `neuronx-cc`
  building `MODULE_jit_prefill_with_kv_cache` from cold, and it is the trap the
  7/31 report already names: check `tpu_jax_requests_total` before concluding a
  hang. Host memory reached 14/15 GB during the compile on a 16 GB box.

## A caveat on the A/B, found while restoring the workaround

The two columns above were each measured warm *within their own process*, but
they are not warm in the same sense, and a restart exposed the difference.

Setting the variable back to `=1` and restarting produced a process that
**invokes `neuronx-cc` and took ~36 minutes to finish compiling** (restart
20:54:38, compiler idle 21:31). Three smoke requests issued during that window
each timed out at 600 s with `tpu_jax_requests_total` showing 0 success *and* 0
failed — still in flight, not failed. Once the compiler went idle, the first
request returned **`Paris.` in 175 s**, correct. The original `=1` process, by
contrast, served in 77–84 s and left the NEFF cache at 3.6 MB, consistent with it
dispatching almost everything to the host and barely touching the compiler.

Two consequences:

- The 77–84 s figure describes a process that had already settled, not a cold
  `=1` start. A fair cold-start comparison between the two configurations has not
  been made.
- Cache reuse across a configuration change is not free. The cache is keyed on
  the HLO module and the compile flags, and both differ between the `=1` and `=0`
  paths — and `NEURON_CC_FLAGS` itself changed mid-session when the invalid
  `--cache_dir` was removed, so entries written earlier record a flag set that no
  longer matches. Any future build-once/serve-many plan needs to treat a flag
  change as a cache invalidation.

The smoke test on the restored `=1` path did complete, and it is correct — but
its 175 s is a post-compile first call, not comparable to the 77–84 s steady
state. Three numbers, three different warmth states, and only the `=0` column
above is a settled measurement.

## What this does NOT establish

- **Parity covers fp16/bf16-KV only.** The passing runs used the unquantized
  export with `--quant-mode fp16 --kv-cache-dtype bf16`. They say nothing
  directly about the `w4a16` + `int8` KV combination the server actually runs;
  that is precisely why it is now the suspect.
- **The HTTP path is not what parity exercises.** Parity drives
  `JaxGemmaEngine` in-process. The server adds FastAPI, the entrypoint's env
  vars, and `neuron_entrypoint.py`. The 7/31 report already ruled out the
  threadpool as a cause, but the two paths are not identical.
- **Four prompts, 24 tokens, batch 1.** Greedy only. Longer contexts that wrap
  the 512-slot ring, batching, and bucket ladders past 128 tokens are all
  unexercised.
- **No claim about which op miscomputes**, only about which *feature* carries it.
- **Single prompt, batch 1, one bucket, 5 tokens.** The tok/s figures describe
  this configuration and must not be quoted as engine capability, per the
  measurement rule in `CLAUDE.md`.

## State at session end

The instance was stopped (not terminated) by a scheduled `shutdown` at 21:55 UTC,
a 4-hour cost cap set on the host itself so it would hold regardless of the
controlling session. A `neuronx-cc` compile for the restored `=1` path was still
running when it stopped and is lost; everything else persists:

| | |
|---|---|
| instance | `i-0022c706bb9e89263`, `inf2.xlarge`, `us-east-2c`, **stopped** |
| cache volume | `vol-00eec2eec5a0d172c`, 200 GiB, retained |
| NEFF cache | ~103 MB, 145 `MODULE_*` dirs |
| checkpoint | 7.8 GB on the cache volume, no re-download |
| phase markers | `os-packages`, `python-deps`, `neuron-probe` — no re-bootstrap |

A start (not a fresh launch) resumes in minutes. Note the config on the host is
`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1` and
`NEURON_CC_FLAGS=--model-type=transformer`, both edited in
`/etc/gemma4-inf2.env` directly; `user_data.sh` now carries the equivalent
changes, so a fresh launch and this host should agree.

Spot capacity for inf2 was unavailable all session — placement scores pinned at
3/10 across us-east-1, us-east-2, and us-west-2 for both `inf2.xlarge` and
`inf2.8xlarge`, with four `InsufficientInstanceCapacity` failures across three
AZs. On-demand at $0.7582/hr was the only way to get hardware; the run cost ~$3.

## What to do next

1. **Run `jax_neuron/parity.py --subject-platform neuron` with the workaround
   off.** The fault now emits tokens, so the greedy-token comparison has
   something to diff — previously it had an empty completion and could only
   report that everything diverged. Compute the reference off-Inf2
   (`--save-reference ref.json --skip-subject` on any torch box) and run the
   subject on the device.
2. **A/B `--window-kv on|off` under the same conditions.** The 7/31 parity run
   passed on Neuron *unwindowed*; the serving path is windowed, and no run has
   ever compared them on device.
3. **Bisect the logits/sampler path first.** A repeated constant token is more
   consistent with a broken argmax or a logits tensor that is not what the
   sampler thinks it is than with an attention defect.

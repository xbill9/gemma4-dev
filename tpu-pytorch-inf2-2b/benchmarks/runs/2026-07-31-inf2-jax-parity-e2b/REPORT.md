# Does the pure-JAX Gemma 4 engine compute the right thing on Inferentia2?

**Run:** 2026-07-31
**Instance:** `inf2.8xlarge` spot, `us-east-2a`, `i-02c848270f446aede`
**AMI:** `ami-05235a8b272ee7f7e` — Deep Learning Base Neuron AMI (Ubuntu 22.04) 20260511, Neuron SDK 2.29.1
**Stack:** jax 0.6.2 / jaxlib 0.6.2, jax-neuronx 0.6.2.1.0.6446, libneuronxla 2.2.17544.0, neuronx-cc 2.24.8799.0
**Checkpoint:** `google/gemma-4-E2B-it-qat-q4_0-unquantized` (bf16 weights, dense `fp16` path)
**Harness:** `jax_neuron/parity.py`
**Reference:** `transformers` PyTorch, float32, host CPU

## Result: greedy token parity on Neuron

| Subject platform | window_kv | Prompts matched |
|---|---|---|
| CPU (JAX) | off | **4/4** |
| CPU (JAX) | on | **4/4** |
| Neuron | off | **4/4** |
| Neuron | on | not run — spot reclaim, see below |

Token-identical, not merely similar: every generated id matches the reference,
and both sides stop at the same token (5, 17, 24, 24 tokens for the four
prompts — the first two hit EOS early and agree on where).

This closes milestone 3 of `jax_neuron/README.md`. Milestone 2 established that
`neuronx-cc` *accepts* the engine's graphs; this establishes that the accepted
graphs *compute the right thing* on the real checkpoint through the real
serving class.

## What was actually compared

The subject is `JaxGemmaEngine` — the same class `jax_openai_server.py` loads,
with the same loader, cached decode step, and sampler. Not a hand-built model:
per `deployments/aws-inf2/README.md`, parity against a freshly constructed model
does not exercise the code path a server process uses.

The reference is Hugging Face `transformers` in PyTorch, float32, on the host
CPU — a different implementation of the same math, not a rearrangement of this
one. Greedy on both sides (`temperature=0` takes the `argmax` branch, which is
deterministic and identical on every backend), so a divergence is the model and
never the sampler.

Both sides are handed **byte-identical token ids**. The harness prepends `<bos>`
itself, which makes the engine's own prepend a no-op, so the two cannot silently
answer different questions. `tests/test_parity_harness.py` pins that equivalence
against the rule in `JaxGemmaEngine.generate_stream`.

The checkpoint is the QAT export with weights stored *unquantized*, so both
sides read the same bf16 numbers and any divergence is a porting bug rather than
a quantization difference. Evaluating the compressed W4A16 export is milestone 5
and nothing here speaks to it.

### Stage 0 mattered

```
'hello world' -> [23391, 1902] -> 'hello world'   bos=2 eos=1
```

The tokenizer check runs before anything loads. In the sibling port every single
garbage-output incident was innocent silicon, and a `tokenizer.json` that maps
every prompt to `<unk>` is the most common cause. It cost two seconds here.

## The bug this found: windowed KV attends to prompt padding

The first CPU run failed **0/4**. That is the entire point of running the CPU
oracle first — the accelerator was never a suspect.

The subject stayed fluent and drifted:

| Prompt | Reference | Subject |
|---|---|---|
| `The capital of France is` | ` Paris.` | ` Paris.` then `<turn\|>` forever |
| `Q: What is 17 * 23?\nA:` | ` 17 * 23 = 391` | ` 391` |
| `def fibonacci(n):` | `if n <= 1: return n ... fibonacci(n-` | `if n <= 0: return 0 / return 1 / return 1` |

Prefill was right — token 0 agreed everywhere — and decode drifted a variable
number of tokens in. An A/B with `--window-kv off` passed 4/4 with everything
else identical, isolating it to the windowed path in one run.

**Cause.** `make_ring_decode_mask` marked every ring slot below
`positions_written` as attendable, on the assumption that the ring is densely
packed with real tokens. It is not. Serving pads every prompt up to a bucket
before prefill, so slots between the true prompt length and the bucket edge hold
zeroed K/V. A sliding layer attended to all of them — 58 padding slots out of 64
for a short prompt. The global layers were never affected because their mask is
built from `valid`.

The fix takes the mask from `valid`: ring slot `j` holds the most recent position
congruent to `j mod ring_len`, and is attendable only if that position is a real
token.

**Why no test caught it.** Every case in `tests/test_windowed_kv.py` passed
`valid_prompt = ones(...)`. Padding and windowing were never exercised together,
so a bug that only exists at their intersection could not be seen. The suite now
runs padded prompts through both paths, wrapped and unwrapped.

This is the failure mode `CLAUDE.md` names: code that runs, reports success, and
computes the wrong thing. Nothing raised, nothing warned, and the output read as
plausible English.

### A second, smaller defect on the way in

The same first run crashed before it could diverge:

```
add got incompatible shapes for broadcasting: (1, 8, 1, 88), (1, 1, 1, 512)
```

`init_kv_cache` sizes a windowed sliding layer as `min(max_seq_len,
sliding_window)`, so a 64-token bucket with 24 new tokens gets an 88-slot ring —
narrower than E2B's 512 window. The decode step built its mask from
`config.sliding_window` regardless. The width now comes from the buffer the
scores are computed against. Same root confusion as above: the config is not the
authority on the buffer's shape, the buffer is.

Both bugs are in the shared engine and affect **TPU equally**. Neither is
Neuron-specific; porting to a second platform is just what surfaced them.

## Four defects in the deployment scaffold

`deployments/aws-inf2/user_data.sh` had never completed a launch. Each of these
aborted the bootstrap under `set -euo pipefail`, leaving a host that boots, looks
healthy, and serves nothing:

| Defect | Symptom |
|---|---|
| `install -d /opt/gemma4` left the parent root-owned | `Permission denied: '/opt/gemma4/venv'` — the venv is created as `ubuntu` |
| `--index-url` instead of `--extra-index-url` | The Neuron repo has no `jax` wheel, so `jax<=0.6.2,>=0.4.30` resolves against nothing: `No matching distribution found for jax` |
| `libneuronxla` left unpinned | `jax-neuronx==0.6.2.1.0.*` requires only `libneuronxla>=2.2.12677.0`, so pip took 3.0.3854.0, which wants an NRT 3.0 runtime this AMI line does not ship: `undefined symbol: nrta_event_register_xu_completion, version NRT_3.0.0` |
| systemd unit had no `PATH` | `libneuronxla` shells out to `neuronx-cc` by bare name; the unit's default PATH excludes the venv, so the first compile dies as `XlaRuntimeError: UNKNOWN: sh: 1: neuronx-cc: not found` |

The third is the interesting one: the install *succeeds* and fails much later at
PJRT load, and the pin that looks like it controls the SDK line does not control
the component that actually binds to the runtime.

## Runtime probe reproduced on this stack

`jax_neuron/probe.py` passes on the SDK-2.29.1 AMI with the SDK-2.28 wheel and
`libneuronxla` pinned to the 2.2 line:

```json
{"device_count": 2, "platform": "neuron", "finite": true,
 "jax_version": "0.6.2", "first_call_s": 2.53, "warm_call_s": 0.001}
```

`deployments/aws-inf2/README.md` says to pair the pin with an SDK-2.28 DLAMI. No
such AMI is offered in us-east-2 — the oldest images that state a version are
2.29.0. The combination above is what was measured to work; move the pin and the
AMI together, and re-run the probe when you do.

## What this does NOT establish

- **No performance claim.** The timings in the log (125 s for the first prompt
  including compile, ~8 s for 17–24 tokens after) are a correctness run on a
  cold cache at batch 1. They are not a benchmark and must not be quoted as one.
  The 19.04 GB/token DMA estimate from the compile probe is still unconfirmed on
  hardware; that needs `neuron-monitor`, not this.
- **Buffer donation is still unverified as an optimization.** `donate_cache=True`
  was active and output was correct, which shows donation does not break
  correctness on Neuron. Whether the plugin actually donates or silently copies
  is a separate measurement.
- **W4A16 is untested on device.** This ran the dense `fp16` path against the
  unquantized export, deliberately. Milestone 5 stands.
- **Four short prompts, 24 tokens, batch 1, one bucket.** Longer contexts,
  batching, int8 KV, and the HTTP path are all unexercised here.
- **No windowed-KV result on Neuron.** The CPU A/B is what isolated and then
  confirmed the fix, and the unwindowed Neuron run above is token-exact. The
  windowed Neuron run was started and did not finish: spot reclaimed the
  instance at 16:04:57 GMT, about a minute after the unwindowed result landed.
  It matters because the fix introduces a `take_along_axis` gather into a
  Neuron-compiled graph, and nothing here shows `neuronx-cc` accepts it. That
  question does not need hardware — `jax_neuron/compile_probe.py` answers it on
  any x86-64 Linux box — and it should be answered before the windowed path is
  used on Inf2.

## Reproduce

```bash
# once, on a box with torch and the checkpoint
python3 jax_neuron/parity.py --local-dir "$CKPT" --max-new-tokens 24 \
  --skip-subject --save-reference ref.json

# subject, on the Inf2 host
python3 jax_neuron/parity.py --local-dir "$CKPT" --max-new-tokens 24 \
  --quant-mode fp16 --reference ref.json --subject-platform neuron
```

Exit status is non-zero if any prompt diverges. `--subject-platform cpu` runs the
same comparison without touching a NeuronCore, and is the first thing to try
when output looks wrong.

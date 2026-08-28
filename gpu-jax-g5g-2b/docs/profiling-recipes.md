# Profiling recipes for `gpu-jax-g5g-2b`

`profile_decode.py`'s docstring has pointed here since it was written; the file did not exist
until 2026-08-26. Everything below is measured on a T4G.

## The two tools, and which question each answers

| Tool | Needs | Answers |
| --- | --- | --- |
| `profile_decode.py` | nothing extra | per-kernel GPU time for one decode step |
| `profile_prefill.py` | nothing extra | prefill peak memory, from `compiled.memory_analysis()` |
| **xprof** | `requirements-profiling.txt` | TensorCore use, allocator fragmentation, roofline constants |

Both scripts parse artifacts `jax.profiler.trace()` already writes, so **try them before
installing anything.** xprof adds a viewer and the memory/kernel rollups, not a new
measurement path.

## Running the decode profiler

It needs the GPU to itself, and it is not part of the deploy payload — ship it first.

```bash
systemctl stop jax-g5g
set -a; . /opt/jax-g5g/env; set +a
cd /opt/jax-g5g/app
PYTHONPATH=/opt/jax-g5g/app python3.14 profile_decode.py \
    --ple-bits 4 --int8-lm-head --steps 20 --top 12
```

`--int8-lm-head` was added 2026-08-26. Before that the flag did not exist, so the one config
worth profiling most could not be profiled at all.

## Installing xprof

**On demand only.** It is deliberately excluded from `_SERVING_REQUIREMENTS` and
`requirements-serving.txt` — a serving image should not carry a profiler, and xprof pulls
`gcsfs` + `google-cloud-storage` behind a 39 MB wheel.

```bash
python3.14 -m pip install --break-system-packages -r <path>/requirements-profiling.txt
```

**That file is not on the instance unless you put it there.** It is deliberately excluded from
the deploy payload — a serving image should not carry a profiler — so nothing ships it, and
this recipe used to name `/opt/jax-g5g/requirements-profiling.txt`, a path that has never
existed. xprof then "installed" with `Could not open requirements file` and the extraction
died on `ModuleNotFoundError: No module named 'xprof'`, both in logs nobody read.
`tune_loop.py --xprof` ships it alongside `profile_decode.py`; by hand, copy it first.

`--break-system-packages` is required from the Ubuntu 24.04 base onward — the system
interpreter is marked externally-managed (PEP 668).

**aarch64 note:** the wheel is `manylinux_2_35_aarch64` and the Ubuntu 22.04 DLAMI base is
glibc 2.35 — *exactly* at the floor. It installs, but an older base would silently lose it.

## Getting data out of xprof without the UI

`xprof --logdir … --port 6006` serves a UI, and this rig opens no inbound port for it (and
must not). To extract the same data programmatically:

```python
from xprof.convert import raw_to_tool_data as R
xp = glob.glob("/tmp/jaxtrace/plugins/profile/*/*.xplane.pb")
print(R.xspace_to_tool_names(xp))          # what this trace supports
data, _ = R.xspace_to_tool_data(xp, "kernel_stats^", {})
```

Tools that work here: `kernel_stats`, `memory_profile`, `roofline_model`, `op_profile`,
`hlo_stats`, `overview_page`, `trace_viewer@`.

**`memory_viewer` returns nothing** — it needs the HLO proto, which the trace does not carry
(`Can not load hlo proto from options`). Use `profile_prefill.py` for allocation questions
instead; it reads the optimized HLO directly.

## Traps

- **`tensorboard` is not needed, and will not render these on its own.** xprof is the
  successor to `tensorboard-plugin-profile` and serves its own UI; its requirements list
  contains no tensorboard, and a `tensorboard` install on this box has `Required-by:` empty.
  Installing plain `tensorboard` and pointing it at the logdir gets you a TensorBoard with no
  profile plugin. Use `xprof --logdir … --port 6006`.

### Viewing a profile without opening a port

This rig opens **no inbound port** for a UI and must not — the security group allows 8000 for
the endpoint and nothing else, and there is no inbound SSH rule or key at all. Two ways to
look at a trace anyway, in preference order:

1. **Locally, off the returned trace.** `tune_loop.py --xprof` brings
   `jaxtrace/plugins/profile/<run>/*.xplane.pb` back inside the run directory, so
   `pip install xprof && xprof --logdir benchmarks/runs/<run>/jaxtrace --port 6006` serves the
   UI on your own machine with no instance involved. This works after the instance is gone,
   which matters here: G5g is spot and reclamation has ranged from 21 minutes to 19 hours.
2. **On the box, through SSM port forwarding** — no inbound rule required, because SSM tunnels
   over the agent's outbound connection:

   ```bash
   aws ssm start-session --target <instance-id> \
       --document-name AWS-StartPortForwardingSession \
       --parameters '{"portNumber":["6006"],"localPortNumber":["6006"]}'
   ```

   Start `xprof --logdir /tmp/jaxtrace --port 6006` on the instance first. **Do not add an
   inbound rule for 6006** — the tunnel is what makes that unnecessary.
- **The `Program` row dominates roofline output.** The trace spans the whole process
  including load, so per-op roofline rows are swamped. The *device constants* in that tool
  are the useful part: peak HBM 298.1 GiB/s, peak FLOP 65,126 GFLOP/s, ridge 203.5 FLOP/byte.
- **`hlo_stats` and `op_profile` come back thin** for a decode-only trace — a couple of ops.
  The kernel table from `profile_decode.py` is the richer source.
- **Read `Kernel uses TensorCore`, not the kernel name.** Names like `gemvx::kernel<…float…>`
  already imply fp32, but the explicit column is what settles it.
- **Warm up before profiling.** Cold measures several times slower and XLA compiles per
  shape.

## What profiling has actually found here

| Date | Finding |
| --- | --- |
| 2026-08-24 | `wrapped_convert` is 55% of decode — bf16 weights against a float16 compute dtype |
| 2026-08-25 | **0.0% of kernel time uses TensorCores**; 54.4% conversion + 32.6% fp32 gemvx = 87% dtype tax |
| 2026-08-25 | Allocator fragmentation **0.661** at peak with 2.9 GiB free — the contiguous-block OOMs |
| 2026-08-26 | `int8_lm_head` does **not** remove the conversion: it dequantizes the whole int8 table to fp16 every step |

The through-line: **this rig is not compute-bound or bandwidth-bound, it is dtype-bound.**
Anything that does not remove a conversion will not move the number much.

# Turing on a T4 you already own: the third delivery of the same clamp

**Status: this rig has run nothing.** Every measurement cited below was taken on
`gpu-vllm-g5g-2b` (T4G, Graviton2) on 2026-08-12 or on `gpu-vllm-g4dn-2b` (T4, x86_64) on
2026-08-30, or is arithmetic. What is new here is not a measurement about the chip — the chip
is identical — it is **where the patch goes and what that costs**.

Read `@HARDWARE.md` "T4G" for the chip and `@MODELS.md` for Gemma 4's head geometry. Neither is
re-derived here.

## The problem, in one paragraph

Gemma 4's attention layers have different head dims — sliding **256**, full **512**. Only FA4
and TRITON_ATTN support heterogeneous head dims, FA4 is unavailable, so vLLM **forces** Triton
and says so in the log. Triton's unified-attention kernel at `head_size=512` wants **98,304 B**
of shared memory per block. Turing allows **65,536**, and only if the kernel opts into the
dynamic attribute — the default static limit is 49,152. `65,536 < 98,304` is arithmetic, and the
same silicon has already produced the failure on two siblings:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536
```

**The backend is not the knob.** MEASURED on the G5g sibling: `VLLM_ATTENTION_BACKEND` is not a
recognized variable in vLLM v0.27 (`Unknown vLLM environment variable detected`) and vLLM forces
Triton for this model regardless. The tile size inside the forced kernel is the knob, and
`patch_triton_turing.py` is what changes it.

## What is new: three rigs, three deliveries, and this one is the cheapest and least contained

| Rig | Host | SM 7.5 kernels published? | Delivery | Cost |
| --- | --- | :---: | --- | --- |
| `gpu-vllm-g5g-2b` | aarch64 | **no** | from-source vLLM build; the patch rides along | ~67 min |
| `gpu-vllm-g4dn-2b` | x86_64 | yes (amd64 image) | derived docker image, one `COPY` | seconds |
| **`gpu-vllm-t4-2b`** | **x86_64** | **unverified for the WHEEL** | **edit the installed module in site-packages** | **milliseconds** |

The two EC2 rigs patch a file inside an image they built and can throw away. **This rig edits
the host's own vLLM**, and three consequences follow that do not exist on either sibling:

- **The blast radius is the machine, not the rig.** `/opt1/pyuser/lib/python3.13/site-packages`
  is shared by every process and every rig on this host. A bad patch is not a rebuild away.
- **`pip install -U vllm` silently reverts it.** There is no tag to pin, so nothing records that
  the clamp was ever applied. `verify_turing_patch` after *any* vLLM change, not just after a
  deliberate upgrade.
- **The idempotency check had to change.** The EC2 scripts key "already patched" on a marker
  carrying their own rig name, which is safe when one rig owns one image. Here a sibling's script
  could have patched the same file first, so this copy keys on a shared `SENTINEL` and records
  provenance in `MARKER`. Without that split the file gets **two** clamps, the second halving the
  tiles again to 16 — it serves, slower, and reports success. `tests/test_server.py` pins it.

## The question this rig cannot yet answer

**Does the published `torch`/`vllm` wheel pair carry SM 7.5 kernels?**

For the docker image this is settled and published — read from the image config of
`vllm/vllm-openai`, one manifest list, two platforms:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | no |

**A wheel is a different artifact and nobody in this tree has read its arch list.** That is the
rig's one genuinely open question, and until it is answered the rig cannot claim it will serve.

**The install currently on this host cannot answer it either**, which is worth stating because it
looks like it should. `/opt1/pyuser` holds a CUDA vLLM against **`torch 2.11.0+cpu`**:

```
$ PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
2.11.0+cpu None []

$ PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13 -c "import vllm"
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

**An empty arch list from a CPU wheel is not evidence that SM 7.5 is missing** — a CPU build
reports `[]` for every architecture. Fix torch first, then re-run `verify_gpu_arch`. Reading
"arch list: []" as an answer about Turing would send an operator to a 67-minute from-source
build they may not need, on 2 vCPU.

**And `sm_75` is the string to match, not `7.5`.** torch prints `['sm_75', 'sm_80', ...]`. The
first version of `verify_gpu_arch` here matched on `"7.5"` and therefore reported a *working*
Turing build as unsupported — the worst direction for this check to fail in. Pinned by
`test_a_working_cuda_torch_reports_half_the_question_done`.

## What the clamp does at Gemma 4's shapes

fp16, `BLOCK_M=16`, budget 60000 — carried from the G5g sibling's verified run:

| layer | head | path | tile in → out | bytes in → out |
| --- | ---: | --- | --- | --- |
| `sliding_attention` | 256 | prefill / decode | 32 → 32 | 40,960 → unchanged |
| `full_attention` | 512 | decode | 16 → 16 | 49,152 → unchanged |
| **`full_attention`** | **512** | **prefill** | **32 → 16** | **81,920 → 49,152** |

Narrow by design: only the 512-wide global prefill path moves, and it lands under 65,536.

**`TURING_SMEM_BUDGET` is 60000, not 65536, and that is not a rounding preference.** The tile
arithmetic does not account for the kernel's accumulators, so budgeting the hard limit still
overflows.

## Why the patch refuses rather than guessing

A patch that silently matches nothing is worse than one that fails: it leaves an unpatched
module behind a patched-looking install, and the failure surfaces ten minutes later at engine
start, attributed to the wrong thing. The script exits 2 with the surrounding source attached
when an anchor or identifier is missing.

**The subtlest check is the insertion point.** Upstream copies the tile constants into
`tile_size` and `launch_num_stages` into `launch_kwargs` *before* the launch, so a clamp at the
launch site would rewrite three variables nothing reads afterwards — marker present,
verification passing, kernel still asking for 98,304 bytes. The insertion point is derived from
the code (after the last tile assignment, before the first read), not from a launch-site
pattern. The stage variable is found with `ast` rather than a regex, because
`launch_kwargs["num_stages"] = launch_num_stages` reads as an assignment to `num_stages` to a
line-oriented matcher, and picking it would bind a local nothing reads: **half the fix, applied
silently, reported as success.**

## Where this host differs from both siblings, and it is not the GPU

The T4 here is the same TU104 at the same clocks: 15360 MiB, 1590 MHz SM, 5001 MHz memory,
compute capability 7.5, 70 W. The differences are all host-side:

- **Three filesystems.** `/` has 4.23 GB free, `/tmp` has 3.88 GB, `/opt1` has 249.20 GB — and
  `~/.cache` is a **symlink** into `/opt1`. So the checkpoint has room while a default
  `pip install` does not, which is the opposite of what a single `df /` suggests. The first draft
  of this rig read `df /`, concluded "disk binds", and was wrong in both directions at once.
- **Two interpreters, and the default is the wrong one.** `python3` is pyenv 3.12.13 with
  site-packages on the full root disk; `/usr/bin/python3.13` with `PYTHONUSERBASE=/opt1/pyuser`
  has the room and the packages. Every probe in `server.py` goes through `PYTHON_BIN`.
- **No swap.** This *inverts* the `local-vllm-cpu-2b` warning rather than repeating it: there,
  exceeding host RAM is accepted and paid for in 15.4 GB of swap, so thrashing is
  indistinguishable from loading. Here it is a prompt OOM kill with a `dmesg` line.
- **2 vCPU against the sibling's 4.** Not expected to cap throughput — the g4dn run reasons that
  decode is GPU-bandwidth-bound at these rates — but engine startup is compilation-heavy: 82.39 s
  of a 150.90 s engine init on 4 vCPU. ARITHMETIC: budget roughly double.
- **7.80 GB of host RAM against the sibling's 16 GiB.** vLLM mmaps the safetensors and copies
  shard by shard so peak RSS is well below the 10.25 GB checkpoint, and this is recorded as an
  **UNMEASURED risk rather than a verdict**. It is the most likely thing to surprise the first run.

## What is still open

- **Nothing here has been run.** No install, no serve, no token.
- **The wheel's arch list**, above. It decides the rig.
- **Whether 7.80 GB of host RAM stages a 10.25 GB checkpoint.** Unmeasured anywhere in this tree.
- **The clamp is not upstream.** Until it lands, every vLLM upgrade re-runs this patch — and here,
  unlike on the EC2 siblings, nothing tags the image to remind you.

## Sources

- Turing measurements: `gpu-vllm-g5g-2b/docs/turing-aarch64-gap.md`,
  `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-12-first-serve-g5g/REPORT.md`
- The same configuration MEASURED on a real T4:
  `gpu-vllm-g4dn-2b/benchmarks/runs/2026-08-30-first-serve-g4dn/`
- Shared-memory limits and the 48/64 KiB distinction: `@HARDWARE.md`, "T4G"
- Head-dim split and the KV open discrepancy: `@MODELS.md`
- The delivery comparison: `gpu-vllm-g4dn-2b/docs/turing-shared-memory.md`

# gpu-vllm-t4-2b

vLLM serving **`google/gemma-4-E2B-it`** on **one NVIDIA Tesla T4 already attached to
this Compute Engine VM**.

**Status 2026-09-17: scaffolded. Nothing installed, nothing served, nothing
measured.** `benchmarks/runs/` is empty on purpose.

| | |
| :--- | :--- |
| Platform | GCE `n1-standard-2`, `us-west2-b` — the GPU is attached, nothing provisions it |
| GPU | Tesla T4, TU104, compute capability **7.5**, 15360 MiB, 5001 MHz memory clock |
| Runtime | vLLM, CUDA, **pip-installed** (no docker: no `nvidia` runtime on this host) |
| Model | `google/gemma-4-E2B-it`, **float16** (Turing has no bf16 datapath) |
| Endpoint | `http://127.0.0.1:8000` — known before start, not discovered |
| A/B twin | **`gpu-vllm-g4dn-2b`** — same T4, same clocks, same flags, on EC2 |

## Start here

```bash
make capacity        # free arithmetic over three filesystems; run this first
make arch            # does the installed build actually have SM 7.5 kernels?
make patch           # clamp Triton's tiles to Turing's 64 KiB
make verify-patch    # is the clamp in the module vLLM imports?
make serve           # refuses unless all of the above are satisfied
```

The order is not decorative: each step's failure is cheaper than the next one's.

## What blocks it today

**The vLLM already installed here cannot use the T4.** `/opt1/pyuser` (the user base
for `/usr/bin/python3.13`) holds a CUDA vLLM sitting on **`torch 2.11.0+cpu`**:

```
$ PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13 -c "import vllm"
ImportError: libcudart.so.13: cannot open shared object file
```

Two packages present, neither usable. Fix torch for that interpreter — keeping the
`PYTHONUSERBASE` / `PIP_CACHE_DIR` / `TMPDIR` redirections this host needs — then
re-run `make arch`. `make capacity` prints the command.

**VRAM is not the problem**, and it is worth saying so because it is the reflex
answer for a 16 GB card. The g4dn sibling MEASURED **9.8 GiB of weights inside 15.0
GiB of usable HBM** on this same part, with a **329,579-token** KV pool left over.

## Three filesystems, and one `df` lies about all of them

MEASURED on this host 2026-09-17:

| path | free | holds |
| :--- | ---: | :--- |
| `/` | **4.23 GB** | pyenv 3.12's site-packages |
| `/tmp` | **3.88 GB** | pip's unpack directory |
| `/opt1` | **249.20 GB** | `~/.cache` (a symlink), `/opt1/pyuser`, `/opt1/tmp` |

The first version of this rig read `df /`, saw 4.23 GB and concluded "disk binds".
That was wrong in both directions: the 10.25 GB checkpoint has 249 GB of room
because `~/.cache` points at the big volume, while a *default* `pip install` fails
on `/tmp` regardless. `check_host_capacity` measures each target path separately.

## The one open question

**Does the published wheel pair carry SM 7.5 kernels?** For the docker image the
answer is published — the `linux/amd64` manifest lists `7.5 8.0 8.6 8.9 9.0 10.0
12.0` while `linux/arm64` of the same tag omits 7.5 — but a **wheel is a different
artifact** and nobody in this tree has read its arch list. The install currently
here cannot answer it either: a CPU torch reports `[]` for every architecture, which
is not evidence about Turing.

`verify_gpu_arch` asks the interpreter, and `start_vllm_server` refuses until the
answer is positive. Note that the string to match is **`sm_75`**, not `7.5` — the
first version of that check matched `"7.5"` and reported a *working* Turing build as
unsupported.

## Why this rig exists

1. **It isolates the control plane.** Against `gpu-vllm-g4dn-2b` it differs in slots
   1 and 3 only, and slot 3 differs only in spelling (`g4dn` is the EC2 family whose
   GPU is this same T4). Same silicon, same clocks, same engine, same flags — one
   rig provisions its hardware, the other simply has it.
2. **It is the third and cheapest delivery of the same Turing clamp.**
   `gpu-vllm-g5g-2b` needs a ~67-minute from-source build; `gpu-vllm-g4dn-2b` builds
   a derived docker image; here the patch edits one file in site-packages. It is
   also the **least contained** — that file is the host's own vLLM, shared with
   everything else on the box, and `pip install -U vllm` silently reverts it.

`docs/turing-on-a-gce-t4.md` has the full write-up. `@HARDWARE.md` ("T4G") owns the
chip, `@MODELS.md` owns the checkpoint, `@NAMING.md` owns the name.

## Not to be renamed

- **Not `local-vllm-t4-2b`.** Nothing provisions the GPU, which is `local`'s test,
  but `@NAMING.md` decides the slot by who owns the machine: "SSH into a cloud VM you
  provisioned and it keeps that VM's platform value." This is a GCE VM.
- **Not `gce-vllm-t4-2b`.** `gce` is reserved for TPU-on-Compute-Engine; "a GPU rig
  is `gpu` wherever it runs."
- **Not `gpu-vllm-g4dn-2b`-style.** The instance-family spelling is an EC2 rule.

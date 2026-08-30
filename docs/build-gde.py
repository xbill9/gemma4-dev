#!/usr/bin/env python3
"""Build the Medium import page for the pure-JAX Turing-vs-Ada article.

Emits a CONTENT-ADDRESSED `gde-<sha10>.html` into `docs/`, because Medium's
importer caches by URL and ignores the query string. Tables are rendered to PNG
because the importer strips markdown tables entirely.

Table rendering is reused from `gpu-jax-g6-2b/make-medium.py` rather than
reimplemented, so these tables match every other rendered table in the tree.
That is a build-time import of a tool, not a rig importing a sibling's engine --
the rigs stay siblings.

Two importer rules are baked in here and are invisible in the rendered output:
no `<link rel="canonical">` (the importer resolves it and serves that URL's
cached copy), and no links inside `<figcaption>` (Medium silently drops the
whole figure). See `docs/README.md`.

The cover is generated separately by `make-cover-gde.py`; this script only
references it, and fails loudly if it is missing.
"""

import hashlib
import importlib.util
import pathlib
import sys

DOCS = pathlib.Path(__file__).resolve().parent
ROOT = DOCS.parent
IMG = DOCS / "img"

# The cover, produced by make-cover-gde.py. Content-addressed, so this constant
# changes whenever the cover is re-rendered.
COVER = "cover-medium-gde-4556fbaf.jpg"
COVER_ALT = (
    "Two stat tiles on a dark ground: the T4G spends 87% of decode on "
    "dtype conversion at 12.9 tok/s, the L4 spends 0.0% at 48.4 tok/s."
)


def load_renderer():
    """Borrow parse_table/render_table from the g6 rig's make-medium.py."""
    path = ROOT / "gpu-jax-g6-2b" / "make-medium.py"
    if not path.exists():
        sys.exit(f"missing table renderer: {path}")
    spec = importlib.util.spec_from_file_location("_mkmedium", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MM = load_renderer()


# --------------------------------------------------------------------------
# tables -- every figure sourced from the two summary.json files named below
# --------------------------------------------------------------------------

SPEC = """
| | G5g | G6 |
|---|---|---|
| Chip | NVIDIA **T4G** — Turing, SM 7.5, 15,360 MiB | NVIDIA **L4** — Ada, SM 8.9, 23,034 MiB |
| Host | `g5g.2xlarge` spot — Graviton2, **aarch64** | `g6.2xlarge` spot — **x86_64**, `us-east-1d` |
| Checkpoint | `google/gemma-4-E2B-it`, dense reference | `google/gemma-4-E2B-it`, dense reference |
| Compute dtype | `float16` (device-chosen) | `bfloat16` (device-chosen) |
| Stack | jax 0.11.1, CUDA from pip | jax 0.11.1, Python 3.14 |
| Run cited | `2026-08-28-full-run-cached-g5g` | `2026-08-28-first-serve-g6` |
"""

NUMBERS = """
| | T4G (Turing) | L4 (Ada) |
|---|---|---|
| Decode, gauge — 41 / 521 / 2,057 in | 12.9 / 13.0 / 12.9 tok/s | 48.5 / 48.4 / 48.3 tok/s |
| End-to-end, same cells | 12.43 / 11.28 / 8.22 tok/s | 46.23 / 42.87 / 34.57 tok/s |
| Weights resident | 6.155 GB | 6.155 GB |
| Total kernel time, 20 decode steps | 1,466.0 ms | 362.8 ms |
"""

PROFILE = """
| | T4G (SM 7.5) | L4 (SM 8.9) |
|---|---|---|
| dtype conversion | 54.1% | **0.0%** |
| fp32 `gemvx` | 32.8% | **absent** |
| Tensor Core | 0.0% | **0.0%** |
| Total kernel time | 1,466.0 ms | **362.8 ms** |
| Decode, gauge | 12.9 tok/s | **48.4 tok/s** |
"""


def render(md: str, stem: str) -> str:
    """Render a markdown table to a content-addressed PNG; return its filename."""
    hdr, body = MM.parse_table([ln for ln in md.strip().splitlines() if ln.strip()])
    tmp = IMG / f".{stem}.tmp.png"
    MM.render_table(hdr, body, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()[:8]
    out = IMG / f"{stem}-{digest}.png"
    tmp.replace(out)
    return out.name


def main():
    if not (IMG / COVER).exists():
        sys.exit(f"missing cover: {IMG / COVER} -- run make-cover-gde.py first")

    t_spec = render(SPEC, "gde-table-spec")
    t_num = render(NUMBERS, "gde-table-numbers")
    t_prof = render(PROFILE, "gde-table-profile")

    title = "Gemma 4 in Pure JAX: What Changes Between Turing and Ada, and What Doesn't"

    body = f"""<h1>{title}</h1>
<figure><img src="img/{COVER}" alt="{COVER_ALT}"></figure>
<p>This article is about running a hand-written <strong>Gemma 4</strong> port in
<strong>pure JAX</strong> on two NVIDIA GPUs a generation apart, and about the two places
the abstraction leaks. One of them costs 87% of decode, and nothing in the logs is red.</p>
<p>The code is here:</p>
<p><a href="https://github.com/xbill9/gemma4-dev">github.com/xbill9/gemma4-dev</a></p>
<p><strong>What this was measured on.</strong> One port, one build, one checkpoint, two cards:</p>
<figure><img src="img/{t_spec}" alt="Specification of the two rigs under test"><figcaption>The two rigs, and the run each number below comes from</figcaption></figure>
<p>Build id <code>51bc52c9e2e9</code> on both, config <code>ple4 + int8_lm_head</code>, and
<code>tpu_jax_weight_bytes</code> = 6,155,450,950 — the same integer on both cards. Sweeps are
64 output tokens, concurrency 1, 3 repeats per cell, median. <strong>"Decode, gauge"</strong> is
the engine's steady-state counter; <strong>"end-to-end"</strong> is wall time over the whole
request, prefill included.</p>
<h4>What is this project trying to Do?</h4>
<p>This project aims to serve one Gemma 4 checkpoint from one JAX port across every accelerator
I can rent, and to find out — by measurement, not by reading docs — which parts of "it's just
JAX" are true.</p>
<p>The port lives in <code>ports/gemma4/</code> and is driven by a generation loop behind an
OpenAI-compatible server. No PyTorch, no vLLM, no <code>torch_xla</code>. The same source runs
on both cards with nothing changed but a config file — which is the claim under test.</p>
<p>It mostly holds. Two things do not, and they are the interesting part.</p>
<h4>Gemma 4 E2B is not a stock transformer</h4>
<p>Any port has to carry four irregularities, and none of them are optional:</p>
<ol>
<li><strong>Two attention geometries.</strong> Sliding layers use <code>head_dim=256</code>,
global layers use <strong>512</strong>. Most inference stacks assume one head dimension per model.</li>
<li><strong>8:1 MQA</strong>, so the KV budget is nothing like the parameter count would suggest.</li>
<li><strong>A KV-share map</strong> that collapses <strong>35 layers onto 15 caches</strong>.</li>
<li><strong>A 512-slot sliding ring</strong>, plus per-layer embeddings (PLE) held in a
<strong>4.70 GB</strong> table that gets quantized to 4 bits on load.</li>
</ol>
<p>That first one is worth dwelling on, because it is what breaks other stacks. On the vLLM path,
the heterogeneous head dims force the Triton attention backend:</p>
<figure><img src="img/gde-code-01-4cdea716.png" alt="Gemma4 model has heterogeneous head dimensions"></figure>
<p>And on a Turing GPU that backend then asks for shared memory the hardware does not have:</p>
<figure><img src="img/gde-code-02-a55c14fc.png" alt="triton.runtime.errors.OutOfResources: out of resource: shared memory,"></figure>
<p><strong>JAX never enters that conversation.</strong> Attention is ordinary XLA rather than a
hand-tiled kernel, so there is no per-block shared-memory ceiling in the attention path at all.
This is the clearest win of the whole exercise: the irregular geometry that is a special case
everywhere else is just array shapes here.</p>
<h4>The dtype policy has to read the device, not the config</h4>
<p>This is the single most expensive lesson in the repo.</p>
<p><strong>A wrong compute dtype does not raise. It emulates.</strong> <code>bfloat16</code> on a
pre-Ampere GPU does not fail — XLA routes it through fp32 and you simply lose most of your decode
to conversion. Nothing in the logs is red.</p>
<p>So the port does not take the dtype from a config file. It reads the live compute capability
off the device and decides:</p>
<pre><code>COMPUTE_DTYPE = float16 if IS_PRE_AMPERE else bfloat16</code></pre>
<p>On the SM 8.9 Ada card that resolves to <code>bfloat16</code>. On the SM 7.5 Turing card,
<strong><code>float16</code></strong> — Turing's only real 16-bit datapath, since it has neither
bf16 nor fp8.</p>
<p>The first line the process emits states what it decided, so a misconfiguration is one grep away
rather than a mystery in the throughput:</p>
<figure><img src="img/gde-code-03-95f29414.png" alt="INFO ports.gemma4.jax_e_model: jax_e_model device policy: platform=gpu"></figure>
<p><code>pallas_interpret=False</code> matters just as much — it is the difference between serving
and silently running a simulator.</p>
<h4>Where the abstraction actually leaks: Pallas</h4>
<p>Here is the part that does not port, and it is not a bug — it is a real hardware difference
wearing a portable API.</p>
<p>The fused <strong>W4A16 kernel is written in Pallas</strong>, and it was tiled for a device with
16 MB of scratchpad per core. At this model's shapes the tiles want <strong>550 KiB – 1.1 MiB per
block</strong>.</p>
<p>On a GPU, Pallas lowers through Triton, and those tiles become <strong>shared memory</strong>.
Turing gives you 64 KiB per block. Ada raises the ceiling, but nowhere near a megabyte.</p>
<p>So the fast path <strong>cannot run on either card</strong>. The rig computes the requirement at
startup and refuses with the arithmetic attached, rather than dying as a cryptic
<code>OutOfResources</code> at the first token:</p>
<pre><code>check_w4a16_fits_scoped_memory()</code></pre>
<p>The practical consequence: both GPU rigs serve the <strong>dense reference checkpoint</strong>
at 16-bit. <strong>Pallas is portable as an API and not portable as a memory model.</strong> That
boundary is worth knowing before you plan a port around a fused kernel.</p>
<h4>The bug that returns <code>200 OK</code></h4>
<p>A padding-eviction bug in the KV ring cache cost a week, and it is the kind only Gemma 4's
geometry produces.</p>
<p>The invariant is: <strong>a cache index is an absolute real position, and padding never occupies
an index a real position uses.</strong> A port that right-pads into the 512-slot ring violates it,
and the failure mode is not a crash or a NaN. It is a <strong>token loop</strong> — a clean HTTP
<code>200</code>, <code>status: "success"</code>, and output like <code>The The The The</code>.</p>
<p>Nothing in the logs is red. Nothing in the metrics is red. The only thing that catches it is a
degeneracy check on the output itself, which the server now runs on every response.</p>
<p>The scariest bugs in this whole project all returned success.</p>
<h4>What did port, with no changes at all</h4>
<p>Enough that the exercise was worth it:</p>
<ul>
<li><strong>The model code.</strong> All four irregularities, both attention geometries, the
KV-share map, the ring — identical source on both cards.</li>
<li><strong>The compilation cache.</strong> XLA's persistent cache works the same way; on the T4G
rig it restores <strong>805 files / 12 MB in 6 seconds</strong> onto a fresh instance from a box
that had already been terminated.</li>
<li><strong>Bucketing and static shapes.</strong> <code>max_new_tokens</code> is a
<code>static_argnames</code> entry, so <code>(bucket, max_tokens)</code> is the compiled shape on
every backend. Warm up at the shape you measure — on the T4G the first request off a fresh engine
took <strong>18.06 s against 4.50 s warm</strong>, a 4.0x whole-request ratio
(<code>2026-08-21-cuda13-py314-g5g</code>).</li>
<li><strong><code>pip</code> supplies CUDA.</strong> <code>jax[cuda13]</code> means the install,
cache restore included, takes <strong>117 seconds</strong> with no build step, no CUDA toolkit and
no Rust — against a from-source build on the vLLM path measured in tens of minutes.</li>
</ul>
<h4>Honest numbers</h4>
<p>Two runs, named, both 2026-08-28, so you can check them:</p>
<figure><img src="img/{t_num}" alt="Throughput and kernel time on the T4G against the L4"><figcaption>Same port and same weights on both cards</figcaption></figure>
<p><strong>Decode is flat and end-to-end is not.</strong> Decode moves 0.8% across a 50x context
range on the T4G and 0.4% on the L4. End-to-end falls hard on both. That is prefill being linear
in the padded bucket, not decode degrading — two different claims, and conflating them makes a
benchmark a lie. <strong>Quote the gauge.</strong></p>
<p>A cost proportional to the <strong>weights</strong> rather than the context produces exactly this
shape, which is why KV is not what sets decode speed on either card, despite Gemma 4's whole KV story.</p>
<p>On context: <code>MAX_MODEL_LEN=4096</code> is the honest number on the T4G — 4,105 prompt tokens
serve, 5,120 fails on a prefill transient.</p>
<h4>The number I could not explain</h4>
<p>Profiling decode with xprof on the Turing card gave this:</p>
<figure><img src="img/gde-code-04-68a4c7a1.png" alt="conversion   54.0%   &lt;-- dtype conversion"></figure>
<p><strong>Zero.</strong> 1,466 ms of kernels across 108 distinct kernels on a Tensor Core GPU,
without one Tensor Core firing. More than half of decode went to converting numbers between formats
before any math happened.</p>
<p>The obvious hypothesis was bf16 weights being converted on a chip with no bf16 datapath. So I
converted the checkpoint to float16 host-side and re-ran. Parameter dtypes read
<code>{{'float16': 541, 'uint8': 1, 'int8': 1}}</code> — and <strong>conversion stayed at
54.0%</strong>.</p>
<p>What I can stand behind is that the measurement is real: the same profile on a
<strong>different instance, a different AMI and a restored cache</strong> landed at 1466.0 ms
against 1467.1 ms. <strong>1.1 ms apart on 1467.</strong></p>
<p>I wrote that paragraph expecting to publish an open question. The control ran first, and it
answered most of it — the same port, same build, same weights, on an Ada card where
<code>_compute_dtype()</code> returns <code>bfloat16</code> off the live compute capability with no
code change:</p>
<figure><img src="img/{t_prof}" alt="Decode profile on the T4G against the L4"><figcaption>The dtype tax, and what survives it</figcaption></figure>
<p><strong>So it was the datapath, not the checkpoint.</strong> Converting the stored weights to
float16 changed nothing because storage dtype was never the problem — Turing has no native bf16,
and the fp32 <code>gemvx</code> line is the tell: XLA was round-tripping through fp32 regardless of
what the file on disk said. Give it a card where storage and compute dtype actually match and the
54% conversion and the 32.8% fp32 path vanish <strong>together</strong> — an 87% tax gone, for 3.7x
the throughput and a rig sitting at its bandwidth roofline instead of 26% of it.</p>
<p><strong>What survived is the part I still cannot explain.</strong> Tensor Core utilization is
0.0% on the Ada card too — 100 distinct kernels, 362.8 ms of them, and not one Tensor Core firing.
Removing the dtype pressure made the machine four times faster without making it touch the hardware
it was sold for. That is the open question now, and it is a better one than the one I started with.</p>
<p>One caveat I will not paper over: the two boxes differ in host architecture and base image as
well as in GPU — aarch64 against x86_64. The <em>payload</em> is byte-identical across them, which
is why I am willing to attribute this to the chip, but it is not a single-variable experiment and I
am not going to call it one.</p>
<h4>Summary</h4>
<p>One JAX port, two GPU generations. The model code, the compilation cache and the static-shape
discipline all transferred untouched, and Gemma 4's awkward geometry — the thing that forces a
special-case Triton kernel on other stacks — turned out to be the easiest part, because in JAX it
is just shapes.</p>
<p>What did not transfer was the fused kernel, because it was written against a memory model the
GPUs do not have. And what nearly went unnoticed was the dtype tax: 87% of decode on pre-Ampere
hardware, with <strong>nothing red in the logs</strong>. It took a profile to find it and a second
chip to explain it. The scariest numbers in this project all looked like success.</p>"""

    html = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head><body>{body}</body></html>\n'

    digest = hashlib.sha256(html.encode()).hexdigest()[:10]
    out = DOCS / f"gde-{digest}.html"
    out.write_text(html)
    print(f"wrote {out.name}  ({len(html):,} bytes)")
    print(f"  tables: {t_spec}  {t_num}  {t_prof}")
    print(f"  cover:  {COVER}")


if __name__ == "__main__":
    main()

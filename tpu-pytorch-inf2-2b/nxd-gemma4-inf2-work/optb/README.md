# Option B — Gemma-4-E2B on Inferentia2 via `torch_neuronx.trace()` of HF transformers

Bypasses AWS NxD entirely (NxD cannot express gemma4's within-forward cross-layer KV-sharing).
Instead we trace the HuggingFace transformers-5.13 `Gemma4TextModel` forward directly. HF threads
KV-sharing via a plain Python dict `shared_kv_states` through the forward → under `torch_neuronx.trace`
this becomes a LIVE graph tensor dependency (exactly why it works on TPU/XLA).

## Status (2026-07-06) — ✅✅ PROPER KV-CACHE DECODE, FAST (~44 tok/s)
Two-graph prefill + incremental decode with the KV cache threaded as graph I/O
(`optb_kv.py` = build/validate, `optb_kv_run.py "<prompt>"` = serve; see optb_kv_result.txt):
- **PREFILL** graph = the Option B forward, extended to also return the 15 non-shared layers' K/V
  (shared layers 15-34 never write cache — HF `modeling_gemma4.py:1273`; API is `cache.layers[i].keys/.values`).
- **DECODE** graph = single-token forward vs a fixed **MAX=128** KV buffer threaded as explicit tensor I/O.
  Custom `StaticKV.update` = one-hot masked write `buf*(1-oh)+k*oh` (pure arithmetic → trace-safe, unlike
  DynamicCache's growing `cat`). Position/one-hot/masks precomputed on HOST (no int-index ops in graph);
  `attention_mask` passed as a **dict** to bypass the fragile `create_causal_mask`. KV-sharing handled by
  `TextModel.forward`'s `shared_kv_states`.
- Verified: `DEV GEN 'The capital of France is **Paris**.'`, `SEQ_MATCH True` (bf16 ids == fp32 CPU ids).
- **Speed: decode ~23 ms/token = ~44 tok/s**; prefill 0.06s (after neff load). "What is Gemma?" → coherent
  115-token answer. One-time neff load ~86s (load once, serve many).
- **Two gotchas:** wraps MUST register `self.lang/self.head` as submodules (else tracer won't ship weights →
  `Input tensor is not an XLA tensor`); compile **bf16** (`--auto-cast all`) — fp32 neffs are 15.4GB and a
  decode graph overflows a single 16GB core. Load prefill on core 0, decode on core 1 (`NEURON_RT_VISIBLE_CORES=0,1`).

## Status (2026-07-05) — ✅ COHERENT MULTI-TOKEN GENERATION ON DEVICE
- Single token: `torch_neuronx.trace` of the text tower **Compiler status PASS**, `DEVICE argmax 818 'The'`, **MATCH True** (maxabs diff vs CPU 7.6e-05). See optb_trace.py.
- Multi-token: **`DEVICE GEN: 'The capital of France is **Paris**.'`**, `SEQ_MATCH True` — device token ids identical to CPU
  (`[818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106]`, stops at EOS 106). See optb_gen.py / optb_gen_result.txt.
- Arbitrary prompts run against the compiled L=40 graph with NO recompile via optb_ask.py, e.g.
  "What is Gemma?" → "**Gemma** is a family of **lightweight, state-of-the-art open models** developed by **Google Deep**[Mind]"
  (truncated only by the L=40 slot ceiling; ~1–2s).

### Env recovery (the original nxenv died)
The python3.12 `/workspace/nxenv` (over `/opt/conda`) is DEAD — /opt/conda deleted, symlinks broken.
Rebuilt by installing into the surviving AMI Neuron venv:
`/opt/aws_neuronx_venv_pytorch_2_8/bin/pip install transformers==5.13.0`
(that venv has torch 2.8.0+cu128 + torch_neuronx 2.8.0.2.12, no transformers — additive/reversible).
RUN with `export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:$PATH` (torch_neuronx needs `libneuronpjrt-path` on PATH;
do NOT `source activate` under SSM's /bin/sh dash).

### Multi-token method (optb_gen.py)
Same proven wrapper, traced at a PADDED fixed length **L=40**. Host-driven greedy "static re-prefill" loop:
each step right-pads current ids to L, sets `attention_mask=[1]*n+[0]*(L-n)`, recomputes `inputs_embeds`+`per_layer_inputs`
on HOST, runs the ONE traced graph, `argmax` at position `n-1`, appends, stops on EOS. Causal+padding mask makes pad
positions inert → identical to CPU. Compile ~9 min; decode re-runs the full L-forward each token (no KV-cache reuse —
productionization = prefill+decode two-graph with KV cache as graph I/O + seq bucketing).

## Recipe (see optb_trace.py)
1. venv over the gwork base: `python -m venv --system-site-packages /workspace/nxenv`, then
   `pip install -c <pins> transformers==5.13.0 huggingface_hub>=0.35 tokenizers>=0.22` (pin torch/torch_neuronx
   so the neuron build isn't clobbered; vllm/NxDI pip-conflict warnings are harmless — Option B doesn't use them).
2. Load `Gemma4ForConditionalGeneration` fp32, `attn_implementation="eager"`. Use `m.model.language_model` + tied `m.lm_head`, softcap 30.
3. Replace GELU with elementary tanh-GELU (torch_neuronx gelu shim rejects `approximate=` kwarg).
4. Move both embedding tables to HOST: compute `inputs_embeds` + `per_layer_inputs` on CPU, pass into
   `forward(inputs_embeds=, per_layer_inputs=, attention_mask=, use_cache=True, past_key_values=DynamicCache())`.
   (The 262144×8960 PLE table trips neuron-cc and blows the 16GB core — keep it off-device.)
5. Plain `DynamicCache()` (no config) — avoids the sliding-window cache truncation slice the tracer rejects.
   Sliding window is enforced by the attention MASK, so correctness is unaffected.
6. `torch_neuronx.trace(w,(ie,am,ple), compiler_args=["--model-type","transformer","--auto-cast","none"])`.

## Box
inf2.8xlarge `i-01718af33c99f0eeb` (us-east-1), container `gwork`, real weights `/workspace/real-gemma4-E2B-it`,
scripts/logs `/workspace/optb_trace.py` / `optb_trace.log`. Full details in memory `gemma4-e2b-gibberish-rootcause.md`.

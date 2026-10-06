# Gemma-4-E2B on Inferentia2 (inf2 / NxD) — Work-in-Progress Snapshot

## Status
- **Root cause of the multi-day gibberish: SOLVED.** The deployed checkpoint was a stripped,
  text-only export **missing the Per-Layer Embedding (PLE) weights**. The real
  `google/gemma-4-E2B-it` is 5.12B, `model_type=gemma4` / `Gemma4ForConditionalGeneration`,
  multimodal (vision+audio+text). Downloaded to `/workspace/real-gemma4-E2B-it` on the box.
- **Coherent output CONFIRMED** with real weights + transformers 5.13 (`Gemma4ForConditionalGeneration`)
  AND through NxD's own `NeuronGemma3ForCausalLM` in **CPU mode** → "Paris / Jupiter / 144".
- **inf2 device: compiles + runs (tp=2, ~25 tok/s) but outputs gibberish.** The blocker is
  **KV-cache sharing** (`num_kv_shared_layers=20`): shared layers (15-34) must reuse the anchor
  layers' (13 sliding / 14 full) current-token K/V *within a single forward*. NxD has no
  within-forward cross-layer K/V channel — its cache is written by the wrapper *after* the pass,
  and the inline per-layer buffer path is gated off in context encoding (where the 1st token is made).

## The fix that's needed (AWS Neuron core)
Implement vLLM-style inline named-cache KV-sharing in NxD's attention/cache core:
write the anchor's cache **inline during the forward** + shared layers gather the anchor's slot.
vLLM does exactly this via `kv_sharing_target_layer_name` (works on XLA → TPU). vLLM 0.16 has
NO neuron platform, so inf2 must go through NxD. AWS already ships a half-built hook: the
`_gemma4_src_idx` redirect in NxD's gemma3 model (hardcoded wrong: >=20/(18,19); correct for
E2B is >=15/(13,14)).

## Minimal repro
- Model: NxD `NeuronGemma3ForCausalLM` + `real4_cfg.py:RealCfg`, model dir `/workspace/real4`
  (flat gemma3_text config from the real text_config + symlinked real `model.safetensors`).
- `run_inference_with_classes(..., cpu_mode=True)`  -> "Paris."   (correct)
- `run_inference_with_classes(..., cpu_mode=False)` -> gibberish  (device / KV-share broken)

## Files here
- `*.diff` — my NxD source edits (vs pristine `.prefix.bak`/`.g4loop.bak`): FORCE_ZERO removal,
  config-driven KV-share rule + loop `_gemma4_src_idx` redirect, raw-KV output threading,
  past_key_value routing (all COMPILE; device still gibberish — see verdict above).
- `real4_cfg.py`, `compile_real4.py`, `real4_config.json` — repro config/compile.
- Debug prints (LOOPDBG/DECDBG) may remain in the box copies.

Full blow-by-blow in memory: `gemma4-e2b-gibberish-rootcause.md`, `gemma4-autoport-run.md`.

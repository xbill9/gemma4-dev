"""Overnight CPU validation of the Gemma 4 31B loader paths.

The 31B exercises two branches of jax_e_loader that have never executed:

  * hidden_size_per_layer_input == 0  -> NO Per-Layer Embeddings. Every PLE
    lookup, projection and norm must be skipped rather than resolving to None
    and silently producing a broken parameter tree.
  * num_kv_shared_layers == 0         -> NO KV sharing, so first_kv_shared_layer_idx
    equals num_hidden_layers and kv_share_map() runs on a degenerate case where
    every layer owns its own K/V.

Both are structural, so they fail at load or first forward — no TPU needed.
Every step is independently guarded: a failure in one must not prevent the
later ones from reporting, because nobody is awake to restart this.
"""
import os, sys, time, json, traceback, resource

sys.path.insert(0, os.path.expanduser("~/gemma"))
RESULTS = {}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KiB -> GB


def step(name):
    def deco(fn):
        print(f"\n{'='*78}\n### {name}\n{'='*78}", flush=True)
        t0 = time.time()
        try:
            out = fn()
            RESULTS[name] = {"ok": True, "secs": round(time.time()-t0, 1),
                             "peak_rss_gb": round(rss_gb(), 2), "detail": out}
            print(f"--- OK ({time.time()-t0:.1f}s, peak RSS {rss_gb():.2f} GB)", flush=True)
        except Exception as e:
            RESULTS[name] = {"ok": False, "secs": round(time.time()-t0, 1),
                             "error": f"{type(e).__name__}: {e}"}
            print(f"--- FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        with open(os.path.expanduser("~/overnight.json"), "w") as f:
            json.dump(RESULTS, f, indent=1)
        return fn
    return deco


# ---------------------------------------------------------------- environment
@step("00-environment")
def _():
    import jax, jax.numpy as jnp
    x = jnp.ones((512, 512))
    assert float((x @ x).sum()) == 512.0 ** 3
    import multiprocessing
    return {"jax": jax.__version__, "devices": str(jax.devices()),
            "cpus": multiprocessing.cpu_count(),
            "mem_gb": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)}


# ---------------------------------------------------------------- unit tests
@step("01-unit-tests")
def _():
    import subprocess, pathlib
    pathlib.Path(os.path.expanduser("~/gemma/tests/__init__.py")).touch()
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                       cwd=os.path.expanduser("~/gemma"), capture_output=True, text=True,
                       timeout=3600)
    tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
    if r.returncode != 0:
        raise RuntimeError("unit tests failed: " + " | ".join(tail))
    return {"tail": tail}


# ---------------------------------------------------------------- E2B control
@step("02-e2b-control-generation")
def _():
    """A known-good model first. If this fails the environment is wrong, not the 31B."""
    from jax_engine import JaxGemmaEngine
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    mid = "google/gemma-4-E2B-it-qat-w4a16-ct"
    snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(snap)
    eng = JaxGemmaEngine(mid, kv_cache_dtype="bf16", quant_mode="w4a16", max_model_len=256)
    eng.load(local_dir=snap)
    eng.bos_token_id = tok.bos_token_id
    ids = tok("<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n",
              add_special_tokens=False)["input_ids"]
    out, st = eng.generate(ids, max_new_tokens=6, temperature=0.0)
    text = tok.decode(out, skip_special_tokens=True)
    gb = round(eng.weight_bytes/1e9, 2)
    del eng
    return {"weights_gb": gb, "text": text}


# ---------------------------------------------------------------- 31B config
@step("03-31b-config")
def _():
    from huggingface_hub import hf_hub_download
    from jax_engine import config_from_hf
    p = hf_hub_download("google/gemma-4-31B-it-qat-w4a16-ct", "config.json")
    cfg = config_from_hf(json.load(open(p)))
    assert cfg.hidden_size_per_layer_input == 0, "expected NO PLE on 31B"
    assert cfg.num_kv_shared_layers == 0, "expected NO KV sharing on 31B"
    assert cfg.first_kv_shared_layer_idx == cfg.num_hidden_layers, \
        f"every layer should own KV; got first_shared={cfg.first_kv_shared_layer_idx}"
    share = cfg.kv_share_map()
    assert share == list(range(cfg.num_hidden_layers)), "degenerate share map must be identity"
    return {"layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
            "kv_heads": cfg.num_key_value_heads, "window": cfg.sliding_window,
            "ple": cfg.hidden_size_per_layer_input, "shared": cfg.num_kv_shared_layers}


# ---------------------------------------------------------------- 31B download
@step("04-31b-download")
def _():
    from huggingface_hub import snapshot_download
    p = snapshot_download("google/gemma-4-31B-it-qat-w4a16-ct",
                          allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
                          max_workers=8)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(p) for f in fs)
    return {"path": p, "gb": round(total/1e9, 2)}


# ---------------------------------------------------------------- 31B load
@step("05-31b-load")
def _():
    """The actual target: does the no-PLE / no-KV-sharing tree build?"""
    from jax_engine import JaxGemmaEngine
    mid = "google/gemma-4-31B-it-qat-w4a16-ct"
    snap = RESULTS["04-31b-download"]["detail"]["path"]
    eng = JaxGemmaEngine(mid, kv_cache_dtype="int8", quant_mode="w4a16",
                         max_model_len=256, window_kv=True)
    t0 = time.time()
    eng.load(local_dir=snap)
    secs = time.time() - t0

    p = eng.params
    assert "embed_tokens" in p and p["embed_tokens"] is not None
    assert p.get("embed_tokens_per_layer") is None, "31B must have no PLE table"
    nones = []
    for i in range(eng.config.num_hidden_layers):
        lp = p[f"layer_{i}"]
        for k in ("q_proj_packed", "k_proj_packed", "v_proj_packed", "o_proj_packed"):
            if lp["attn"].get(k) is None:
                nones.append(f"layer_{i}.attn.{k}")
        for k in ("gate_proj_packed", "up_proj_packed", "down_proj_packed"):
            if lp["mlp"].get(k) is None:
                nones.append(f"layer_{i}.mlp.{k}")
    assert not nones, f"{len(nones)} missing tensors, first 5: {nones[:5]}"
    globals()["_ENG31"] = eng
    return {"load_secs": round(secs, 1), "weights_gb": round(eng.weight_bytes/1e9, 2),
            "layers_checked": eng.config.num_hidden_layers}


# ---------------------------------------------------------------- 31B forward
@step("06-31b-generation")
def _():
    """CPU generation on a 31B is slow; a handful of tokens proves the graph runs."""
    from transformers import AutoTokenizer
    eng = globals().get("_ENG31")
    if eng is None:
        raise RuntimeError("31B did not load; skipping")
    snap = RESULTS["04-31b-download"]["detail"]["path"]
    tok = AutoTokenizer.from_pretrained(snap)
    eng.bos_token_id = tok.bos_token_id
    ids = tok("<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n",
              add_special_tokens=False)["input_ids"]
    t0 = time.time()
    out, st = eng.generate(ids, max_new_tokens=4, temperature=0.0)
    return {"text": tok.decode(out, skip_special_tokens=True),
            "secs": round(time.time()-t0, 1),
            "prefill_ms": round(st.prefill_ms, 1) if st else None}


print("\n" + "="*78)
print("SUMMARY")
for k, v in RESULTS.items():
    print(f"  {'PASS' if v['ok'] else 'FAIL'}  {k:30} {v['secs']:>8.1f}s  "
          f"{v.get('error') or json.dumps(v.get('detail'))[:90]}")
print("OVERNIGHT_DONE", flush=True)

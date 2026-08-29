#!/usr/bin/env python3
"""Profile decode on the T4G, and test whether batching turns the GEMV into a GEMM.

Runs ON the instance, with the service STOPPED -- the profiler needs the GPU to
itself and this loads a second copy of the weights.

    systemctl stop torch-g5g
    PY=$(cat /opt/torch-g5g/PYTHON_BIN)
    $PY /opt/torch-g5g/app/profile_decode.py --out /opt/torch-g5g/prof

Two questions, deliberately separate:

* **--mode kernels** answers "where do the 92 ms/token go". The rig has never
  been profiled; every claim about its decode breakdown so far is INHERITED from
  the JAX sibling, whose profile explicitly does not transfer.
* **--mode batch** tests the central claim of the first benchmark report -- that
  the missing 3x to the bandwidth bound is `B=1` decode being a matrix-VECTOR
  product. If tok/s does not climb with batch, that story is wrong and the
  report needs correcting.

A tensorboard trace is written alongside, which is what `xprof` and `tensorboard`
are installed for.
"""

import argparse
import json
import os
import time


def resolve_compute_dtype(torch, device):
    """Same policy as the serving path: the dtype the DEVICE has."""
    major, minor = torch.cuda.get_device_capability(device)
    return (torch.float16 if (major, minor) < (8, 0) else torch.bfloat16), (major, minor)


def load(torch, transformers, model_id, logits_to_keep_supported=True):
    device = torch.device("cuda")
    dtype, cap = resolve_compute_dtype(torch, device)
    t0 = time.monotonic()
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        .to(device).eval()
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    load_s = time.monotonic() - t0
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"loaded in {load_s:.1f}s  weights={weights/1e9:.3f} GB  dtype={dtype} sm_{cap[0]}{cap[1]}",
          flush=True)
    print(f"attn implementation: {getattr(model.config, '_attn_implementation', '?')}", flush=True)
    return model, tok, device, dtype, weights


def decode_steps(torch, model, ids, n_steps):
    """Prefill once, then n_steps of single-token decode. Returns seconds/step."""
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "google/gemma-4-E2B-it"))
    ap.add_argument("--out", default="/opt/torch-g5g/prof")
    ap.add_argument("--mode", choices=["kernels", "batch", "both"], default="both")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--batches", default="1,2,4,8")
    args = ap.parse_args()

    import torch
    import transformers
    os.makedirs(args.out, exist_ok=True)

    model, _tok, device, dtype, weights = load(torch, transformers, args.model)
    result = {
        "model": args.model,
        "weights_bytes": weights,
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "steps": args.steps,
        "prompt_tokens": args.prompt_tokens,
    }

    base = torch.randint(10, 1000, (1, args.prompt_tokens), dtype=torch.long, device=device)

    if args.mode in ("batch", "both"):
        # THE experiment. If per-step time is flat in B, the step is bandwidth- or
        # launch-bound on the WEIGHTS and batching is the whole 3x. If it scales
        # with B, it is already compute-bound and the report's story is wrong.
        rows = []
        for b in [int(x) for x in args.batches.split(",")]:
            ids = base.repeat(b, 1)
            try:
                decode_steps(torch, model, ids, 4)              # warm this shape
                per_step = decode_steps(torch, model, ids, args.steps)
                torch.cuda.synchronize()
                rows.append({
                    "batch": b, "ms_per_step": round(per_step * 1000, 3),
                    "tokens_per_second": round(b / per_step, 3),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                    "status": "ok",
                })
                print(f"  B={b:2d}  {per_step*1000:7.2f} ms/step  "
                      f"{b/per_step:7.2f} tok/s", flush=True)
            except torch.OutOfMemoryError as exc:
                rows.append({"batch": b, "status": "oom", "error": str(exc)[:200]})
                print(f"  B={b:2d}  OOM", flush=True)
                torch.cuda.empty_cache()
                break
            torch.cuda.reset_peak_memory_stats()
        result["batch_sweep"] = rows

    if args.mode in ("kernels", "both"):
        from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
        ids = base
        decode_steps(torch, model, ids, 4)
        trace_dir = os.path.join(args.out, "tb")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=True,
        ) as prof:
            decode_steps(torch, model, ids, args.steps)

        # torch renamed the aggregate columns; accept either spelling rather than
        # guessing, because the wrong one silently sorts by CPU time.
        evts = prof.key_averages()
        def dev_us(e):
            for attr in ("self_device_time_total", "self_cuda_time_total"):
                if hasattr(e, attr):
                    return getattr(e, attr)
            return 0
        total = sum(dev_us(e) for e in evts) or 1
        top = sorted(evts, key=dev_us, reverse=True)[:25]
        result["kernels"] = [
            {
                "name": e.key[:110],
                "device_us_total": round(dev_us(e), 1),
                "share_pct": round(dev_us(e) / total * 100, 2),
                "calls": e.count,
                "us_per_call": round(dev_us(e) / e.count, 2) if e.count else 0,
                "per_step_ms": round(dev_us(e) / args.steps / 1000, 3),
            }
            for e in top if dev_us(e) > 0
        ]
        result["kernel_device_us_total"] = round(total, 1)
        result["trace_dir"] = trace_dir
        with open(os.path.join(args.out, "kernels.txt"), "w") as fh:
            fh.write(evts.table(sort_by="self_cuda_time_total", row_limit=40))
        print(f"\n{'kernel':<62}{'share':>8}{'calls':>8}{'ms/step':>10}", flush=True)
        for k in result["kernels"][:12]:
            print(f"{k['name'][:60]:<62}{k['share_pct']:7.2f}%{k['calls']:8d}"
                  f"{k['per_step_ms']:10.3f}", flush=True)

    with open(os.path.join(args.out, "profile.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nwrote", os.path.join(args.out, "profile.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

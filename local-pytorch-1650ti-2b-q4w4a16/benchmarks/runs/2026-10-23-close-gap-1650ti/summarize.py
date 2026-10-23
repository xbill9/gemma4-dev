import json, sys
ref = {json.loads(l)["prompt"]: json.loads(l)["token_ids"] for l in open("reference-text-v1.jsonl") if '"prompt"' in l}
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r["step"] == "prompt":
        d = next((i for i, (x, y) in enumerate(zip(r["token_ids"], ref[r["prompt"]])) if x != y), None)
        same = r["token_ids"] == ref[r["prompt"]]
        print(r["prompt"], r["decode_tok_per_s"], r["prefill_ms"], r["peak_cuda_mib"], "same" if same else f"diff@{d}")
    elif r["step"] == "profile":
        print(r["gpu_kernel_ms_per_token"], r["graph_replay_ms"], r["kernels_per_token"], r["buckets"])
        for t in r["top"][:6]:
            print("  ", t)
    else:
        print({k: v for k, v in r.items() if k not in ("torch", "transformers")})

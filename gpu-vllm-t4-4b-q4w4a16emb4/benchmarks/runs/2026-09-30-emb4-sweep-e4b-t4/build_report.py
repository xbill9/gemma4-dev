"""Build the schema-1.1 serving report for this sweep from the raw `emb4/*.json`.

Each cell reports its median-throughput repeat, embedded verbatim under `raw`, with
all three repeats' output throughput and their cv in `raw.repeats` — the same
convention as the MI300X sweep reports. Copied from gpu-vllm-t4-2b-w4a16's
2026-09-29-emb4-sweep-t4; only the run id, model and the E4B facts changed.

    python3 build_report.py     # writes ../../reports/2026-09-30-emb4-sweep-e4b-t4.json
"""

import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ID = "2026-09-30-emb4-sweep-e4b-t4"
OUT = os.path.join(HERE, "..", "..", "reports", f"{RUN_ID}.json")
CELL = re.compile(r"c(\d+)-in(\d+)-out(\d+)\.rep(\d+)\.json$")
MODEL = "xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4"


def lat(r: dict, name: str) -> dict:
    return {k: round(r[f"{k}_{name}_ms"], 2) for k in ("mean", "median", "p99")}


def main() -> None:
    cells: dict = {}
    for path in glob.glob(os.path.join(HERE, "emb4", "*.json")):
        m = CELL.search(path)
        if m:
            c, i, o, _ = map(int, m.groups())
            with open(path) as fh:
                cells.setdefault((i, c, o), []).append(json.load(fh))

    sweep = []
    for (i, c, o), runs in sorted(cells.items()):
        runs.sort(key=lambda r: r["output_throughput"])
        med = runs[len(runs) // 2]
        tput = [round(r["output_throughput"], 2) for r in runs]
        raw = dict(med)
        raw["repeats"] = {
            "n": len(runs),
            "output_tok_per_s": tput,
            "cv_pct": round(statistics.pstdev(tput) / statistics.mean(tput) * 100, 2),
        }
        failed = sum(r.get("failed", 0) for r in runs)
        sweep.append(
            {
                "concurrency": c,
                "input_len": i,
                "output_len": o,
                "status": "ok" if failed == 0 else "failed",
                "request_rate_rps": round(med["request_throughput"], 4),
                "output_tok_per_s": round(med["output_throughput"], 2),
                "total_tok_per_s": round(med["total_token_throughput"], 2),
                "ttft_ms": lat(med, "ttft"),
                "tpot_ms": lat(med, "tpot"),
                "itl_ms": lat(med, "itl"),
                "per_stream_tok_per_s": round(1000 / med["median_tpot_ms"], 2),
                "raw": raw,
            }
        )

    report = {
        "schema_version": "1.1",
        "run": {
            "id": RUN_ID,
            "date": "2026-09-30",
            "operator": "xbill",
            "source": f"benchmarks/runs/{RUN_ID}/",
            "notes": (
                "Three repeats per cell with the same prompt seeds as gpu-vllm-t4-2b-w4a16's "
                "2026-09-29-emb4-sweep-t4 (E2B), so every cell sends the prompts that run did; "
                "slot 4 is the only difference. Served without the drafter, as that run was. Each cell reports its median-throughput repeat; raw.repeats carries "
                "the spread. REPORT.md's table is the mean of the three (results.csv, aggregate.py). "
                "c=16 exceeds --max-num-seqs 8, so its extra requests queue. "
                "The first pass died during 4096/c=4 rep 2; the server was restarted with the same "
                "flags (clearing the prefix cache that cell had filled) and resume.sh ran the ten "
                "remaining cells. See REPORT.md."
            ),
        },
        "hardware": {
            "accelerator": "nvidia-t4",
            "chips": 1,
            "hbm_gb_per_chip": 16,
            "machine_type": "n1-standard-2",
            "host": {"cloud": "gcp", "zone": "us-west2-b", "provisioning": "on-demand"},
        },
        "model": {
            "id": MODEL,
            "family": "gemma-4",
            "parameters_b": 4,
            "weights_dtype": "int4",
            "quantization": "compressed-tensors W4A16, group 32; embed_tokens, PLE table and untied lm_head int4",
            "max_model_len": 16384,
            "architecture_notes": (
                "Google's QAT q4_0 weights repacked to compressed-tensors on the exact QAT grid, text "
                "only (Gemma4ForCausalLM). E4B: 4.5B effective, 8.0B total. Model loading 4.54 GiB, "
                "KV cache 310,499 tokens on the warm start (evidence/setup.txt)."
            ),
        },
        "software": {
            "engine": "vllm",
            "version": "0.29.0",
            "backend": "cuda (torch 2.13.0+cu130, triton 3.7.1, TRITON_ATTN, Turing tile clamp applied)",
            "tensor_parallel_size": 1,
            "serve_args": [
                "serve",
                MODEL,
                "--host",
                "127.0.0.1",
                "--dtype",
                "float16",
                "--kv-cache-dtype",
                "auto",
                "--gpu-memory-utilization",
                "0.9",
                "--max-model-len",
                "16384",
                "--max-num-seqs",
                "8",
                "--language-model-only",
            ],
        },
        "throughput": {
            "workload": {
                "tool": "vllm bench serve",
                "dataset": "random",
                "output_len": 128,
                "runs_per_point": 3,
                "notes": "--ignore-eos --temperature 0 --num-warmups 2; sweep.sh.",
            },
            "sweep": sweep,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {os.path.relpath(OUT, HERE)}: {len(sweep)} cells")


if __name__ == "__main__":
    main()

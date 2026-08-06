# Serving Benchmark Reports

A standard format for capturing what a (model, hardware, engine config) deployment is worth:
identity of the stack, a throughput sweep, capability probes, memory anatomy, startup timeline,
and cost. One JSON file per deployment, validated against `serving-report.schema.json`
(JSON Schema draft 2020-12).

- **Schema:** `serving-report.schema.json`
- **Example:** `reports/2026-07-21-gemma4-e2b-v6e1.json` — Gemma 4 E2B on one TPU v6e chip,
  the full dataset behind `devto-post.md`

## Run index

File naming: `reports/<date>-<model-short>-<hw-short>.json`, matching `run.id`
(e.g. `2026-07-21-gemma4-e2b-v6e1.json`). One file per (model, hardware) cell:

| Model \ Hardware | v6e-1 | v6e-8 |
|---|---|---|
| gemma-4-E2B-it | [2026-07-21](reports/2026-07-21-gemma4-e2b-v6e1.json) | — |
| gemma-4-E4B-it | planned | — |

## Why not an existing standard

No public schema covers this combination; the closest are slices of it:

| Standard | Covers | Missing |
|---|---|---|
| `vllm bench serve --save-result` JSON | raw throughput/latency for one run | hardware, cost, capabilities |
| Hugging Face `model-index` (model-card YAML) | eval/capability results, leaderboard-compatible | hardware, serving throughput, cost |
| MLPerf Inference submission format | rigorous perf methodology | capability probes, cost; heavyweight |

This schema embeds the first (see `throughput.sweep[].raw`) and can be exported to the second
if a model card ever needs it.

## Recording a run

1. Copy the example, name it `reports/<date>-<model-short>-<hw-short>.json` (matches `run.id`).
2. Run each sweep point and capture its metrics. Easiest path: the `run_vllm_benchmark`
   MCP tool with `save_result=True` returns a ready-made `throughput.sweep[]` entry
   (typed fields populated, raw dump embedded minus per-request arrays) — one call per
   concurrency level. Running `vllm bench serve` by hand instead: pass `--save-result`,
   embed the emitted JSON under `throughput.sweep[].raw`, and copy the headline numbers
   into the typed fields:

   | `vllm bench serve` field | schema path |
   |---|---|
   | `max_concurrency` | `sweep[].concurrency` |
   | `request_throughput` | `sweep[].request_rate_rps` |
   | `output_throughput` | `sweep[].output_tok_per_s` |
   | `total_token_throughput` | `sweep[].total_tok_per_s` |
   | `median_ttft_ms` / `p99_ttft_ms` | `sweep[].ttft_ms.median` / `.p99` |
   | `median_tpot_ms` / `p99_tpot_ms` | `sweep[].tpot_ms.median` / `.p99` |
   | `median_itl_ms` | `sweep[].itl_ms.median` |

   `per_stream_tok_per_s` is derived: ≈ 1000 / median TPOT.
3. Capability probes get a `verdict` per probe (`pass` / `partial` / `fail` / `blocked` /
   `not_tested`) plus a domain-level verdict, with `conditions` recording any config the
   verdict depends on (e.g. `enable_thinking: true`). A model variant that won't load goes in
   `load_matrix`, not `capabilities`.
4. Anything not measured: omit the section. Only `run`, `hardware`, `model`, `software` are
   required.

## Validating

```bash
python3 -c "import json,sys; json.load(open(sys.argv[1])) and None; print('JSON OK')" reports/<report>.json
check-jsonschema --schemafile serving-report.schema.json reports/<report>.json
```

If `check-jsonschema` is missing: `pip install check-jsonschema` (no virtualenv — system python3).

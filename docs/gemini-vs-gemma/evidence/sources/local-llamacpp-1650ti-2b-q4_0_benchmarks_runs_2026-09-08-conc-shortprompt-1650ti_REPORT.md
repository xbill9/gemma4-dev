# Short-prompt / long-generation concurrency — `local-llamacpp-1650ti-2b-q4_0`, 2026-09-08

128 in / 512 out, concurrency 1→16. Machine-readable:
`../../reports/2026-09-08-conc-shortprompt-1650ti.json`. Harness: `sweep.py` at the rig root,
prompt cache defeated and **verified defeated** (see below).

This is the sweep `CLAUDE.md` named as the open question after 2026-09-03: *"A long-prompt workload
will not benefit from concurrency on this card. Short-prompt, long-generation is the shape that
would — untested, and the obvious next sweep."*

**Headline: it does — 2.6× against the long-prompt shape's 1.47×, peaking at 164 tok/s at c=16.
But only c=1 and c=16 reproduce. Everything between them is bimodal, and the cause is not heat.**

## The curve, thermally controlled

Card cooled to 52 °C before every level, 3 trials per level, `cooled.json`.

| c | median tok/s | modes observed | spread | TTFT ms | TPOT ms | per stream |
| ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | 63.77 | 62.9 / 63.8 / 63.8 | 1.5% | 669 | 14.49 | 63.8 |
| 2 | 62.58 | **62.3, 62.6 / 97.6** | **56.7%** | 1337 | 29.72 | 31.3 |
| 4 | 128.19 | **94.5 / 128.2, 129.0** | **36.6%** | 2305 | 26.91 | 32.0 |
| 8 | 109.91 | **109.6, 109.9 / 127.8** | **16.6%** | 4839 | 64.14 | 13.7 |
| 16 | 164.07 | 164.0 / 164.1 / 164.4 | 0.2% | 9244 | 80.05 | 10.3 |

**Do not quote c=2, c=4 or c=8 from this table.** A median of three samples drawn from two modes
reports whichever mode won two tosses; the c=2 median (62.58) and the c=4 median (128.19) are
opposite modes of the same instability, which is why the curve appears to *fall* from c=1 to c=2.

The ascending as-run pass (`concurrency.json`, no cooldown) gave 64.73 / 99.23 / 129.49 / 160.48 /
163.16 and a 5-repeat confirmation put c=8 at 159.80. Neither reproduced at c=8 later in the
session (~110–128). **c=16 is the only multi-stream level that reproduced across every pass.**

## The instability is decode batching, not temperature

TTFT is constant within each level and TPOT is not:

| c | TTFT ms (3 trials) | TPOT ms (3 trials) |
| ---: | :--- | :--- |
| 2 | 1377.7 / 1325.9 / 1337.1 | **17.93** / 29.63 / 29.72 |
| 4 | 2364.5 / 2436.2 / 2304.5 | **26.58** / 37.91 / **26.91** |
| 8 | 4879.9 / 4689.2 / 4838.5 | 63.74 / **53.86** / 64.14 |

Prefill costs the same every time; the per-token decode rate is what moves. At c=2 the slow mode's
29.6 ms/token is **2× the c=1 figure of 14.4** — two streams taking turns rather than decoding in
one batch. That is llama.cpp intermittently failing to co-schedule decode across slots, and it is
the whole of the variance.

**The thermal hypothesis was tested and rejected.** It was the obvious reading — this is a Max-Q
part, `CLAUDE.md` already records that it throttles and once inverted a curve, and the card is
power-capped (≈39 W of a 40 W limit, 1530–1770 MHz against a 2100 MHz maximum). Three things rule
it out:

- **c=16 is the hottest level and the most stable** (0.2%), and it is also the fastest. Thermal
  decay cannot be strongest in the middle of the range.
- **Fast and slow trials occur at the same temperature and the same clock.** c=4 trial 2 is slow at
  1680 MHz; trial 3 is fast at 1740 MHz, 1 °C hotter.
- **The minima reproduce to 0.2%** across independent passes (c=4: 95.26 then 95.10; c=8: 129.98
  then 130.27). Throttling is continuous; this is two discrete states.

Heat is real here — clocks fall ~10% within a single level and `nvidia-smi` shows a nonzero
SW-power-cap counter — but it is not what these spreads are.

## What this shape is actually measuring

Every request hit the 512-token cap: `output_tokens_total` is exactly `c × 512` at every level, so
every generation ended `finish_reason: length`. Gemma 4 reasons, and at 512 tokens most of that
budget is the thinking block. **The throughput is real; the task is fictional.** This is a decode
throughput probe, not a measurement of answering anything.

Prefill is the minority of the work at this shape (128 in vs 512 out), which is exactly why
concurrency pays here and did not at 512/128 — that run was 4× prefill-dominated and prefill does
not batch. The two results agree; they are different phases in different proportions.

## Prompt cache: verified, not assumed

`llamacpp:prompt_tokens_cached_total 0` against `prompt_tokens_total 17689` (`metrics.prom`).

The 2026-09-03 run had to discard an artifact after discovering the prefix cache had answered 656
of 661 prompt tokens despite the harness's leading nonce. `/metrics` was enabled on this rig on
2026-09-08, so that condition is now checked directly rather than inferred from a suspiciously
good number. `cache_prompt:false` plus a leading nonce is confirmed sufficient.

## Reproducing

Server sizing differs from the context sweep — llama.cpp splits `--ctx-size` across slots:

```
llama-server -m $MODEL_PATH --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 16384 -ctk f16 -ctv f16 -fa 1 -t 4 --parallel 16 --metrics
```

16 × (128 + 512) = 10240 ≤ 16384, and `--parallel 16` ≥ max concurrency. Measured 1818 MiB of
4096 — matching the figure `CLAUDE.md` records for this configuration.

```
python3 sweep.py --base http://127.0.0.1:8080/v1 --model $MODEL_NAME \
  --rig local-llamacpp-1650ti-2b-q4_0 --out benchmarks/runs/<id> \
  --concurrency 1,2,4,8,16 --conc-input 128 --conc-output 512 \
  --repeats 3 --prompt-mode unique --decode-source auto
```

`--decode-source auto` resolves to `stream`, as it must: `llama-server` emits no
`usage.decode_tokens_per_second`.

## Open

- **Why decode batching is intermittent below c=16.** Arrival timing against llama.cpp's batch
  formation is the obvious suspect and is untested. Until it is understood, this rig cannot report
  a trustworthy mid-range concurrency number at any shape.
- **`--repeats 3` is not enough for a bimodal cell.** The 2026-09-03 fix that added repeats assumed
  noise around one value. Reporting a median of a bimodal sample is worse than reporting the modes.
- Whether the same bimodality is present at 512/128 and was hidden by that shape's prefill
  dominance. The 2026-09-03 run's one wide cell (c=8, 7.7%) is a candidate.

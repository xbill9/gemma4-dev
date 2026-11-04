# 2026-11-04 — llama.cpp with the whole 8 GB card, 1 to 64 requests

The card serves nothing but the model, so memory llama.cpp leaves unused buys nothing. The question is whether
llama.cpp gets faster when allowed more of it. Its 2026-11-03 sweeps (`../2026-11-03-concurrency-3050oem`,
`../2026-11-03-concurrency-drafters-3050oem`) peaked at 2,621 MiB: 64 slots x 1,024 tokens, default batches
(`-b 2048 -ub 512`), and the 1,260 MiB per-layer table in CPU RAM, because those runs passed no `-ot`. Same
binary (`7182ef71e`), model, flags otherwise and `sweep.py` arguments.

`probe.sh`: four settings at 8 and 64 requests, 3 repeats. `sweep.sh`: the full 1–64 sweep, 5 repeats, with the
best setting, without a drafter and with the exact Q4_0 drafter at 2 tokens. All ran after an Android build on the
host had finished.

#### Probe, aggregate output tok/s

| setting | 512/128 at 8 | at 64 | 128/512 at 8 | at 64 | peak memory |
|---|---|---|---|---|---|
| as 2026-11-03 | 237 | 368 | 313 | 627 | 2,621 MiB |
| + per-layer table on the card (`-ot`) | 239 | 372 | 313 | 633 | 3,881 MiB |
| + `-ub 2048 -b 2048` | **242** | **377** | **314** | **633** | 4,255 MiB |
| + shared cache (`-kvu`) | 222 | 206 | 288 | 443 | 4,281 MiB |

#### Full sweep with `-ot` and `-ub 2048 -b 2048`

- **No drafter: level with 2026-11-03**, 0.996–1.024x at every level of both shapes (64 requests: 375 against
  368 and 631 against 633 tok/s), at 4,257 MiB.
- **With the drafter: slower than 2026-11-03**, 0.66–0.92x on 512/128 and 0.39–0.62x on 128/512 (one request:
  117 against 190 tok/s on 128/512), at 4,439 MiB. Which of the two changes slows the drafter was not
  separated.

**More of the card does not make llama.cpp faster.** Its limit under load is the time per generated token,
24.6 ms at 8 requests and 110 ms at 64 on 512/128, against SGLang's 9.7 and 33.2 ms on the same card. Its
2026-11-03 numbers stand as its best, with and without the drafter.

#### Files

`a-base/`, `b-ot/`, `c-ot-ub2048/`, `d-ot-ub2048-kvu/` (probe), `full-none/`, `full-q4x-n2/` (sweep): each with
`concurrency.json` per shape, `llama-server.log`, `gpu.csv`, `command.txt`; `probe.log`, `sweep.log`,
`version.txt`, `chain.sh`.

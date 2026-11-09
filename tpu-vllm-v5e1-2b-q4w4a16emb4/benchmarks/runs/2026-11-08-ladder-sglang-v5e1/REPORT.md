# v5e-1 ladder, SGLang-JAX arm: Gemma 4 E2B does not load (2026-11-08)

Two unattended runs on flex-start v5e-1 queued resources in `us-west4-a` (`ladder-v5e1-sglang`, `ladder-v5e1-sglang2`), each driven by `flight/run-sgl.sh` and deleted by `finish.sh` when it ended. Raw archives: `gs://aisprint-491218-ladder-v5e1/2026-11-08-ladder-sglang-v5e1/<qr>/ladder-run.tgz`, SHA-256 checked against `ladder-run.sha256` on download.

Stack (`ladder-v5e1-sglang2/versions.txt`): sglang-jax `b3e3e16b2` (0.1.0rc1.dev113), jax/jaxlib 0.11.1, libtpu 0.0.46.3, transformers 5.12.1, CPU torch 2.15.0. Model `google/gemma-4-E2B-it` at revision `4adf1574`, bf16, `--tp-size 1 --context-length 16384`.

#### Result

No job served a request, so there is no speed, GSM8K or sweep number for SGLang-JAX.

- `ladder-v5e1-sglang` (19:11 UTC): `ModuleNotFoundError: No module named 'torch'` building `Gemma4Processor`. Setup gap; `setup-sgl.sh` then added CPU torch and torchvision.
- `ladder-v5e1-sglang2` (19:27 UTC): the scheduler fails in `get_num_kv_heads_by_tp` with `TypeError: '>=' not supported between instances of 'int' and 'NoneType'`.

#### Why the second run fails

`sgl_jax/srt/configs/gemma4.py` sets `num_key_value_heads = getattr(tc, "num_global_key_value_heads", <sliding count>)`. E2B's `config.json` carries `"num_global_key_value_heads": null`, so `getattr` returns `None` and the default is never used; the 31B config has `4` there. A one-line change (`or`) gets past this point.

Past it, `sgl_jax/srt/models/gemma4.py` has no per-layer embedding and no KV-sharing code (0 matches for `per_layer`/`kv_shared`), and E2B uses both (`hidden_size_per_layer_input` 256, `num_kv_shared_layers` 20). sglang-jax `main` had no later commit to either Gemma 4 file as of 2026-11-08. SGLang-JAX implements the Gemma 4 31B and 26B A4B decoders, not E2B.

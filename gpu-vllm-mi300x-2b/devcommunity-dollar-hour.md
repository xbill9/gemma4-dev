Category: Artificial Intelligence (AI) → Cloud & Edge AI → AMD Developer Cloud
Tags: rocm, gpu, instinct, performance, developer-cloud

---

# Serving Gemma 4 E2B on one MI300X with vLLM — a 12-cell sweep, and where the ceiling actually is

A follow-up to my numeric-formats post, from the same Developer Cloud droplet. This time the
card was serving: `google/gemma-4-E2B-it`, reference bf16, one MI300X (`gfx942`, 191.7 GiB),
`vllm/vllm-openai-rocm:nightly-rocm100`, `--max-model-len 32768 --gpu-memory-utilization 0.90`.
Load came from vLLM's own `bench serve` in a second container with no GPU device attached,
128 output tokens throughout, three repeats per cell.

Output tokens per second, median of three, worst-cell coefficient of variation 7.96%:

| context | 1 stream | 4 | 16 | 64 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 340.9 | 1,124.1 | 3,569.1 | **10,284.0** |
| 1,024 | 305.1 | 969.1 | 2,653.6 | 5,398.2 |
| 8,192 | 192.8 | 445.5 | 688.2 | 702.7 |

Counting prefill too, 64 streams at 1,024 context moves 48,583 tokens a second. Single stream
at 1,024 context: 32.67 ms to first token, 3.04 ms per token after it.

**The long-context ceiling is prefill, not KV memory.** The boot log states the engine's own
arithmetic:

```
Available KV cache memory: 155.04 GiB
GPU KV cache size: 9,026,017 tokens, Maximum concurrency for 32,768 tokens per request: 275.45x
```

The heaviest cell in the grid, 64 x 8,192, wants 524,288 KV tokens — 5.8% of that pool, by
arithmetic. Yet the 8,192 row flattens at 688 → 703 tok/s from 16 to 64 streams while the 128
row keeps climbing. Time per output token barely moves with context (2.85 / 3.04 / 3.9 ms);
time to first token does (12.97 / 32.67 / 169.61 ms). On the TPUs I run the same checkpoint on,
KV capacity is what runs out first. On 192 GB serving a 2B model it never binds, so the sizing
question on this card is how much prefill you can afford, not how many sequences fit.

**Benchmarking this server measured its prefix cache until I stopped it.** `vllm bench serve`
derives prompts from `--seed`, which defaults to 0, and prefix caching is on in this deployment. An
earlier sweep reused seeds 0/1/2 in every cell, and seed 0's prompts had already been served
earlier that day. Warm repeat against the cold ones, same cell, same minute:

```
ctx   conc   warm tok/s   cold mean   ratio
 128    64      10849.2     10496.2    1.03x
1024    64       8930.4      6252.7    1.43x
8192    16       1961.5       870.2    2.25x
```

Near 1x at short context and 2.25x at 8,192 is exactly the gradient a prefill-saving cache
predicts, which is how I spotted it — worst-cell cv on that run was 51.1%. Per-repeat seeds are
not enough either, because cells at one context length draw from the same pool. The numbers
above give every cell and every repeat a seed no other run uses. If you are comparing vLLM
configs on Instinct, check this before believing a long-context win.

Three operational notes from the same box:

- **The newest `rocm/vllm` image I tried cannot load Gemma 4.** The current vendor build dies in
  config parsing, before touching the GPU: `AmbiguousGlobalPerLayerAttributeError: 'head_dim'
  is a per-layer attribute`. Gemma 4 uses 256-wide heads on sliding-attention layers and 512 on
  full-attention ones, and that build predates vLLM's `Gemma4ModelArchConfigConvertor`. It is
  a version-pairing bug, not a ROCm one. The upstream nightly works.
- **The nightly tag moved under me overnight.** Same `nightly-rocm100` tag, two different vLLM
  builds a day apart: `0.29.1rc1.dev187+gaf1c01499` on the 16th, `0.3.1.dev3+g0bfc7a15d` on the
  17th. Read the version from the running container, not the tag, and pin a digest if drift
  matters.
- **It takes 160 seconds from `docker run` to an endpoint that answers**, for weight load,
  `torch.compile` and graph capture. A running container is not a serving model; poll
  `/v1/models`, not `docker ps`.

On cost: at $1.99/hr (read from the DigitalOcean API's `price_hourly`), 64 streams at 1,024
context works out to 2,713 output tok/s per dollar-hour, by arithmetic. That was 1.4x–2.9x a
TPU v5e-1 serving the same checkpoint when both are priced on-demand. There is no preemptible
tier for these droplets, though, and at 16 streams or fewer a spot v5e beat it per dollar. And
powering the droplet off does not stop the meter — only destroying it does.

Open question for anyone who has tuned this: the server selected the `TRITON_ATTN` backend and
I have not tried `VLLM_ROCM_USE_AITER=1`. Given that prefill is the ceiling here, I'd expect
AITER's attention kernels to move the 8,192 row more than anything else. Has anyone measured
that on gfx942 with a hybrid sliding/full-attention model?

Full write-up, with the deploy steps and the price tables:
https://xbill999.medium.com/serving-gemma-4-on-an-amd-mi300x-what-1-99-an-hour-buys-bcc000cd5edc

Rig and benchmark reports (Apache-2.0):
https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

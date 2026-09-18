Category: Community Corner → Showcase → Projects & Demos
Tags: rocm, gpu, instinct, performance, developer-cloud

---

# gpu-vllm-mi300x-2b: an MCP devops agent that serves Gemma 4 on one MI300X

An open-source rig (Apache-2.0) that serves `google/gemma-4-E2B-it` through vLLM on one AMD
Instinct MI300X, on an AMD Developer Cloud droplet, and drives it through a single-file MCP
server — so Claude Code, Codex or any MCP client can run the whole lifecycle in plain language.

**Repo:** https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-mi300x-2b

**What the agent can do (20 tools):**

- **Droplet control** over the DigitalOcean v2 API: `list_droplets`, `droplet_status`,
  `start_droplet`, `stop_droplet`, `reboot_droplet`, `action_status`. There is deliberately no
  create or destroy — those are dollar-per-hour decisions left to a human in the console.
- **On the box** over SSH: `gpu_status` (rocm-smi / amd-smi), `run_on_droplet`, `ssh_command`.
- **Serving:** `check_image`, `pull_image`, `download_weights`, `deploy_vllm`, `stop_vllm`,
  `serving_status`, `server_logs`, `analyze_logs`.
- **Verification:** `query_model`, and `verify_capabilities`, which probes text, thinking, tool
  calling and vision with requests whose right answer is known in advance (the vision probe is a
  generated checkerboard, not a fetched photo).
- **Benchmarking:** `run_vllm_benchmark` runs `vllm bench serve` in a second container with no
  GPU device attached, so it is safe beside a live server.

**What it found on the card** (full numbers in the Cloud & Edge AI thread linked below):

- 10,284 output tok/s at 64 streams, 128-token context, bf16, median of three repeats.
- The long-context ceiling is prefill, not KV memory: the heaviest cell uses 5.8% of a
  9,026,017-token KV pool — the opposite of the sizing rule on the TPUs I run the same model on.
- `check_image` exists because the newest `rocm/vllm` image I tried cannot load Gemma 4 at all
  (`AmbiguousGlobalPerLayerAttributeError` on `head_dim`); the upstream nightly works.
- The benchmark harness gives every cell and repeat a unique `--seed`, because with prefix
  caching on, a reused seed inflated the 8,192-context result 2.25x.

Measurement thread: https://devcommunity.amd.com/t/serving-gemma-4-e2b-on-one-mi300x-with-vllm-a-12-cell-sweep-and-where-the-ceiling-actually-is/1048

Write-up: https://xbill999.medium.com/serving-gemma-4-on-an-amd-mi300x-what-1-99-an-hour-buys-bcc000cd5edc

Feedback welcome — especially from anyone who has measured `VLLM_ROCM_USE_AITER=1` on gfx942.

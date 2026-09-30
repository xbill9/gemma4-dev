# gce-vllm-v6e1-31b-awq

One cell of the Gemma 4 repack sweeps: **`cyankiwi/gemma-4-31B-it-AWQ-4bit`** through vLLM on **TPU v6e-1
(`ct6e-standard-1t`)**, provisioned as a Compute Engine instance. An artifact rig: measurements only.

Slot 5, `awq`: cyankiwi's AWQ export, stored as compressed-tensors W4A16.

It holds one boot log and no suite: on the stock image the checkpoint failed to load with
`NotImplementedError: compressed-tensors scheme ... is not yet supported in the JAX path`
(`QUANTIZATION.md`, "Gemma 4 is JAX-path only").

## Other runs

- [`2026-09-24-sweep-v6e1`](benchmarks/runs/2026-09-24-sweep-v6e1/)

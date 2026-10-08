# CLAUDE.md — `gpu-vllm-mi300x-4b-q4w4a16`

## This is an artifact rig, not a serving rig

No `server.py`, no MCP server, no skill, no plugin manifest, no `tpu.env`, and none is owed. It holds
MI300X data-type sweep measurements for `xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text` and nothing else. Do not scaffold it into a full
rig. `NAMING.md` has the rule for artifact rigs.

## Where the runs came from

`gpu-vllm-mi300x-2b/dtype_sweep.py --size 4b` serves this checkpoint through the engine rig
`gpu-vllm-mi300x-2b-q4w4a16` (every checkpoint setting comes from the environment, so its own
`tpu.env` model lines do not apply) and files the run here through `SUITE_OUTPUT_DIR`. The arm's
settings are the `q4w4a16` entry of `SIZES["4b"]` in that script.

A number from this rig is an A/B twin only of the other arms of the same run id: same droplet, same
image digest, same day. Never difference it against a TPU, L4 or T4 rig and read it as a hardware result.

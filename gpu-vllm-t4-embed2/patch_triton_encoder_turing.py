#!/usr/bin/env python3
"""Make vLLM's encoder-only Triton attention fit Turing's 64 KiB per block.

THIS IS NOT THE SIBLINGS' PATCH, and the difference is the first thing to carry.
`gpu-vllm-t4-2b` clamps `triton_unified_attention`, the DECODER kernel. This model
is bidirectional (`AttentionType.ENCODER_ONLY`), so vLLM's TRITON_ATTN backend
routes every layer through `_forward_encoder_attention` ->
`vllm/v1/attention/ops/triton_prefill_attention.py::context_attention_fwd` and the
unified kernel never runs. Applying the sibling's clamp here edits code nothing
calls and fixes nothing.

THE MODEL RUNS IN float32 ON THIS PART, AND THAT IS WHAT BREAKS THE KERNEL.
Google: float16 is incompatible with EmbeddingGemma 2, and Turing has no bf16
datapath (vLLM would silently cast bf16 to fp16 anyway). So `--dtype float32`.
MEASURED 2026-10-07 on this T4, vLLM 0.31.1rc1.dev63+gd169c5a90, stock kernel:

    head_dim 256 (sliding layers), BLOCK 32 fp32:
        OutOfResources: shared memory, Required: 69632, Hardware limit: 65536
    head_dim 256, BLOCK 16 fp32:          fits, max|err| 5.7e-06 vs torch
    head_dim 512 (full layers), BLOCK 16 fp32, num_stages 1:
        OutOfResources: Required: 66560      <- 16 is tl.dot's floor; no tile fits
    head_dim 512, BLOCK 16, q/k/v float16, fp32 accumulate:
        fits, relative error 4.6e-04 vs an fp32 reference

So the clamp does two things, both only on CUDA parts below SM 8.0:

  * BLOCK <= 16 whenever head_dim >= 256;
  * for float32 inputs at head_dim >= 512, run THIS KERNEL ONLY in float16 (the
    kernel accumulates in fp32 regardless) and copy the result back. q, k and v
    are RMS-normalised immediately before attention in this model (q_norm,
    k_norm, a weightless v_norm), so their range is bounded and fp16 rounding is
    the only cost. The end-to-end parity numbers are in the rig's CLAUDE.md;
    `make parity` re-measures them.

FlexAttention was tried first and is NOT the route: it needs `--enforce-eager`
plus two more edits (its `create_block_mask` is compiled with
`mode="reduce-overhead"`, whose CUDA-graph output is overwritten when the
sliding and full masks are built back to back), and its fp32 template still
needs exactly 65,536 B at the 16 floor. Both edits were reverted.

DESIGN RULE, inherited from the sibling: FAIL LOUDLY, NEVER SILENTLY NO-OP. A miss
on the anchor is exit 2 with the surrounding source, because an unpatched kernel
fails minutes later at engine start and reads like a configuration problem.

THE TARGET IS THIS RIG'S OWN site-packages (`PYTHONUSERBASE=/opt1/pyuser-embed2`),
not the `/opt1/pyuser` the generative T4 rigs share. Re-run after any vLLM
reinstall there: pip silently reverts this.

Usage:
    patch_triton_encoder_turing.py PATH            # patch in place (idempotent)
    patch_triton_encoder_turing.py --check PATH    # 0 = patched, 1 = not patched
"""

import sys

SENTINEL = "Turing encoder-attention clamp"
MARKER = "# gpu-vllm-t4-embed2: Turing encoder-attention clamp"

# The two lines upstream uses to size the tile. The clamp goes immediately after
# them and before `sm_scale`, the next statement; BLOCK is first read at the grid.
ANCHOR = "    if Lk >= 512:\n        BLOCK = min(BLOCK, 32)\n"

# Identifiers the inserted block reads. Absent means upstream restructured the
# function and the clamp would raise NameError at the first token.
REQUIRED = ("def context_attention_fwd(", "current_platform", "BLOCK = get_block_size(q.dtype)", "o: torch.Tensor")

CLAMP = f"""    {MARKER}
    # head_dim 256 needs BLOCK 16 in fp32; head_dim 512 fits no fp32 tile at all,
    # so that kernel runs in fp16 (fp32 accumulate) and writes back. See the rig.
    _turing_out = None
    if current_platform.is_cuda() and not current_platform.has_device_capability(80):
        if Lk >= 256:
            BLOCK = min(BLOCK, 16)
        if q.dtype == torch.float32 and Lk >= 512:
            _turing_out = o
            q, k, v = q.half(), k.half(), v.half()
            o = torch.empty_like(q)
"""

# The copy-back goes after the launch. Upstream ends the function with the launch
# call, so the anchor is the function's last line, matched with its indentation.
TAIL_ANCHOR = "        Lk=Lk,\n    )\n"
TAIL = """    if _turing_out is not None:
        _turing_out.copy_(o)
"""


def _refuse(msg: str, text: str, around: str) -> None:
    idx = text.find(around)
    ctx = text[max(0, idx - 400) : idx + 800] if idx >= 0 else text[:1200]
    print(f"REFUSED: {msg}\n--- source ---\n{ctx}", file=sys.stderr)
    sys.exit(2)


def is_patched(text: str) -> bool:
    return SENTINEL in text


def patch(text: str) -> str:
    if is_patched(text):
        return text
    for ident in REQUIRED:
        if ident not in text:
            _refuse(
                f"identifier {ident!r} not found; upstream restructured the file", text, "def context_attention_fwd("
            )
    if text.count(ANCHOR) != 1:
        _refuse(f"tile anchor matched {text.count(ANCHOR)} times, expected 1", text, "BLOCK = get_block_size")
    fn = text.index("def context_attention_fwd(")
    if text.count(TAIL_ANCHOR, fn) != 1:
        _refuse(f"launch-end anchor matched {text.count(TAIL_ANCHOR, fn)} times after the function", text, "Lk=Lk")
    text = text.replace(ANCHOR, ANCHOR + CLAMP)
    tail_at = text.index(TAIL_ANCHOR, fn) + len(TAIL_ANCHOR)
    return text[:tail_at] + TAIL + text[tail_at:]


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [a for a in argv if a != "--check"]
    if len(paths) != 1:
        print(__doc__.split("Usage:")[1], file=sys.stderr)
        return 64
    path = paths[0]
    with open(path) as fh:
        text = fh.read()
    if check:
        print("PATCHED" if is_patched(text) else "UNPATCHED")
        return 0 if is_patched(text) else 1
    if is_patched(text):
        print(f"already patched: {path}")
        return 0
    new = patch(text)
    compile(new, path, "exec")
    with open(path, "w") as fh:
        fh.write(new)
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

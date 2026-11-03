# CLAUDE.md — local-litertlm-3050oem-2b-q4w4a16emb4

Gemma 4 E2B in LiteRT-LM on the RTX 3050 OEM, from a `.litertlm` **built here** that holds the QAT Q4_0 grid
exactly: every int4 code and fp16 scale equals `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`. Sister of
`local-litertlm-3050oem-2b-int248` (Google's own `.litertlm`); slot 5 is the only slot that differs. Read that
rig's `CLAUDE.md` for everything about the runtime (WebGPU on Vulkan, protobuf, `max_completion_tokens`, the
`pkill` gotcha); it all applies here. No MCP server.

## Read first

- **`tpu.env` decides.** Port **9380** (the int248 rig is 9379); both fit on the card.
- **There is no published artifact.** `make build` makes it (`build.sh`: tools, convert, repack, pack), about
  20 minutes and ~45 GB RAM at the export's peak on this 46 GB host. Do not run it beside a large job.
- **The converter lives outside the shared python3.** `build.sh tools` installs litert-torch with
  `pip --target` into `CONVERTER_DIR` with an explicit `--no-deps` list, a stand-in tensorflow schema module, and
  the PyPI `tflite` schema. Not a virtualenv, and the system interpreter's packages are untouched: litert-torch's
  `xdsl==0.28.0` pin needs `typing_extensions<4.13`, which would otherwise downgrade the shared one under mcp.
- **`repack_litertlm.py` matches tensors by value, not name.** The converter's tensor names are fused module
  paths; with GPU composites q/k/v and gate/up are fused into one weight. The script offers those
  concatenations as sources, refuses an ambiguous match, refuses a tensor where over 40% of codes would change
  (the converter's own re-rounding changes 6.6–18%), and re-reads every tensor after writing.
- **`--apply_gpu_composites=True` is what makes it fast.** Without it the converted graph decodes at ~45–49
  tok/s whether scales are per row or per block of 32; with it 80+.
- **No drafter.** Google's `.litertlm` carries an MTP drafter section built for its own weights; this build has
  none, so `SPECULATIVE_DECODING=off`.

## Formats (measured)

- compressed-tensors pack-quantized: int32 words, nibble i at bits 4i, stored as code + 8 (codes −8…7).
- TFLite int4: two codes per byte, low nibble first, two's complement. Blockwise quantization
  (`QuantizationDetails` type 2) holds a scale-tensor index and block size; scales are fp16, no zero points.

## Commands

`make install | build | import | serve | status | query | bench | test | lint`. Tests are `unittest`, offline.

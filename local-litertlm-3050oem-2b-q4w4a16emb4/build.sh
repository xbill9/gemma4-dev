#!/bin/bash
# Build the exact QAT Q4_0 .litertlm from source. Values come from tpu.env (the Makefile exports them).
#
#   build.sh tools     litert-torch + pinned deps into CONVERTER_DIR (pip --target, NOT a virtualenv;
#                      the shared python3 is untouched) and a stand-in tensorflow schema module
#   build.sh convert   litert-torch export of QAT_SOURCE: int4 blocks of 32 + GPU composites
#   build.sh repack    write REPACK_SOURCE's exact codes and scales into the converted graphs
#   build.sh pack      bundle with the tokenizer and metadata from Google's .litertlm
#   build.sh all       all four
set -euo pipefail
cd "$(dirname "$0")"
RIG=$PWD
W=$BUILD_DIR

tools() {
  mkdir -p "$CONVERTER_DIR"
  # --no-deps with an explicit list: a plain --target install pulls a second torch, CUDA wheels and
  # transformers. xdsl 0.28.0 is litert-converter's pin and needs typing_extensions < 4.13 and
  # immutabledict < 4.2.2; those older copies live only in CONVERTER_DIR.
  python3 -m pip install -q --no-deps --target "$CONVERTER_DIR" \
    litert-torch==0.10.0 ai-edge-litert==2.3.0 ai-edge-quantizer==0.10.1 litert-converter==0.5.0 \
    xdsl==0.28.0 typing_extensions==4.12.2 immutabledict==4.2.1 ordered-set==4.1.0 \
    jax==0.12.0.1 jaxlib==0.12.0.1 torchao==0.18.0 \
    backports.strenum fire jaxtyping kagglehub kagglesdk lark ml_dtypes multipledispatch opt_einsum \
    tabulate termcolor wadler_lindig
  # litert_torch's bonsai_flux2 model imports tensorflow's TFLite schema at import time; point it at
  # ai_edge_litert's copy instead of installing TensorFlow.
  local sh=$CONVERTER_DIR-tfshim/tensorflow/lite/python
  mkdir -p "$sh"
  touch "$CONVERTER_DIR-tfshim/tensorflow/__init__.py" "$CONVERTER_DIR-tfshim/tensorflow/lite/__init__.py" "$sh/__init__.py"
  printf 'from ai_edge_litert.schema_py_generated import *  # noqa: F401,F403\n' > "$sh/schema_py_generated.py"
  # The PyPI tflite schema, for reading and editing the graphs.
  python3 -m pip install -q --no-deps --target "$CONVERTER_DIR-tflite" tflite==2.18.0
}

convert() {
  local src
  src=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('$QAT_SOURCE', revision='$QAT_SOURCE_REVISION', allow_patterns=['*.json','*.safetensors','*.jinja']))")
  mkdir -p "$W/convert"
  # Exit status ignored on purpose: litert-torch 0.10.0's own bundling step fails against
  # litert-lm-builder 0.19.0 (no TfLiteModelType) AFTER writing the three *_quantized.tflite we use.
  CUDA_VISIBLE_DEVICES= PYTHONPATH="$CONVERTER_DIR:$CONVERTER_DIR-tfshim" \
    python3 -m litert_torch.generative.export_hf --model="$src" --output_dir="$W/convert" \
    --quantization_recipe=dynamic_wi4b32_afp32 --externalize_embedder=True --apply_gpu_composites=True \
    --bundle_litert_lm=True --keep_temporary_files=True > "$W/convert/export.log" 2>&1 || true
  for f in embedder per_layer_embedder model; do
    test -s "$W/convert/${f}_quantized.tflite" || { echo "export produced no ${f}_quantized.tflite; see $W/convert/export.log"; exit 1; }
    rm -f "$W/convert/$f.tflite"   # the float graphs: 20 GB, not needed
  done
}

repack() {
  local src
  src=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('$REPACK_SOURCE', revision='$REPACK_SOURCE_REVISION', allow_patterns=['*.json','*.safetensors']))")
  CUDA_VISIBLE_DEVICES= python3 -P "$RIG/repack_litertlm.py" "$CONVERTER_DIR-tflite" "$src" "$W/exact" \
    "$W/convert/embedder_quantized.tflite" "$W/convert/per_layer_embedder_quantized.tflite" "$W/convert/model_quantized.tflite"
}

pack() {
  local p=$W/pack
  mkdir -p "$p/google"
  echo "$TOKENIZER_FROM_SHA256  $TOKENIZER_FROM" | sha256sum -c --quiet -
  litert-lm-peek --litertlm_file "$TOKENIZER_FROM" --dump_files_dir "$p/google" > /dev/null
  cp "$RIG/pack/model.toml" "$p/"
  cp "$p/google/LlmMetadataProto.pbtext" "$p/google/Section1_SP_Tokenizer.spiece" "$p/"
  rm -rf "$p/google"
  ln -sf "$W/exact/embedder_quantized.tflite" "$p/embedder.tflite"
  ln -sf "$W/exact/per_layer_embedder_quantized.tflite" "$p/per_layer_embedder.tflite"
  ln -sf "$W/exact/model_quantized.tflite" "$p/prefill_decode.tflite"
  mkdir -p "$(dirname "$MODEL_PATH")"
  (cd "$p" && litert-lm pack model.toml --output "$MODEL_PATH" --allow-overwrite)
  sha256sum "$MODEL_PATH"
  echo "expected $MODEL_SHA256 (tpu.env); a different hash means a different build"
}

case "${1:-}" in
  tools|convert|repack|pack) "$1" ;;
  all) tools; convert; repack; pack ;;
  *) echo "usage: build.sh tools|convert|repack|pack|all"; exit 2 ;;
esac

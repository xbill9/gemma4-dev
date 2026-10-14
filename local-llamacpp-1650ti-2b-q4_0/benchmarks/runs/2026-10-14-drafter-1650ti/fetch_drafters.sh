#!/bin/bash
# Fetch the exact Q4_0 Gemma 4 drafters from Hugging Face at pinned revisions; verify SHA-256.
# A file already present with the right hash is kept.
set -eu
get() {  # model revision sha256
  dir=$HOME/models/gemma-4-$1-it-qat-assistant
  f=$dir/gemma-4-$1-it-qat-assistant-q4_0-exact.gguf
  mkdir -p "$dir"
  if ! echo "$3  $f" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fL -o "$f.part" "https://huggingface.co/xbill9/gemma-4-$1-it-qat-assistant-q4_0-exact-gguf/resolve/$2/gemma-4-$1-it-qat-assistant-q4_0-exact.gguf"
    mv "$f.part" "$f"
    echo "$3  $f" | sha256sum -c -
  else
    echo "$f: OK (already present)"
  fi
}
get E2B 6188294 060f5a0158f8120d25137866b90e04e21cd0d2d23de1db7a9cc3b7f00e60bd11
get E4B a946809 dc965cb963e56b86fbc04db8efec83163886085e88514574650a57fa55a2ee24

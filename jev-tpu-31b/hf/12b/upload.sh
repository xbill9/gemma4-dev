#!/bin/bash
# upload.sh [--public] -- publish the repacked 12B QAT W4A16 checkpoint to the logged-in Hugging Face
# account (run `hf auth login` first, with a write token). The repo is created private unless
# --public is given; review it on the Hub, then flip it public there. MODEL_CARD.md beside this
# script is the card; the checkpoint directory's README.md must match it.
set -eu
SRC=${SRC:-$HOME/models/gemma-4-12B-it-qat-q4_0-w4a16-ct}
NAME=gemma-4-12B-it-qat-q4_0-w4a16-ct
PRIVATE=--private; [ "${1:-}" = --public ] && PRIVATE=
cmp "$SRC/README.md" "$(dirname "$0")/MODEL_CARD.md" || { echo "README.md in $SRC differs from MODEL_CARD.md"; exit 1; }
USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")
REPO="$USER/$NAME"
hf repo create "$REPO" --repo-type model $PRIVATE --exist-ok
hf upload "$REPO" "$SRC" . --repo-type model --commit-message "Gemma 4 12B-it QAT as compressed-tensors W4A16 (unofficial repack)"
echo "uploaded: https://huggingface.co/$REPO"

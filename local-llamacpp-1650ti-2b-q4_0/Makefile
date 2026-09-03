# local-llamacpp-1650ti-2b-q4_0 — local llama.cpp rig, GTX 1650 Ti, Gemma 4 E2B q4_0.
#
# There are no provisioning targets here and none is owed: the card is in this
# machine. `make serve` is the whole of what `make deploy` means elsewhere, and
# teardown is Ctrl-C. See NAMING.md, "`local` is the absence of a control plane".
#
# Values come from tpu.env, which is the source of truth. ?= so a real
# environment variable always wins.
include tpu.env
export

SKILL_NAME ?= $(notdir $(CURDIR))-management

.PHONY: all help install test lint clean serve stop status query info bench-validate

all: help

help:
	@echo "========================================================="
	@echo " local-llamacpp-1650ti-2b-q4_0 — GTX 1650 Ti, Gemma 4 E2B q4_0"
	@echo "========================================================="
	@echo "  make install       - pip install -r requirements.txt (system python3)"
	@echo "  make info          - resident-vs-lazy split, read off the artifact"
	@echo "  make serve         - run llama-server in the foreground"
	@echo "  make status        - is it up?"
	@echo "  make query         - one chat completion against the local endpoint"
	@echo "  make test          - offline unittest suite"
	@echo "  make lint          - ruff on the module list below"
	@echo "  make clean         - remove caches"
	@echo "========================================================="
	@echo "Memory: only ~1.31 GiB of the 3.35 GB artifact is resident."
	@echo "per_layer_token_embd (58% of the file) is TENSOR_READ_LAZY and stays"
	@echo "in the mmap. Never pass --no-mmap: it breaks that and forces an OOM."

install:
	python3 -m pip install -r requirements.txt

info:
	python3 inspect_gguf.py

# Foreground on purpose. Ctrl-C is a complete teardown and costs nothing, so the
# backgrounding and pid-tracking a cloud rig needs would be ceremony here. The
# MCP server has start/stop tools for the case where you do want it detached.
serve:
	$(LLAMA_SERVER_BIN) \
	  -m $(MODEL_PATH) \
	  --host $(HOST) --port $(PORT) \
	  -ngl $(N_GPU_LAYERS) \
	  -c $(CONTEXT_SIZE) \
	  -ctk $(KV_CACHE_TYPE) -ctv $(KV_CACHE_TYPE) \
	  -fa $(FLASH_ATTENTION) \
	  -t $(THREADS) \
	  --parallel $(PARALLEL_SLOTS)

status:
	@curl -fsS $(ENDPOINT)/health && echo "  <- healthy" || echo "not serving at $(ENDPOINT)"

# /v1/chat/completions, NOT /v1/completions: raw completions return an empty
# string on -it checkpoints, which reads as a broken deploy and is not one.
query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_tokens":128}' \
	  | python3 -m json.tool

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py inspect_gguf.py sweep.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/runs

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# local-ollama-1650ti-2b-q4_0 — local Ollama rig, GTX 1650 Ti, Gemma 4 E2B q4_0.
#
# There are no provisioning targets here and none is owed: the card is in this
# machine. `make serve` is the whole of what `make deploy` means elsewhere, and
# teardown is Ctrl-C. See NAMING.md, "`local` is the absence of a control plane".
#
# Values come from tpu.env, which is the source of truth. ?= so a real
# environment variable always wins.
include tpu.env
export

.PHONY: all help install test lint clean serve pull status query info residency bench-validate

all: help

help:
	@echo "========================================================="
	@echo " local-ollama-1650ti-2b-q4_0 — GTX 1650 Ti, Gemma 4 E2B q4_0"
	@echo "========================================================="
	@echo "  make install     - pip install -r requirements.txt (system python3)"
	@echo "  make pull        - fetch the stock tag into the local blob store (4.3 GB)"
	@echo "  make text-only   - rebuild the projector-free variant (saves 1150 MiB)"
	@echo "  make serve       - run the Ollama daemon in the foreground"
	@echo "  make status      - is it up?"
	@echo "  make info        - what Ollama thinks the tag is"
	@echo "  make residency   - is the model fully on the GPU, and what does the driver say"
	@echo "  make query       - one chat request against the local endpoint"
	@echo "  make test        - offline unittest suite"
	@echo "  make lint        - ruff on the module list below"
	@echo "  make clean       - remove caches"
	@echo "========================================================="
	@echo "The STOCK tag loads a 986 MB vision+audio projector unconditionally:"
	@echo "2762 MiB resident against the llama.cpp sibling's 1618, and ollama ps"
	@echo "does not count it. MODEL_NAME is the projector-free variant: 1612 MiB."
	@echo "Use 'make residency', not 'ollama ps'."

install:
	python3 -m pip install -r requirements.txt

pull:
	OLLAMA_HOST=$(HOST):$(PORT) $(OLLAMA_BIN) pull $(MODEL_STOCK_TAG)

# Build the text-only variant: the SAME Ollama blob with the projector layer
# dropped. Worth 1150 MiB on a 4096 MiB card, and it is what MODEL_NAME points
# at. Re-run after `make pull` on a fresh machine; the model blob is shared, so
# this costs no extra disk.
text-only:
	OLLAMA_HOST=$(HOST):$(PORT) $(OLLAMA_BIN) create $(MODEL_NAME) -f modelfiles/text-only.Modelfile
	@echo "--- capabilities (vision/audio should be ABSENT):"
	@OLLAMA_HOST=$(HOST):$(PORT) $(OLLAMA_BIN) show $(MODEL_NAME) | sed -n '/Capabilities/,/^$$/p'

# Foreground on purpose. Ctrl-C is a complete teardown and costs nothing, so the
# backgrounding and pid-tracking a cloud rig needs would be ceremony here. The
# MCP server has start/stop tools for the case where you do want it detached.
#
# EVERY VARIABLE BELOW IS LOAD-BEARING. OLLAMA_LLM_LIBRARY pins native sm_75 SASS
# (unpinned, the CUDA 13 driver selects cuda_v13 and may JIT from PTX at load).
# OLLAMA_KEEP_ALIVE stops the 5-minute idle unload from being recorded as sweep
# latency. OLLAMA_CONTEXT_LENGTH is PER SEQUENCE and is MULTIPLIED by
# OLLAMA_NUM_PARALLEL — llama.cpp DIVIDES its -c across slots, so the equivalent
# of the sibling's `-c 32768 --parallel 32` is 1024 x 32, not 32768 x 32. A
# MULTIMODAL model additionally gets a 2048-token minimum per slot (measured at
# 1024 and 1536); MODEL_NAME is projector-free, so these values are taken
# literally. Read the child's -c off run/ollama.log rather than predicting it.
serve:
	OLLAMA_HOST=$(HOST):$(PORT) \
	OLLAMA_LLM_LIBRARY=$(OLLAMA_LLM_LIBRARY) \
	OLLAMA_CONTEXT_LENGTH=$(OLLAMA_CONTEXT_LENGTH) \
	OLLAMA_NUM_PARALLEL=$(OLLAMA_NUM_PARALLEL) \
	OLLAMA_KEEP_ALIVE=$(OLLAMA_KEEP_ALIVE) \
	OLLAMA_MODELS=$(OLLAMA_MODELS) \
	  $(OLLAMA_BIN) serve

status:
	@curl -fsS $(ENDPOINT)/api/version && echo "  <- up" || echo "not serving at $(ENDPOINT)"

info:
	@curl -fsS $(ENDPOINT)/api/show -d '{"model":"$(MODEL_NAME)"}' | python3 -m json.tool | head -40

# `ollama ps` alone is not enough: its SIZE excludes the projector. Read the
# driver too.
residency:
	@curl -fsS $(ENDPOINT)/api/ps | python3 -m json.tool
	@nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# /api/chat with think enabled. THE DEFAULT DISCARDS THE THINKING BLOCK: at a
# small num_predict the block has not closed and BOTH content and thinking come
# back empty, which reads as a broken daemon and is not one.
query:
	@curl -fsS $(ENDPOINT)/api/chat \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"stream":false,"think":true,"options":{"num_predict":1024}}' \
	  | python3 -m json.tool

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py sweep.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/runs

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

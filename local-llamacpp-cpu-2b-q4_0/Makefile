# local-llamacpp-cpu-2b-q4_0 — local llama.cpp rig, CPU only, Gemma 4 E2B q4_0.
#
# There are no provisioning targets here and none is owed: the CPU is in this
# machine. `make serve` is the whole of what `make deploy` means elsewhere, and
# teardown is Ctrl-C. See NAMING.md, "`local` is the absence of a control plane".
#
# Values come from tpu.env, which is the source of truth. ?= so a real
# environment variable always wins.
include tpu.env
export

SKILL_NAME ?= $(notdir $(CURDIR))-management
LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp
MODEL_DIR = $(dir $(MODEL_PATH))

.PHONY: all help install build download test lint clean serve stop status metrics query info bench-validate

all: help

help:
	@echo "========================================================="
	@echo " local-llamacpp-cpu-2b-q4_0 — CPU only, Gemma 4 E2B q4_0"
	@echo "========================================================="
	@echo "  make install       - pip install -r requirements.txt (system python3)"
	@echo "  make build         - CPU-only llama.cpp build (GGML_CUDA=OFF) at build-cpu/"
	@echo "  make download      - fetch the GGUF into MODEL_PATH's directory"
	@echo "  make info          - resident-vs-lazy split, read off the artifact"
	@echo "  make serve         - run llama-server in the foreground, no GPU"
	@echo "  make status        - is it up?"
	@echo "  make metrics       - llama.cpp prometheus metrics (needs METRICS=1)"
	@echo "  make query         - one chat completion against the local endpoint"
	@echo "  make test          - offline unittest suite"
	@echo "  make lint          - ruff on the module list below"
	@echo "  make clean         - remove caches"
	@echo "========================================================="
	@echo "Never pass --no-mmap: per_layer_token_embd (58% of the file) is"
	@echo "TENSOR_READ_LAZY and requires mmap."

install:
	python3 -m pip install -r requirements.txt

# A separate build directory from any GPU build, so the two cannot be confused.
# GGML_NATIVE=ON targets this host's AVX2/AVX-VNNI; the binary is not portable.
# cmake is pip-installable into the system python3 if it is missing.
build:
	@command -v cmake >/dev/null || { echo "cmake not found; install with: pip install cmake"; exit 1; }
	@test -d $(LLAMA_CPP_DIR) || git clone $(LLAMA_CPP_REPO) $(LLAMA_CPP_DIR)
	cmake -S $(LLAMA_CPP_DIR) -B $(LLAMA_CPP_DIR)/build-cpu -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build $(LLAMA_CPP_DIR)/build-cpu --config Release -j --target llama-server llama-bench
	@echo "built at commit $$(git -C $(LLAMA_CPP_DIR) rev-parse --short HEAD) — record it as LLAMA_CPP_COMMIT in tpu.env"

# This repo is PUBLIC, unlike most Google checkpoints: on 2026-09-16 `hf download`
# fetched the GGUF unauthenticated, with no licence acceptance and no `hf auth
# login`. A token only buys rate limits and faster downloads here.
download:
	@command -v hf >/dev/null || { echo "hf not found; install with: pip install huggingface_hub"; exit 1; }
	hf download $(MODEL_NAME) $(notdir $(MODEL_PATH)) --local-dir $(MODEL_DIR)

info:
	python3 inspect_gguf.py

# Foreground on purpose. Ctrl-C is a complete teardown and costs nothing.
# -ngl 0 and the empty CUDA_VISIBLE_DEVICES are NOT tpu.env values: this rig is
# CPU only whatever binary or environment it is pointed at.
serve:
	CUDA_VISIBLE_DEVICES= $(LLAMA_SERVER_BIN) \
	  -m $(MODEL_PATH) \
	  --host $(HOST) --port $(PORT) \
	  -ngl 0 \
	  -c $(CONTEXT_SIZE) \
	  -ctk $(KV_CACHE_TYPE) -ctv $(KV_CACHE_TYPE) \
	  -fa $(FLASH_ATTENTION) \
	  -t $(THREADS) \
	  -tb $(THREADS_BATCH) \
	  --parallel $(PARALLEL_SLOTS) \
	  $(if $(filter 1,$(METRICS)),--metrics,)

status:
	@curl -fsS $(ENDPOINT)/health && echo "  <- healthy" || echo "not serving at $(ENDPOINT)"

# llama.cpp's own Prometheus endpoint, not ours. 501 here means the server was
# started without --metrics, not that it is unhealthy. Operational questions
# only, NOT the benchmark decode column.
metrics:
	@curl -fsS $(ENDPOINT)/metrics || echo "no /metrics — server started without --metrics (METRICS=1 in tpu.env)"

# /v1/chat/completions, NOT /v1/completions: raw completions return an empty
# string on -it checkpoints. max_tokens is 1024 and must stay >= 512: Gemma 4
# reasons first, llama.cpp routes that to reasoning_content, and content stays
# empty until the block closes. Expect this to be slow on a CPU.
query:
	@curl -fsS --max-time 900 $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_tokens":1024}' \
	  | python3 -m json.tool

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py attest.py inspect_gguf.py sweep.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/runs

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

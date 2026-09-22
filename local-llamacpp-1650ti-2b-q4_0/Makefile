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
LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp

# The card is sm_75 (Turing, GTX 1650 Ti). Build for exactly that: this binary is
# not portable and is not meant to be. FORCE_MMQ stays OFF -- see tpu.env for the
# build-mmq experiment.
CUDA_ARCH ?= 75

# NEVER pass a bare `-j`. `cmake --build -j` with no value hands make an UNBOUNDED
# job count; llama.cpp has ~188 CUDA translation units and each nvcc forks cicc and
# cudafe++, so an unbounded CUDA build reliably OOMs this 15 GiB host. That is what
# happened on 2026-09-22 (dmesg victims: cicc). 6 is deliberately BELOW nproc for
# that reason -- MEASURED the same day, a full CUDA rebuild at -j6 peaks at 4.0 GiB
# and never touches swap, in 10m09. Override with BUILD_JOBS=N if you know better.
BUILD_JOBS ?= 6

.PHONY: all help install build test lint clean serve stop status metrics query info bench-validate

all: help

help:
	@echo "========================================================="
	@echo " local-llamacpp-1650ti-2b-q4_0 — GTX 1650 Ti, Gemma 4 E2B q4_0"
	@echo "========================================================="
	@echo "  make install       - pip install -r requirements.txt (system python3)"
	@echo "  make build         - build llama.cpp with CUDA (sm_$(CUDA_ARCH), -j$(BUILD_JOBS))"
	@echo "  make info          - resident-vs-lazy split, read off the artifact"
	@echo "  make serve         - run llama-server in the foreground"
	@echo "  make status        - is it up?"
	@echo "  make metrics       - llama.cpp prometheus metrics (needs METRICS=1)"
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

# Until 2026-09-22 this rig had NO build target and its only record of these flags
# was the CMakeCache.txt of whatever was last built -- which is not a record, it is
# an artifact. They live here now.
#
# Separate build directory from the CPU arm's build-cpu/ so the two cannot be
# confused; the CPU arm refuses a binary with a GPU backend beside it.
#
# After a toolchain change, DELETE the build dir first: a stale CMakeCache.txt keeps
# the old compiler, and /usr/local/cuda being a symlink hides that from the cached
# path. Both arms were reconfigured from clean on 2026-09-22 for exactly that.
build:
	@command -v cmake >/dev/null || { echo "cmake not found; install with: pip install cmake"; exit 1; }
	@command -v nvcc >/dev/null || { echo "nvcc not found; CUDA toolkit missing"; exit 1; }
	@test -d $(LLAMA_CPP_DIR) || git clone $(LLAMA_CPP_REPO) $(LLAMA_CPP_DIR)
	cmake -S $(LLAMA_CPP_DIR) -B $(LLAMA_CPP_DIR)/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build $(LLAMA_CPP_DIR)/build --config Release -j$(BUILD_JOBS) --target llama-server llama-bench
	@echo "built at commit $$(git -C $(LLAMA_CPP_DIR) rev-parse --short HEAD) — record it as LLAMA_CPP_COMMIT in tpu.env"

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
	  -tb $(THREADS_BATCH) \
	  --parallel $(PARALLEL_SLOTS) \
	  $(if $(filter 1,$(METRICS)),--metrics,)

status:
	@curl -fsS $(ENDPOINT)/health && echo "  <- healthy" || echo "not serving at $(ENDPOINT)"

# llama.cpp's own Prometheus endpoint, not ours. 501 here means the server was
# started without --metrics, not that it is unhealthy. See tpu.env, METRICS —
# and note it is for operational questions, NOT for the benchmark decode column.
metrics:
	@curl -fsS $(ENDPOINT)/metrics || echo "no /metrics — server started without --metrics (METRICS=1 in tpu.env)"

# /v1/chat/completions, NOT /v1/completions: raw completions return an empty
# string on -it checkpoints, which reads as a broken deploy and is not one.
#
# max_tokens is 1024 and must stay >= 512. There is a SECOND empty-reply path
# here that hits /v1/chat/completions, the endpoint you were told to use
# instead: Gemma 4 emits a thinking block, llama.cpp routes it to
# reasoning_content, and content stays empty until the block closes. MEASURED
# 2026-09-08 on this very prompt -- 1556 chars of reasoning and 378 completion
# tokens to answer "in one sentence, what is a TPU?". At the old 128 this
# target returned content "" with finish_reason "length" on a healthy server.
# Matches query_model's default in server.py. See CLAUDE.md, "Gemma 4 reasons".
query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions \
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

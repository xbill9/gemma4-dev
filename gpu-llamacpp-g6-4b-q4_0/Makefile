# gpu-llamacpp-g6-4b-q4_0 — llama.cpp, exact Q4_0 E4B + exact Q4_0 QAT drafter, NVIDIA L4 (EC2 G6).
#
# Run these ON the instance (launched with gpu-pytorch-g6-2b-q4w4a16's create_g6_instance).
# tpu.env decides every value; a real environment variable wins (?=).
include tpu.env
export

LLAMA_CPP_REPO ?= https://github.com/ggml-org/llama.cpp
BUILD_JOBS ?= 8
PY ?= $(shell cat /opt/torch-g6-q4/PYTHON_BIN 2>/dev/null || echo python3)

.PHONY: help toolkit build download draft serve status query

help:
	@echo "make toolkit   - apt install the CUDA 13.1 compiler and cuBLAS headers (driver supports 13.2)"
	@echo "make build     - clone llama.cpp at LLAMA_CPP_COMMIT, build llama-server/llama-bench for sm_$(CUDA_ARCH)"
	@echo "make download  - exact Q4_0 E2B GGUF at MODEL_REVISION, SHA-256 checked"
	@echo "make draft     - exact Q4_0 QAT drafter at SPEC_DRAFT_HF_REVISION, SHA-256 checked"
	@echo "make serve     - foreground llama-server with the drafter (empty SPEC_DRAFT_MODEL to turn it off)"

toolkit:
	apt-get -o DPkg::Lock::Timeout=600 install -y cmake cuda-nvcc-13-1 cuda-cudart-dev-13-1 libcublas-dev-13-1 cuda-nvrtc-dev-13-1

build:
	@test -d $(LLAMA_CPP_DIR) || git clone $(LLAMA_CPP_REPO) $(LLAMA_CPP_DIR)
	git -C $(LLAMA_CPP_DIR) fetch -q origin && git -C $(LLAMA_CPP_DIR) checkout -q $(LLAMA_CPP_COMMIT)
	cmake -S $(LLAMA_CPP_DIR) -B $(LLAMA_CPP_DIR)/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
	  -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$(CUDA_HOME)/bin/nvcc
	cmake --build $(LLAMA_CPP_DIR)/build --config Release -j$(BUILD_JOBS) --target llama-server llama-bench

download:
	$(PY) -c "from huggingface_hub import hf_hub_download as d; d('$(MODEL_NAME)', '$(notdir $(MODEL_PATH))', revision='$(MODEL_REVISION)', local_dir='$(dir $(MODEL_PATH))')"
	echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c -

draft:
	$(PY) -c "from huggingface_hub import hf_hub_download as d; d('$(SPEC_DRAFT_HF)', '$(notdir $(SPEC_DRAFT_MODEL))', revision='$(SPEC_DRAFT_HF_REVISION)', local_dir='$(dir $(SPEC_DRAFT_MODEL))')"
	echo "$(SPEC_DRAFT_SHA256)  $(SPEC_DRAFT_MODEL)" | sha256sum -c -

serve:
	$(LLAMA_SERVER_BIN) -m $(MODEL_PATH) --host $(HOST) --port $(PORT) -ngl $(N_GPU_LAYERS) -c $(CONTEXT_SIZE) \
	  -ctk $(KV_CACHE_TYPE) -ctv $(KV_CACHE_TYPE) -fa $(FLASH_ATTENTION) -t $(THREADS) -tb $(THREADS_BATCH) \
	  --parallel $(PARALLEL_SLOTS) --reasoning $(REASONING) $(if $(filter 1,$(METRICS)),--metrics,) \
	  $(if $(OVERRIDE_TENSORS),-ot '$(OVERRIDE_TENSORS)',) \
	  $(if $(SPEC_DRAFT_MODEL),-md $(SPEC_DRAFT_MODEL) --spec-type draft-mtp -ngld all --spec-draft-n-max $(SPEC_DRAFT_N_MAX),)

status:
	@curl -fsS http://127.0.0.1:$(PORT)/health && echo "  <- healthy" || echo "not serving on :$(PORT)"

query:
	@curl -fsS http://127.0.0.1:$(PORT)/v1/chat/completions -H 'Content-Type: application/json' \
	  -d '{"messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_tokens":1024}' | $(PY) -m json.tool

# gpu-litertlm-g6-2b-int248 — LiteRT-LM on one NVIDIA L4 (EC2 G6), Gemma 4 E2B.
#
# Run ON the instance (gpu-pytorch-g6-2b-q4w4a16's create_g6_instance; SSM runs as root). Values come
# from tpu.env, exported to every recipe; override one on the make command line.
include tpu.env
export

CONFIG := run/config.json
PIP = $(PYTHON) -m pip install --user --break-system-packages

.PHONY: all help vulkan install download import config serve status query bench test lint

all: help

help:
	@echo "  make vulkan    - apt libvulkan1 + vulkan-tools; list Vulkan devices (LiteRT-LM's GPU path)"
	@echo "  make install   - litert-lm + protobuf into the system python3 user site"
	@echo "  make download  - the .litertlm at MODEL_HF_REVISION, SHA-256 checked"
	@echo "  make import    - register MODEL_PATH as MODEL_ID in LITERT_LM_DIR"
	@echo "  make serve     - OpenAI-compatible server on PORT, foreground"
	@echo "  make status | query | bench | test | lint"

# Fails unless a Vulkan device named like the L4 enumerates. UNVERIFIED on the DLAMI.
vulkan:
	apt-get -o DPkg::Lock::Timeout=600 install -y libvulkan1 vulkan-tools
	ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || true
	vulkaninfo --summary 2>/dev/null | grep -i 'deviceName' | tee /dev/stderr | grep -q 'L4'

install:
	$(PIP) -r requirements.txt

download:
	@mkdir -p $(dir $(MODEL_PATH))
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c --quiet - 2>/dev/null || \
	  curl -fL -o $(MODEL_PATH).part https://huggingface.co/$(MODEL_NAME)/resolve/$(MODEL_HF_REVISION)/$(MODEL_FILE)
	@test ! -f $(MODEL_PATH).part || mv $(MODEL_PATH).part $(MODEL_PATH)
	echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c -

import:
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c -
	@test -f $(LITERT_LM_DIR)/models/$(MODEL_ID)/model.litertlm && echo "already imported as $(MODEL_ID)" || \
	  litert-lm import $(MODEL_PATH) $(MODEL_ID)

config:
	$(PYTHON) render_config.py $(CONFIG)

serve: config
	litert-lm serve --config $(CONFIG) --host $(HOST) --port $(PORT)

status:
	@curl -fsS $(ENDPOINT)/v1/models >/dev/null && echo "serving at $(ENDPOINT)" || echo "not serving at $(ENDPOINT)"

# max_completion_tokens: the server ignores max_tokens.
query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_ID)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_completion_tokens":256}' \
	  | $(PYTHON) -m json.tool

bench:
	litert-lm benchmark --backend $(BACKEND) --speculative-decoding $(SPECULATIVE_DECODING) -p 1024 -d 256 --runs 3 $(MODEL_ID)

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	ruff check render_config.py tests

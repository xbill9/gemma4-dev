# local-litertlm-3050oem-2b-int248 — LiteRT-LM on the RTX 3050 OEM, Gemma 4 E2B.
#
# No provisioning targets and none owed: the card is in this machine (NAMING.md,
# "`local` is the absence of a control plane"). `make serve` is the deploy and
# Ctrl-C is the teardown. Values come from tpu.env and are exported to every
# recipe, LITERT_LM_DIR included. Override one on the command line, which beats
# an `include`d assignment: `make serve SPECULATIVE_DECODING=off`. A plain
# environment variable does NOT win inside make (that needs `make -e`);
# render_config.py run on its own does honour os.environ.
include tpu.env
export

CONFIG := run/config.json

.PHONY: all help install download import config serve status query bench test lint clean

all: help

help:
	@echo "========================================================="
	@echo " local-litertlm-3050oem-2b-int248 — RTX 3050 OEM, Gemma 4 E2B, LiteRT-LM"
	@echo "========================================================="
	@echo "  make install   - pip install -r requirements.txt (system python3, user site)"
	@echo "  make download  - fetch the .litertlm to MODEL_PATH and check MODEL_SHA256"
	@echo "  make import    - register it under MODEL_ID in LITERT_LM_DIR (a hard link)"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status    - is it up?"
	@echo "  make query     - one chat completion"
	@echo "  make bench     - litert-lm benchmark on the GPU, 1024 in / 256 out"
	@echo "  make test      - offline unittest suite"
	@echo "  make lint      - ruff"
	@echo "========================================================="
	@echo "GPU path is WebGPU on Vulkan, not CUDA. See CLAUDE.md."

install:
	python3 -m pip install --user --break-system-packages -r requirements.txt

download:
	@mkdir -p $(dir $(MODEL_PATH))
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c --quiet - 2>/dev/null || \
	  curl -fL -o $(MODEL_PATH).part https://huggingface.co/$(MODEL_NAME)/resolve/$(MODEL_HF_REVISION)/$(MODEL_FILE)
	@test ! -f $(MODEL_PATH).part || mv $(MODEL_PATH).part $(MODEL_PATH)
	echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c -

import:
	@test -f $(LITERT_LM_DIR)/models/$(MODEL_ID)/model.litertlm && echo "already imported as $(MODEL_ID)" || \
	  litert-lm import $(MODEL_PATH) $(MODEL_ID)

config:
	python3 render_config.py $(CONFIG)

serve: config
	litert-lm serve --config $(CONFIG) --host $(HOST) --port $(PORT)

status:
	@curl -fsS $(ENDPOINT)/v1/models >/dev/null && echo "serving at $(ENDPOINT)" || echo "not serving at $(ENDPOINT)"

query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_ID)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_completion_tokens":256}' \
	  | python3 -m json.tool

# The engine's own synthetic benchmark. Its decode figure with the drafter on is
# NOT a chat speed: acceptance depends on the text. See the first-light REPORT.md.
bench:
	litert-lm benchmark --backend $(BACKEND) --speculative-decoding $(SPECULATIVE_DECODING) \
	  -p 1024 -d 256 --runs 3 $(MODEL_ID)

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check render_config.py tests benchmarks/runs/*/*.py
	@echo "lint OK"

clean:
	rm -rf __pycache__ .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

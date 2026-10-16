# local-litertlm-i71360p-2b-q4wi4b32 — local LiteRT-LM rig, CPU only (i7-1360P), Gemma 4
# E2B on the QAT 4-bit grid.
#
# No provisioning targets, and none is owed: the CPU is in this machine. `make serve`
# is the whole of what `make deploy` means elsewhere; teardown is Ctrl-C.
#
# Values come from tpu.env, which is the source of truth. ?= so a real environment
# variable always wins.
include tpu.env
export

SKILL_NAME ?= $(notdir $(CURDIR))-management
MODEL_DIR = $(dir $(MODEL_PATH))
BUILD_DIR ?= $(HOME)/models/work/$(notdir $(CURDIR))
CONFIG_FILE = run/litertlm-config.json

.PHONY: all help install source model config info serve status query bench test lint clean

all: help

help:
	@echo "========================================================="
	@echo " local-litertlm-i71360p-2b-q4wi4b32 — CPU only, Gemma 4 E2B QAT int4"
	@echo "========================================================="
	@echo "  make install  - pip install -r requirements.txt (system python3)"
	@echo "  make source   - fetch the QAT checkpoint at SRC_REVISION"
	@echo "  make model    - build MODEL_PATH from it (build_litertlm.py)"
	@echo "  make info     - storage types and signatures, read off the artifact"
	@echo "  make serve    - litert-lm serve in the foreground, CPU backend"
	@echo "  make status   - is it up?"
	@echo "  make query    - one chat completion against the local endpoint"
	@echo "  make bench    - litert-lm benchmark, 512 prefill / 128 decode"
	@echo "  make test     - offline unittest suite"
	@echo "  make lint     - ruff on the module list below"
	@echo "========================================================="

install:
	python3 -m pip install -r requirements.txt

source:
	hf download $(SRC_MODEL) --revision $(SRC_REVISION) --local-dir $(SRC_DIR)

# Writes the bundle to BUILD_DIR, then moves it into place. build_report.json (grid
# coverage per quantized tensor) stays beside it in BUILD_DIR.
model:
	python3 build_litertlm.py $(SRC_DIR) $(BUILD_DIR) --cache-length $(CACHE_LENGTH) --prefill $(PREFILL_LENGTHS) --reuse $(BUILD_DIR)/reuse
	mkdir -p $(MODEL_DIR)
	mv $$(ls -t $(BUILD_DIR)/*.litertlm | head -1) $(MODEL_PATH)
	sha256sum $(MODEL_PATH)

info:
	@d=$$(mktemp -d); litert-lm unpack --output-dir $$d/u $(MODEL_PATH) >/dev/null && python3 inspect_tflite.py $$d/u; rm -rf $$d

# Same JSON as server.py's render_config(); a test keeps them agreeing. "cpu" is
# written here, not read from tpu.env.
config:
	@mkdir -p run
	@python3 -c 'import server, json; print(json.dumps(server.render_config(), indent=1))' > $(CONFIG_FILE)
	@cat $(CONFIG_FILE)

serve: config
	$(LITERT_LM_BIN) serve --config $(CONFIG_FILE) --host $(HOST) --port $(PORT)

status:
	@curl -fsS $(ENDPOINT)/v1/models >/dev/null && echo "serving at $(ENDPOINT)" || echo "not serving at $(ENDPOINT)"

# The model loads on the first request, so the first query after `make serve` is slow.
query:
	@curl -fsS --max-time 900 $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_PATH)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_tokens":256,"reasoning_effort":"none"}' \
	  | python3 -m json.tool

bench:
	$(LITERT_LM_BIN) benchmark $(MODEL_PATH) --backend cpu --cpu-thread-count $(CPU_THREAD_COUNT) -p 512 -d 128

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py attest.py build_litertlm.py inspect_tflite.py tests
	@echo "lint OK"

clean:
	rm -rf __pycache__ .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

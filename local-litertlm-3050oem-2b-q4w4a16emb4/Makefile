# local-litertlm-3050oem-2b-q4w4a16emb4 — LiteRT-LM on the RTX 3050 OEM, Gemma 4 E2B,
# the QAT Q4_0 grid held exactly. No provisioning: the card is in this machine.
# Values from tpu.env, exported to every recipe; override on the command line.
include tpu.env
export

CONFIG := run/config.json

.PHONY: all help install tools convert repack pack build import config serve status query bench test lint clean

all: help

help:
	@echo "local-litertlm-3050oem-2b-q4w4a16emb4 — RTX 3050 OEM, Gemma 4 E2B, exact QAT Q4_0 in LiteRT-LM"
	@echo "  make install   - litert-lm + protobuf into the system python3 user site"
	@echo "  make build     - tools + convert + repack + pack (about 20 min, ~45 GB RAM peak)"
	@echo "  make import    - register MODEL_PATH as MODEL_ID in LITERT_LM_DIR"
	@echo "  make serve     - OpenAI-compatible server on PORT, foreground"
	@echo "  make status | query | bench | test | lint"

install:
	python3 -m pip install --user --break-system-packages -r requirements.txt

tools convert repack pack:
	bash build.sh $@

build:
	bash build.sh all

import:
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c -
	@test -f $(LITERT_LM_DIR)/models/$(MODEL_ID)/model.litertlm && echo "already imported as $(MODEL_ID)" || \
	  litert-lm import $(MODEL_PATH) $(MODEL_ID)

config:
	python3 render_config.py $(CONFIG)

serve: config
	litert-lm serve --config $(CONFIG) --host $(HOST) --port $(PORT)

status:
	@curl -fsS $(ENDPOINT)/v1/models >/dev/null && echo "serving at $(ENDPOINT)" || echo "not serving at $(ENDPOINT)"

# max_completion_tokens: the server ignores max_tokens.
query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_ID)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_completion_tokens":256}' \
	  | python3 -m json.tool

bench:
	litert-lm benchmark --backend $(BACKEND) --speculative-decoding $(SPECULATIVE_DECODING) \
	  -p 1024 -d 256 --runs 3 $(MODEL_ID)

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check render_config.py repack_litertlm.py tests benchmarks/runs/*/*.py
	bash -n build.sh
	@echo "lint OK"

clean:
	rm -rf __pycache__ .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

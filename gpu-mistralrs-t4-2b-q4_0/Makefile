# gpu-mistralrs-t4-2b-q4_0 — mistral.rs on the Tesla T4, Gemma 4 E2B exact QAT Q4_0 GGUF (forked from local-mistralrs-3050oem-2b-q4_0).
#
# No provisioning targets: the card is in this machine (NAMING.md, `local`). `make serve` is the
# deploy and Ctrl-C the teardown. serve.py builds the command from tpu.env.
include tpu.env
export

.PHONY: all help src install command serve status query test lint

all: help

help:
	@echo "  make src       - clone mistral.rs at MISTRALRS_VERSION into MISTRALRS_SRC"
	@echo "  make install   - cargo install mistralrs-cli with MISTRALRS_FEATURES into MISTRALRS_BIN's root"
	@echo "  make command   - print the mistralrs serve command"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status | query | test | lint"

src:
	test -d $(MISTRALRS_SRC) || git clone --depth 50 --branch $(MISTRALRS_VERSION) https://github.com/EricLBuehler/mistral.rs.git $(MISTRALRS_SRC)

install: src
	@echo "$(MMPROJ_SHA256)  $(MMPROJ_PATH)" | sha256sum -c -
	cd $(MISTRALRS_SRC) && CUDA_COMPUTE_CAP=$(CUDA_COMPUTE_CAP) cargo install --locked --path mistralrs-cli \
	  --features "$(MISTRALRS_FEATURES)" --root $(dir $(patsubst %/,%,$(dir $(MISTRALRS_BIN))))

command:
	@python3 serve.py --print

serve:
	python3 serve.py

status:
	@curl -fsS $(ENDPOINT)/health >/dev/null && echo "serving on $(ENDPOINT)" || echo "not serving on $(ENDPOINT)"

query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions -H 'Content-Type: application/json' \
	  -d '{"model":"default","messages":[{"role":"user","content":"Name the planets in one line."}],"max_tokens":128,"temperature":0}' \
	  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check serve.py tests

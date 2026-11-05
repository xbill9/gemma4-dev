# gpu-mistralrs-g6-2b-q4_0 — mistral.rs on one NVIDIA L4 (EC2 G6), Gemma 4 E2B exact QAT Q4_0 GGUF.
#
# Run ON the instance, after gpu-llamacpp-g6-2b-q4_0's `make toolkit download`. serve.py builds the
# command from tpu.env.
include tpu.env
export

PY ?= python3
CARGO = $(HOME)/.cargo/bin/cargo

.PHONY: all help deps rust src mmproj install command serve status query test lint

all: help

help:
	@echo "  make deps      - apt: build-essential pkg-config libssl-dev git"
	@echo "  make rust      - rustup, stable toolchain, into HOME"
	@echo "  make src       - clone mistral.rs at MISTRALRS_VERSION into MISTRALRS_SRC"
	@echo "  make mmproj    - Google's projector GGUF, SHA-256 checked"
	@echo "  make install   - cargo install mistralrs-cli for sm_$(CUDA_COMPUTE_CAP) into MISTRALRS_BIN's root"
	@echo "  make command   - print the mistralrs serve command"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status | query | test | lint"

deps:
	apt-get -o DPkg::Lock::Timeout=600 install -y build-essential pkg-config libssl-dev git

rust:
	test -x $(CARGO) || (curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal)

src:
	test -d $(MISTRALRS_SRC) || git clone --depth 50 --branch $(MISTRALRS_VERSION) https://github.com/EricLBuehler/mistral.rs.git $(MISTRALRS_SRC)
	test "$$(git -C $(MISTRALRS_SRC) rev-parse --short=7 HEAD)" = "$(MISTRALRS_COMMIT)"

mmproj:
	$(PY) -c "from huggingface_hub import hf_hub_download as d; d('$(MMPROJ_HF)', '$(MMPROJ_HF_FILE)', revision='$(MMPROJ_HF_REVISION)', local_dir='$(dir $(MMPROJ_PATH))')"
	echo "$(MMPROJ_SHA256)  $(MMPROJ_PATH)" | sha256sum -c -

install: src
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | sha256sum -c -
	@echo "$(MMPROJ_SHA256)  $(MMPROJ_PATH)" | sha256sum -c -
	cd $(MISTRALRS_SRC) && PATH=$(CUDA_HOME)/bin:$$PATH CUDA_COMPUTE_CAP=$(CUDA_COMPUTE_CAP) \
	  $(CARGO) install --locked --path mistralrs-cli -j $(CARGO_BUILD_JOBS) \
	  --features "$(MISTRALRS_FEATURES)" --root $(dir $(patsubst %/,%,$(dir $(MISTRALRS_BIN))))

command:
	@$(PY) serve.py --print

serve:
	$(PY) serve.py

status:
	@curl -fsS $(ENDPOINT)/health >/dev/null && echo "serving on $(ENDPOINT)" || echo "not serving on $(ENDPOINT)"

query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions -H 'Content-Type: application/json' \
	  -d '{"model":"default","messages":[{"role":"user","content":"Name the planets in one line."}],"max_tokens":128,"temperature":0}' \
	  | $(PY) -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check serve.py attest.py sweep.py tests

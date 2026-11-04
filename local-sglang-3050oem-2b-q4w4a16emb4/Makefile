# local-sglang-3050oem-2b-q4w4a16emb4 — SGLang on the RTX 3050 OEM, Gemma 4 E2B QAT W4A16 (int4 tables).
# No provisioning: the card is in this machine. `make serve` is the deploy and Ctrl-C the teardown.
include tpu.env
export

PIP = python3 -m pip install --user --break-system-packages
WHEELDIR ?= $(HOME)/.cache/sglang-install
WHEEL313 = sglang-$(SGLANG_VERSION)-cp313-cp313-manylinux_2_34_x86_64.whl
WHEEL314 = sglang-$(SGLANG_VERSION)-cp314-cp314-manylinux_2_34_x86_64.whl
# sglang requirements missing from the system python on 2026-11-04 that the HTTP server needs; the rest
# (deep-ep, deep-gemm, flash-mla, flash-attn-4, tilelang, torchaudio, ...) are left out
MISSING = sglang-kernel==$(SGLANG_KERNEL_VERSION) blobfile==3.0.0 easydict gguf openai-harmony==0.0.4 \
  soundfile==0.13.1 timm==1.0.25 torch_memory_saver zstandard pycryptodomex cffi

.PHONY: all help install command serve status query test lint

all: help

help:
	@echo "  make install   - sglang cp313 wheel retagged cp314 + missing requirements, all --no-deps"
	@echo "  make command   - print the server command"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status | query | test | lint"

install:
	mkdir -p $(WHEELDIR)
	python3 -m pip freeze --user > $(WHEELDIR)/freeze-before-install.txt
	cd $(WHEELDIR) && python3 -m pip download --no-deps --only-binary :all: --python-version 3.13 sglang==$(SGLANG_VERSION)
	cp $(WHEELDIR)/$(WHEEL313) $(WHEELDIR)/$(WHEEL314)
	$(PIP) --no-deps $(WHEELDIR)/$(WHEEL314)
	$(PIP) --no-deps $(MISSING)
	$(PIP) ipython pycparser

command:
	@python3 serve.py --print

serve:
	python3 serve.py

status:
	@curl -fsS $(ENDPOINT)/health >/dev/null && echo "serving on $(ENDPOINT)" || echo "not serving on $(ENDPOINT)"

query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"Name the planets in one line."}],"max_tokens":128,"temperature":0}' \
	  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check serve.py tests

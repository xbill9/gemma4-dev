# gpu-sglang-g6-2b-q4w4a16emb4 — SGLang on one NVIDIA L4 (EC2 G6), Gemma 4 E2B QAT W4A16 (int4 tables).
# Run ON the instance, after gpu-vllm-g6-2b-q4w4a16emb4's `make pip install` (same interpreter; it brings torch).
include tpu.env
export

PIP = $(PYTHON) -m pip install --user --break-system-packages
SITE = $(shell $(PYTHON) -m site --user-site)
WHEELDIR ?= /opt/cache/sglang-install
WHEEL313 = sglang-$(SGLANG_VERSION)-cp313-cp313-manylinux_2_34_x86_64.whl
WHEEL314 = sglang-$(SGLANG_VERSION)-cp314-cp314-manylinux_2_34_x86_64.whl
# The 3050 list of requirements missing for the HTTP server (2026-11-04). On a fresh instance others may
# be missing too: `make check` imports the server and names the first one.
MISSING = sglang-kernel==$(SGLANG_KERNEL_VERSION) blobfile==3.0.0 easydict gguf openai-harmony==0.0.4 \
  soundfile==0.13.1 timm==1.0.25 torch_memory_saver zstandard pycryptodomex cffi

.PHONY: all help install check command serve status query test lint

all: help

help:
	@echo "  make install   - sglang cp313 wheel retagged cp314 + missing requirements, all --no-deps, + patches/"
	@echo "  make check     - sglang.launch_server --help runs (imports the server)"
	@echo "  make command   - print the server command"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status | query | test | lint"

install:
	mkdir -p $(WHEELDIR)
	$(PYTHON) -m pip freeze --user > $(WHEELDIR)/freeze-before-install.txt
	cd $(WHEELDIR) && $(PYTHON) -m pip download --no-deps --only-binary :all: --python-version 3.13 sglang==$(SGLANG_VERSION)
	cp $(WHEELDIR)/$(WHEEL313) $(WHEELDIR)/$(WHEEL314)
	$(PIP) --no-deps $(WHEELDIR)/$(WHEEL314)
	$(PIP) --no-deps $(MISSING)
	$(PIP) ipython pycparser
	patch -N -p1 -d $(SITE) < patches/text-config-per-layer.diff || true
	patch -N -p1 -d $(SITE) < patches/int4-embeddings.diff || true

check:
	$(PYTHON) -m sglang.launch_server --help > /dev/null && $(PYTHON) -c "import sglang; print('sglang', sglang.__version__)"

command:
	@$(PYTHON) serve.py --print

serve:
	$(PYTHON) serve.py

status:
	@curl -fsS $(ENDPOINT)/health >/dev/null && echo "serving on $(ENDPOINT)" || echo "not serving on $(ENDPOINT)"

query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"Name the planets in one line."}],"max_tokens":128,"temperature":0}' \
	  | $(PYTHON) -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check serve.py attest.py sweep.py tests

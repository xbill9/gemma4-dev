# gpu-vllm-g6-2b-q4w4a16emb4 — vLLM on one NVIDIA L4 (EC2 G6), Gemma 4 E2B QAT int4.
#
# Run these ON the instance (launched with gpu-pytorch-g6-2b-q4w4a16's create_g6_instance; SSM runs
# as root). tpu.env decides every value; override one on the make command line.
include tpu.env
export

PIP = $(PYTHON) -m pip install --user --break-system-packages

.PHONY: all help pip install check drafter command serve status query test lint

all: help

help:
	@echo "  make pip       - apt install python3-pip (Ubuntu 26.04 system python has none)"
	@echo "  make install   - vllm==$(VLLM_VERSION) with its own pins (torch==$(TORCH_VERSION)) into the user site"
	@echo "  make check     - torch sees the L4, vllm imports"
	@echo "  make drafter   - W4A16 repack of Google's QAT assistant into DRAFTER_DIR"
	@echo "  make command   - print the vllm serve command"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status | query | test | lint"

pip:
	apt-get -o DPkg::Lock::Timeout=600 install -y python3-pip

# No system torch on the instance, so vLLM's torch pin is taken as is (the 3050 rig skipped it).
install:
	$(PYTHON) -m pip freeze --user > freeze-before-install.txt
	$(PIP) vllm==$(VLLM_VERSION)

check:
	$(PYTHON) -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
	  torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('vllm', vllm.__version__)"
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

drafter:
	$(PYTHON) -c "from huggingface_hub import snapshot_download as s; \
	  print(s('$(DRAFTER_SOURCE)', revision='$(DRAFTER_SOURCE_REVISION)'))" > drafter-src.txt
	$(PYTHON) repack/repack_drafter.py $$(cat drafter-src.txt) $(DRAFTER_DIR)
	sha256sum $(DRAFTER_DIR)/*.safetensors

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

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	ruff check serve.py attest.py sweep.py repack flight tests

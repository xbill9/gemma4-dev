# gpu-vllm-t4-embed2 — EmbeddingGemma 2 under vLLM on the T4 attached to this VM.
#
# No provisioning targets, as in gpu-vllm-t4-2b: the GPU is already here and is
# billed whether this runs or not. The generative T4 rigs share the GPU and port
# 8000 — stop theirs before `make serve`.
#
# `make serve` WORKS HERE, unlike the sibling's: start_vllm_server launches the
# engine with subprocess.Popen in its own session, so it outlives this one-shot
# interpreter. (The sibling's asyncio child dies with the event loop.)
#
# START WITH `make capacity`.
include tpu.env
export

.PHONY: all help capacity arch patch verify-patch serve stop status verify query test lint clean bench-validate

all: help

help:
	@echo "=============================================================="
	@echo " gpu-vllm-t4-embed2 — EmbeddingGemma 2 on one attached Tesla T4"
	@echo "=============================================================="
	@echo "  make capacity     - CAN this host install and serve? run first"
	@echo "  make arch         - torch SM 7.5 + a vLLM/transformers that know the model"
	@echo "  make patch        - clamp the encoder attention kernel for Turing"
	@echo "  make verify-patch - is the clamp in the module vLLM imports?"
	@echo "  make serve        - start /v1/embeddings (refuses if unsafe)"
	@echo "  make status       - is it up, which model, how much VRAM"
	@echo "  make verify       - dimension, norm, ranking, Matryoshka checks"
	@echo "  make query        - one raw /v1/embeddings request"
	@echo "  make test         - offline unittest suite"
	@echo "  make lint         - ruff on the module list below"
	@echo "=============================================================="
	@echo "STATE 2026-10-07: serving and measured once —"
	@echo "  benchmarks/runs/2026-10-07-first-serve-t4/REPORT.md"

# The order is not decorative: each step's failure is cheaper than the next one's.
capacity:
	@python3 -c "import asyncio, server; print(asyncio.run(server.check_host_capacity()))"

arch:
	@python3 -c "import asyncio, server; print(asyncio.run(server.verify_gpu_arch()))"

patch:
	@python3 -c "import asyncio, server; print(asyncio.run(server.apply_encoder_patch()))"

verify-patch:
	@python3 -c "import asyncio, server; print(asyncio.run(server.verify_encoder_patch()))"

serve:
	@python3 -c "import asyncio, server; print(asyncio.run(server.start_vllm_server()))"

stop:
	@python3 -c "import asyncio, server; print(asyncio.run(server.stop_vllm_server()))"

status:
	@python3 -c "import asyncio, server; print(asyncio.run(server.server_status()))"

verify:
	@python3 -c "import asyncio, server; print(asyncio.run(server.verify_embeddings()))"

# The query prefix is part of the request on purpose: vLLM does not add it.
query:
	@curl -fsS $(ENDPOINT)/v1/embeddings \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","input":"task: search result | query: what is a TPU?","dimensions":128}' \
	  | python3 -c "import json,sys; d=json.load(sys.stdin)['data'][0]['embedding']; print(len(d), [round(x,4) for x in d[:6]])"

# THE MCP SERVER'S DEPS ONLY, never torch or vLLM — those belong to PYTHON_BIN
# under this rig's PYTHONUSERBASE. `make capacity` prints the model-side install.
install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py patch_triton_encoder_turing.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/reports

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

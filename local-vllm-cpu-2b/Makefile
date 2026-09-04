# local-vllm-cpu-2b — vLLM on the local CPU. No accelerator, no control plane.
#
# There are no provisioning targets and none is owed. `make serve` is the whole
# of what `make deploy` means elsewhere, and teardown is Ctrl-C.
#
# START WITH `make capacity`. As of 2026-09-04 it refuses on this host, and
# `make serve` refuses with it — the weights do not fit RAM. Building vLLM
# before that arithmetic passes spends an hour on something that cannot serve.
include tpu.env
export

.PHONY: all help capacity install verify serve stop status query test lint clean bench-validate

all: help

help:
	@echo "========================================================="
	@echo " local-vllm-cpu-2b — vLLM on CPU, Gemma 4 E2B, no chip"
	@echo "========================================================="
	@echo "  make capacity  - CAN this host hold the weights? run this first"
	@echo "  make verify    - is the installed vLLM actually a CPU build?"
	@echo "  make install   - deps for the MCP server only (NOT vLLM itself)"
	@echo "  make serve     - start the OpenAI server (refuses if over budget)"
	@echo "  make status    - is it up, and what has happened to free RAM"
	@echo "  make query     - one chat completion"
	@echo "  make test      - offline unittest suite"
	@echo "  make lint      - ruff on the module list below"
	@echo "========================================================="
	@echo "Blocker 1, RAM: 10.25 GB of weights against ~9.5 GB available."
	@echo "Blocker 2, ISA: this CPU is AVX2-only, so vLLM needs a source build"
	@echo "  and its backend upcasts bf16 to fp32, doubling the weights."
	@echo "Exceeding host RAM does not raise - it swaps, and a thrashing serve"
	@echo "looks exactly like a loading one. That is why serve refuses."

capacity:
	@python3 -c "import asyncio, server; print(asyncio.run(server.check_host_capacity()))"

# THIS DOES NOT INSTALL vLLM. There is no published CPU wheel
# (wheels.vllm.ai/cpu -> 404) and the PyPI `vllm` wheel is the CUDA build, so a
# CPU vLLM is a source build:
#
#   git clone https://github.com/vllm-project/vllm && cd vllm
#   VLLM_TARGET_DEVICE=cpu VLLM_CPU_DISABLE_AVX512=1 pip install -e .
#
# Deliberately not a target: it takes ~an hour, it replaces the system torch,
# and on this host `make capacity` says the result could not serve.
install:
	python3 -m pip install -r requirements.txt

verify:
	@python3 -c "import asyncio, server; print(asyncio.run(server.verify_cpu_backend()))"

serve:
	@python3 -c "import asyncio, server; print(asyncio.run(server.start_vllm_server()))"

stop:
	@python3 -c "import asyncio, server; print(asyncio.run(server.stop_vllm_server()))"

status:
	@python3 -c "import asyncio, server; print(asyncio.run(server.server_status()))"

# /v1/chat/completions, NOT /v1/completions: raw completions return an empty
# string on -it checkpoints.
query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_tokens":1024}' \
	  | python3 -m json.tool

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/runs

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

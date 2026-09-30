# gpu-vllm-t4-4b-q4w4a16emb4 — vLLM on the T4 attached to this Compute Engine VM, Gemma 4 E4B.
#
# There are no provisioning targets and none is owed: the GPU is already here.
# `make serve` is the whole of what `make deploy` means in the EC2 siblings, and
# teardown is `make stop` — which does NOT release capacity, because the VM and
# its attached GPU are billed either way.
#
# START WITH `make capacity`. It is free arithmetic over three filesystems and it
# prints the install command with the redirections this host needs. A default
# `pip install` fails here: `/tmp` has 3.88 GB.
include tpu.env
export

.PHONY: all help capacity arch patch verify-patch serve stop status query test lint clean bench-validate

all: help

help:
	@echo "=============================================================="
	@echo " gpu-vllm-t4-4b-q4w4a16emb4 — vLLM on one attached Tesla T4 (SM 7.5)"
	@echo "=============================================================="
	@echo "  make capacity     - CAN this host install and serve? run first"
	@echo "  make arch         - does the installed build have SM 7.5 kernels?"
	@echo "  make patch        - clamp Triton's tiles to Turing's 64 KiB"
	@echo "  make verify-patch - is the clamp in the module vLLM imports?"
	@echo "  make serve        - start the OpenAI server (refuses if unsafe)"
	@echo "  make status       - is it up, and has VRAM been claimed"
	@echo "  make query        - one chat completion"
	@echo "  make test         - offline unittest suite"
	@echo "  make lint         - ruff on the module list below"
	@echo "=============================================================="
	@echo "STATE 2026-09-30: forked from gpu-vllm-t4-2b-w4a16; nothing served"
	@echo "  or measured as E4B. Serve with ./vllm-t4, not make serve."
	@echo "One rig at a time: the T4 and port 8000 are shared with the 2b rigs."

# The order is not decorative: each step's failure is cheaper than the next one's.
capacity:
	@python3 -c "import asyncio, server; print(asyncio.run(server.check_host_capacity()))"

arch:
	@python3 -c "import asyncio, server; print(asyncio.run(server.verify_gpu_arch()))"

patch:
	@python3 -c "import asyncio, server; print(asyncio.run(server.apply_turing_patch()))"

verify-patch:
	@python3 -c "import asyncio, server; print(asyncio.run(server.verify_turing_patch()))"

serve:
	@python3 -c "import asyncio, server; print(asyncio.run(server.start_vllm_server()))"

stop:
	@python3 -c "import asyncio, server; print(asyncio.run(server.stop_vllm_server()))"

status:
	@python3 -c "import asyncio, server; print(asyncio.run(server.server_status()))"

# /v1/chat/completions, NOT /v1/completions: raw completions return an empty
# string on -it checkpoints (root CLAUDE.md).
query:
	@curl -fsS $(ENDPOINT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL_NAME)","messages":[{"role":"user","content":"In one sentence, what is a TPU?"}],"max_tokens":1024}' \
	  | python3 -m json.tool

# THIS TARGET INSTALLS THE MCP SERVER'S DEPS ONLY, never torch or vLLM — those
# belong to PYTHON_BIN, a different interpreter on a different filesystem.
# `make capacity` prints the model-side command.
install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; install with: pip install ruff"; exit 1; }
	ruff check server.py patch_triton_turing.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/runs

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

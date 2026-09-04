# local-pytorch-cpu-2b — PyTorch/transformers on the local CPU. No accelerator.
#
# No provisioning targets and none is owed. This rig is the reference
# implementation: its job is to be the thing other rigs are checked against.
include tpu.env
export

.PHONY: all help capacity install status first-light query test lint clean bench-validate

all: help

help:
	@echo "========================================================="
	@echo " local-pytorch-cpu-2b — transformers on CPU, Gemma 4 E2B"
	@echo "========================================================="
	@echo "  make capacity     - can this host hold the weights? run first"
	@echo "  make install      - pip install -r requirements.txt (system python3)"
	@echo "  make status       - CPU, live RAM, is the model resident"
	@echo "  make first-light  - load + generate 128 tok, the reproducible run"
	@echo "  make query        - one generation at the full token budget"
	@echo "  make test         - offline unittest suite"
	@echo "========================================================="
	@echo "MEASURED 2026-09-04: 4.587 tok/s, peak RSS 5.71 GB of a 10.25 GB file."
	@echo "RSS AFTER LOAD IS NOT THE FOOTPRINT - safetensors mmaps, so it reads"
	@echo "1.20 GB until generation walks the weights. Size from the file."

capacity:
	@python3 -c "import asyncio, server; print(asyncio.run(server.check_host_capacity()))"

install:
	python3 -m pip install -r requirements.txt

status:
	@python3 -c "import asyncio, server; print(asyncio.run(server.host_status()))"

first-light:
	python3 bench_first_light.py

query:
	@python3 -c "import asyncio, server; print(asyncio.run(server.query_model('In one sentence, what is a TPU?')))"

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS SILENTLY UNLINTED UNTIL IT IS ADDED HERE.
lint:
	@command -v ruff >/dev/null || { echo "ruff not found; pip install ruff"; exit 1; }
	ruff check server.py bench_first_light.py tests
	@echo "lint OK"

bench-validate:
	python3 ../benchmarks/validate.py benchmarks/runs

clean:
	rm -rf __pycache__ .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

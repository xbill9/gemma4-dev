# tokenlab-sglang-amd-31b — Gemma 4 31B on AMD Token Lab's hosted API.
# Nothing to provision. Key: TOKENLAB_API_KEY in .env (mode 0600) or the environment.

PYTHON ?= python3
UI_PORT ?= $(shell sed -n 's/^UI_PORT=//p' tpu.env)
BIN ?= $(HOME)/bin

.PHONY: help ui ask models query probe bin agent agent-web agent-once test lint clean install

help:
	@echo "  make ui       - chat page on http://localhost:$(UI_PORT) (key stays in the local proxy)"
	@echo "  make ask      - terminal REPL (ask.py); one-shot: ./ask.py <prompt>, or pipe into it"
	@echo "  make bin      - symlink ask.py to $(BIN)/ask-tokenlab"
	@echo "  make models   - the model card Token Lab lists for MODEL_NAME"
	@echo "  make query    - one non-streamed chat completion"
	@echo "  make probe    - first-light timing probe (3 requests) into today's benchmarks run"
	@echo "  make agent    - ADK agent (tokenlab_agent) in the terminal: adk run"
	@echo "  make agent-web - ADK dev UI in the browser: adk web"
	@echo "  make agent-once Q='...' - one agent turn, printing tool calls and timings"
	@echo "  make test     - offline unit tests"
	@echo "  make lint     - ruff"

ui:
	$(PYTHON) chat_ui.py --port $(UI_PORT)

ask:
	$(PYTHON) ask.py

bin:
	ln -sf $(CURDIR)/ask.py $(BIN)/ask-tokenlab
	@echo "$(BIN)/ask-tokenlab -> $(CURDIR)/ask.py"

models:
	@$(PYTHON) -c 'import json, tokenlab; print(json.dumps(tokenlab.model_entry(tokenlab.load_config()), indent=2))'

# /chat/completions, not /completions: raw completions return empty on -it models.
query:
	@$(PYTHON) ask.py -r -n 64 "Capital of Australia? One sentence." </dev/null

probe:
	cd benchmarks/runs/2026-11-13-first-light-tokenlab && $(PYTHON) probe.py --n 3

agent:
	adk run tokenlab_agent

agent-web:
	adk web --port 8767

agent-once:
	$(PYTHON) run_agent.py $(Q)

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check tokenlab.py chat_ui.py ask.py run_agent.py tokenlab_agent tests benchmarks/runs/*/probe.py

install:

clean:
	rm -rf __pycache__ tests/__pycache__ tokenlab_agent/__pycache__ .ruff_cache

# gpu-vllm-t4-2b-q4w4a16emb4 — vLLM on the Tesla T4, Gemma 4 E2B QAT int4 (forked from local-vllm-3050oem-2b-q4w4a16emb4).
#
# No provisioning targets: the card is in this machine (NAMING.md, `local`). `make serve` is the
# deploy and Ctrl-C the teardown. serve.py builds the command from tpu.env.
include tpu.env
export

.PHONY: all help install command serve status query test lint

all: help

help:
	@echo "  make install   - vLLM into the system python3 on the existing torch (see tpu.env)"
	@echo "  make command   - print the vllm serve command"
	@echo "  make serve     - OpenAI-compatible server in the foreground"
	@echo "  make status    - is it up?"
	@echo "  make query     - one chat completion"
	@echo "  make test      - offline unittest suite"
	@echo "  make lint      - ruff"

# vLLM pins torch==2.14.0; --no-deps keeps the system torch. Its other requirements are
# installed first, without torch/torchvision/torchaudio.
install:
	curl -s https://pypi.org/pypi/vllm/$(VLLM_VERSION)/json | python3 -c "import json,re,sys; \
	[print(r) for r in json.load(sys.stdin)['info']['requires_dist'] \
	 if not ('extra' in r.split(';',1)[-1] and ';' in r) \
	 and re.match(r'[A-Za-z0-9_.-]+', r).group(0).lower() not in ('torch','torchvision','torchaudio')]" > run-deps.txt
	python3 -m pip install --user --break-system-packages -r run-deps.txt
	python3 -m pip install --user --break-system-packages --no-deps vllm==$(VLLM_VERSION)
	rm -f run-deps.txt

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
	ruff check serve.py attest.py sweep.py tests

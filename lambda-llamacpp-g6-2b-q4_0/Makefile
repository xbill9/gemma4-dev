# lambda-llamacpp-g6-2b-q4_0 — Gemma 4 E2B as a Lambda GPU function (Lambda Managed Instances, g6.2xlarge), llama.cpp 7182ef71e.
#
# lmi.py does the work through the AWS CLI; builder.py builds the image on EC2; tpu.env decides every value.
.PHONY: all help builder builder-down image roles provider function deploy up down status args ui ask bench destroy test lint
UI_PORT := $(shell sed -n 's/^UI_PORT=//p' tpu.env)

all: help

help:
	@echo "  make builder   - start the EC2 image builder (shared by the lambda-*-g6 rigs)"
	@echo "  make image     - build image/Dockerfile on the builder and push it to ECR"
	@echo "  make builder-down - terminate the builder"
	@echo "  make deploy    - roles, image, capacity provider, function, publish (waits for Active)"
	@echo "  make up        - min/max environments to tpu.env's values (an instance starts)"
	@echo "  make down      - min/max environments to 0 (instance released, function kept)"
	@echo "  make status    - version, state, scaling, running g6.2xlarge instances"
	@echo "  make args      - the function's environment (engine flags), offline"
	@echo "  make ui        - local chat page on http://localhost:$(UI_PORT) (proxies to lambda invoke)"
	@echo "  make ask Q='…' - one chat completion through lambda invoke (THINK=1 for thinking)"
	@echo "  make bench OUT=dir - concurrent invokes at 1,4,16 requests"
	@echo "  make destroy   - delete the function (the capacity provider is shared)"
	@echo "  make test      - offline unittest suite"

builder:      ; python3 builder.py up
builder-down: ; python3 builder.py down
image:    ; python3 lmi.py image
roles:    ; python3 lmi.py roles
provider: ; python3 lmi.py provider
function: ; python3 lmi.py function
deploy:   ; python3 lmi.py deploy
up:       ; python3 lmi.py up
down:     ; python3 lmi.py down
status:   ; python3 lmi.py status
args:     ; python3 lmi.py args
destroy:  ; python3 lmi.py destroy
ui:       ; python3 chat_ui.py
ask:      ; python3 lmi.py ask $(if $(THINK),--think) "$(or $(Q),Say hello in five words.)"
bench:    ; python3 bench.py $(or $(OUT),benchmarks/runs/$(shell date +%F)-lambda) --levels $(or $(LEVELS),1,4,16)

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check *.py image/*.py tests

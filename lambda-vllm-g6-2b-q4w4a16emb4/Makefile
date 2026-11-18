# lambda-vllm-g6-2b-q4w4a16emb4 — Gemma 4 E2B as a Lambda GPU function (Lambda Managed Instances, g6.2xlarge).
#
# lmi.py does the work through the AWS CLI; tpu.env decides every value.
.PHONY: all help model image roles provider function deploy up down status ask destroy test lint

all: help

help:
	@echo "  make model     - checkpoint at the pinned revision into image/model"
	@echo "  make image     - build and push the image to ECR"
	@echo "  make deploy    - roles, image, capacity provider, function, publish (waits for Active)"
	@echo "  make up        - min/max environments back to tpu.env's values (instances start)"
	@echo "  make down      - min/max environments to 0 (instances released, function kept)"
	@echo "  make status    - version, state, scaling, running g6.2xlarge instances"
	@echo "  make ask Q='…' - one chat completion through lambda invoke (THINK=1 for thinking)"
	@echo "  make destroy   - delete the function and the capacity provider"
	@echo "  make test      - offline unittest suite"

model:    ; python3 lmi.py model
image:    ; python3 lmi.py image
roles:    ; python3 lmi.py roles
provider: ; python3 lmi.py provider
function: ; python3 lmi.py function
deploy:   ; python3 lmi.py deploy
up:       ; python3 lmi.py up
down:     ; python3 lmi.py down
status:   ; python3 lmi.py status
destroy:  ; python3 lmi.py destroy
ask:      ; python3 lmi.py ask $(if $(THINK),--think) "$(or $(Q),Say hello in five words.)"

test:
	python3 -m unittest discover -s tests -v

lint:
	ruff check lmi.py tests

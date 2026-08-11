# Configuration - tpu.env is the single source of truth; server.py and mcp-run.sh read
# the same file. Override per-invocation with e.g. `make status ZONE=europe-west4-a`.
-include tpu.env

PROJECT_ID ?= $(GOOGLE_CLOUD_PROJECT)
REGION ?= $(GOOGLE_CLOUD_REGION)
ZONE ?= $(GOOGLE_CLOUD_ZONE)
MODEL_PATH ?= $(MODEL_NAME)
BUCKET_NAME ?= $(PROJECT_ID)-bucket
# The instance is named after this rig directory, matching the INSTANCE_NAME default
# server.py derives the same way — so unlike the TPU-API rigs, `make status` and the MCP
# tools address the same object. Set SERVICE_NAME in tpu.env (or on the command line) to
# point these targets at a VM whose name predates that, e.g.
#   make status SERVICE_NAME=tpu-2B-v5e1-devops-agent
SERVICE_NAME ?= $(notdir $(CURDIR))
# Key this rig's MCP server is registered under in the client, and the name server.py
# advertises. It prefixes every tool (mcp__$(MCP_SERVER_NAME)__find_tpu), so it has to be
# unique across the sibling rigs — they used to share "tpu-devops". Same derivation as
# RIG_NAME in server.py; tpu.env or the command line override it.
MCP_SERVER_NAME ?= $(notdir $(CURDIR))

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

format:
	ruff format .
	ruff check --fix .

# Regenerates GemmaTools.md from the @mcp.tool() decorators in server.py
tools:
	python generate_tools_doc.py

# Writes .mcp.json registering this rig's server under $(MCP_SERVER_NAME). Idempotent, and
# it merges into an existing .mcp.json rather than replacing it, so other servers in the
# file survive. .mcp.json is gitignored at the monorepo root; mcp_config.json is the
# committed example of the same entry.
mcp-config:
	@python3 -c 'import json,sys; \
path, name, cmd = sys.argv[1:4]; \
config = json.load(open(path)) if __import__("os").path.exists(path) else {}; \
config.setdefault("mcpServers", {})[name] = {"command": cmd, "args": [], "env": {}}; \
json.dump(config, open(path, "w"), indent=2); \
open(path, "a").write("\n"); \
print(f"✅ registered MCP server {name!r} in {path}")' \
		"$(CURDIR)/.mcp.json" "$(MCP_SERVER_NAME)" "$(CURDIR)/mcp-run.sh"

clean:
	rm -rf __pycache__
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -rf {} +

# Provisioning model for the make path. These are gcloud's own SCREAMING_CASE values for
# `compute instances create`: STANDARD, SPOT, FLEX_START, RESERVATION_BOUND.
#
# UNLIKE THE SIBLING RIG, this is not a second vocabulary for a second path. There, the
# Makefile spoke `tpu-vm` while the MCP tools spoke `queued-resources`, so `make status`
# described a different TPU node from the one the agent had provisioned and the two could
# never see each other. Here both call `gcloud compute instances`, so a VM made by
# `make deploy-tpu` is the same object `list_tpu_instances` returns.
TPU_PROVISIONING_MODEL ?= STANDARD
MACHINE_TYPE ?= ct5lp-hightpu-1t
IMAGE_FAMILY ?= ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e
IMAGE_PROJECT ?= ubuntu-os-accelerator-images
BOOT_DISK_SIZE_GB ?= 200
MAX_RUN_DURATION ?= 4h
REQUEST_VALID_FOR ?= 2h
EXTRA_FLAGS ?=

# One v5e chip as a Compute Engine instance. --maintenance-policy=TERMINATE is required (a
# TPU instance cannot live-migrate) and --scopes=cloud-platform is what lets the startup
# script read the HF token out of Secret Manager at boot.
#
# ⚠️  WHETHER THIS WORKS AT ALL FOR v5e IS UNSETTLED — Google documents v5e as GKE and Cloud
# TPU API only, and this command has never been run. See CLAUDE.md. A rejection costs
# nothing; an acceptance bills until you delete it.
deploy-tpu:
	@echo "🚀 Creating $(TPU_PROVISIONING_MODEL) $(MACHINE_TYPE) instance ($(SERVICE_NAME)) in $(ZONE)..."
	gcloud compute instances create $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--machine-type=$(MACHINE_TYPE) \
		--image-family=$(IMAGE_FAMILY) \
		--image-project=$(IMAGE_PROJECT) \
		--maintenance-policy=TERMINATE \
		--boot-disk-size=$(BOOT_DISK_SIZE_GB)GB \
		--scopes=cloud-platform \
		--provisioning-model=$(TPU_PROVISIONING_MODEL) \
		--max-run-duration=$(MAX_RUN_DURATION) \
		--instance-termination-action=DELETE \
		--labels=rig=$(SERVICE_NAME),purpose=make \
		$(EXTRA_FLAGS)

# Spot: discounted, preempted whenever the capacity is wanted back. On v5e in us-west4 spot
# really is the cheapest of the three — but that is a fact about this chip in this region,
# not a rule: on v6e in us-east5 spot lists dearer than flex-start.
deploy-tpu-spot:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=SPOT

# On-demand: full price, no preemption. Same as plain `make deploy-tpu`.
deploy-tpu-ondemand:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=STANDARD

# Flex-start via Dynamic Workload Scheduler. The sibling rig's Makefile has no equivalent —
# gcloud's tpu-vm path offers no flex-start at all, so it was reachable only through the MCP
# tools' Queued Resource path. Blocks until capacity is granted or the request expires.
#
# EXTRA_FLAGS is threaded into the deploy-tpu recipe above on purpose. The v6e rig this was
# ported from sets it here but never expands it, so its --request-valid-for-duration is
# silently dropped and the request uses gcloud's default wait. Don't reintroduce that.
deploy-tpu-flex:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=FLEX_START \
		EXTRA_FLAGS="--request-valid-for-duration=$(REQUEST_VALID_FOR)"

# Note: Running the container requires SSHing into the TPU VM.
# This target provides the command for reference.
run-container:
	@echo "🛠 Run this command on the TPU VM to start vLLM:"
	@echo "sudo docker run -t --rm --name vllm-gemma4 --privileged --net=host \\"
	@echo "    -v /dev/shm:/dev/shm --shm-size 10gb \\"
	@echo "    -e HF_HOME=/dev/shm \\"
	@echo "    -e HF_TOKEN=\$$HF_TOKEN \\"
	@echo "    vllm/vllm-tpu:nightly \\"
	@echo "    vllm serve $(MODEL_PATH) \\"
	@echo "    --max-model-len 16384 \\"
	@echo "    --tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \\"
	@echo "    --disable_chunked_mm_input \\"
	@echo "    --max_num_batched_tokens 4096 \\"
	@echo "    --enable-auto-tool-choice \\"
	@echo "    --tool-call-parser gemma4 \\"
	@echo "    --reasoning-parser gemma4"

# Deploy the vLLM inference stack
deploy: deploy-tpu

# Every target below reads the instance through `gcloud compute instances`, not
# `gcloud compute tpus tpu-vm`. That is not cosmetic: a `ct5lp-*` instance created by
# `deploy-tpu` does **not** appear in the TPU API's node namespace at all, so a `tpu-vm
# describe` here returns not-found against a VM that is plainly RUNNING. The v6e rig this
# was ported from still has `tpu-vm` in these five targets and they cannot see anything it
# deploys — don't port that back.
#
# The IP also moves: TPU VMs expose it at networkEndpoints[0], GCE instances at
# networkInterfaces[0].accessConfigs[0].natIP.
IP_FORMAT = value(networkInterfaces[0].accessConfigs[0].natIP)

# Destroy the vLLM inference stack
destroy-tpu:
	@echo "🗑️  Destroying instance $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute instances delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--quiet

destroy: destroy-tpu

# Check status of the instance
status:
	@echo "🔍 Checking status of instance $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute instances describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE)

# Get the external IP address of the deployed instance
endpoint:
	@gcloud compute instances describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format='$(IP_FORMAT)'

# Run the performance benchmarking suite
benchmark:
	@echo "📊 Discovering endpoint for $(SERVICE_NAME)..."
	@IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='$(IP_FORMAT)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkInterfaces[0].networkIP)'); fi; \
	if [ -z "$$IP" ]; then echo "❌ Error: Could not find IP for $(SERVICE_NAME)"; exit 1; fi; \
	echo "🚀 Starting benchmark against http://$$IP:8000"; \
	python benchmarking_suite.py --url http://$$IP:8000 --model $(MODEL_PATH)

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model via the instance endpoint (assuming it's running)
query:
	@echo "🔍 Querying $(SERVICE_NAME) with prompt: \"$(PROMPT)\"..."
	@IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='$(IP_FORMAT)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkInterfaces[0].networkIP)'); fi; \
	curl -s -X POST "http://$$IP:8000/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test lint format tools mcp-config clean deploy-tpu deploy-tpu-spot deploy-tpu-ondemand deploy-tpu-flex deploy destroy-tpu destroy status endpoint benchmark run-container query help

help:
	@echo "Make targets:"
	@echo ""
	@echo "  install        - Install Python dependencies"
	@echo "  run            - Start the MCP server on stdio"
	@echo "  test           - Run the test suite (unittest)"
	@echo "  lint           - ruff check + ruff format --check + mypy"
	@echo "  format         - Apply ruff formatting and autofixes"
	@echo "  tools          - Regenerate GemmaTools.md from server.py"
	@echo "  mcp-config     - Write .mcp.json registering this server as $(MCP_SERVER_NAME)"
	@echo "  clean          - Remove caches and logs"
	@echo ""
	@echo "  deploy-tpu     - Create the v5e Compute Engine instance ($(SERVICE_NAME) in $(ZONE), $(TPU_PROVISIONING_MODEL))"
	@echo "  deploy-tpu-spot     - Same, as a preemptible Spot VM (cheapest published v5e rate)"
	@echo "  deploy-tpu-ondemand - Same, at standard on-demand pricing"
	@echo "  deploy-tpu-flex     - Same, queued through Dynamic Workload Scheduler (FLEX_START)"
	@echo "  destroy-tpu    - Delete the instance"
	@echo "  status         - Describe the instance"
	@echo "  endpoint       - Print the instance external IP"
	@echo "  run-container  - Print the docker run command to paste on the TPU VM"
	@echo "  benchmark      - Run benchmarking_suite.py against the live endpoint"
	@echo "  query          - Send one completion request (override with PROMPT=\"...\")"
	@echo ""
	@echo "For the MCP tool catalog see GemmaTools.md, or call the get_help tool."

# Configuration - tpu.env is the single source of truth; server.py and mcp-run.sh read
# the same file. Override per-invocation with e.g. `make status ZONE=us-central1-a`.
-include tpu.env

PROJECT_ID ?= $(GOOGLE_CLOUD_PROJECT)
REGION ?= $(GOOGLE_CLOUD_REGION)
ZONE ?= $(GOOGLE_CLOUD_ZONE)
MODEL_PATH ?= $(MODEL_NAME)
BUCKET_NAME ?= $(PROJECT_ID)-bucket
# The hand-provisioned instance is named after this rig directory, matching the RESOURCE_ID
# default server.py derives the same way. Set SERVICE_NAME in tpu.env (or on the command
# line) to point these targets at a VM whose name predates that, e.g.
#   make status SERVICE_NAME=<older-vm-name>
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

# Provisioning model for this hand-provisioned path. Both this and the MCP tools now go
# through `gcloud compute instances`, so the vocabularies finally agree: STANDARD, SPOT or
# FLEX_START. (They used to differ — the tpu-vm path took `standard|spot|reservation-bound`
# while the Queued Resource path took `flex-start|spot|on-demand`.)
TPU_PROVISIONING_MODEL ?= STANDARD

# Recommended deployment settings for vLLM on Cloud TPU v5p, via Compute Engine.
# ct5p-hightpu-1t-tpu is one v5p chip (95 GiB HBM) — the Cloud TPU API had no shape
# smaller than 4 chips. --scopes is required or the boot-time Secret Manager fetch fails.
deploy-tpu:
	@echo "🚀 Creating $(TPU_PROVISIONING_MODEL) TPU v5p instance ($(SERVICE_NAME)) in $(ZONE)..."
	gcloud compute instances create $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--machine-type=$(MACHINE_TYPE) \
		--image-family=$(IMAGE_FAMILY) \
		--image-project=$(IMAGE_PROJECT) \
		--boot-disk-size=$(BOOT_DISK_SIZE) \
		--boot-disk-type=$(BOOT_DISK_TYPE) \
		--scopes=$(INSTANCE_SCOPES) \
		--provisioning-model=$(TPU_PROVISIONING_MODEL)

# Spot: heavily discounted, preempted whenever the capacity is wanted back, no run limit.
deploy-tpu-spot:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=SPOT

# On-demand: full price, no preemption. Same as plain `make deploy-tpu`.
deploy-tpu-ondemand:
	@$(MAKE) deploy-tpu TPU_PROVISIONING_MODEL=STANDARD

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

# Destroy the vLLM inference stack
destroy-tpu:
	@echo "🗑️  Destroying TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute instances delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--quiet

destroy: destroy-tpu

# Check status of the TPU VM
status:
	@echo "🔍 Checking status of TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute instances describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE)

# Get the IP address of the deployed TPU VM
endpoint:
	@gcloud compute instances describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format='value(networkInterfaces[0].networkIP)'

# Run the performance benchmarking suite
benchmark:
	@echo "📊 Discovering endpoint for $(SERVICE_NAME)..."
	@IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkInterfaces[0].accessConfigs[0].natIP)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkInterfaces[0].networkIP)'); fi; \
	if [ -z "$$IP" ]; then echo "❌ Error: Could not find IP for $(SERVICE_NAME)"; exit 1; fi; \
	echo "🚀 Starting benchmark against http://$$IP:8000"; \
	python benchmarking_suite.py --url http://$$IP:8000 --model $(MODEL_PATH)

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model via the TPU VM endpoint (assuming it's running)
query:
	@echo "🔍 Querying $(SERVICE_NAME) with prompt: \"$(PROMPT)\"..."
	@IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkInterfaces[0].accessConfigs[0].natIP)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute instances describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkInterfaces[0].networkIP)'); fi; \
	curl -s -X POST "http://$$IP:8000/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test lint format tools mcp-config clean deploy-tpu deploy-tpu-spot deploy-tpu-ondemand deploy destroy-tpu destroy status endpoint benchmark run-container query help

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
	@echo "  deploy-tpu     - Create the TPU v5p instance ($(SERVICE_NAME) in $(ZONE), $(TPU_PROVISIONING_MODEL))"
	@echo "  deploy-tpu-spot     - Same, as a preemptible Spot VM (discounted, no run limit)"
	@echo "  deploy-tpu-ondemand - Same, at standard on-demand pricing"
	@echo "  destroy-tpu    - Delete the TPU v5p instance"
	@echo "  status         - Describe the TPU instance"
	@echo "  endpoint       - Print the TPU instance IP"
	@echo "  run-container  - Print the docker run command to paste on the TPU VM"
	@echo "  benchmark      - Run benchmarking_suite.py against the live endpoint"
	@echo "  query          - Send one completion request (override with PROMPT=\"...\")"
	@echo ""
	@echo "For the MCP tool catalog see GemmaTools.md, or call the get_help tool."

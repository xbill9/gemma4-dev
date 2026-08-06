# Configuration - tpu.env is the single source of truth; server.py and mcp-run.sh read
# the same file. Override per-invocation with e.g. `make status ZONE=europe-west4-a`.
-include tpu.env

PROJECT_ID ?= $(GOOGLE_CLOUD_PROJECT)
REGION ?= $(GOOGLE_CLOUD_REGION)
ZONE ?= $(GOOGLE_CLOUD_ZONE)
MODEL_PATH ?= $(MODEL_NAME)
BUCKET_NAME ?= $(PROJECT_ID)-bucket
SERVICE_NAME ?= tpu-2B-v5e1-devops-agent

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

clean:
	rm -rf __pycache__
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -rf {} +

# Recommended deployment settings for vLLM on Cloud TPU v5e (v5litepod)
deploy-tpu:
	@echo "🚀 Creating TPU v5e instance ($(SERVICE_NAME)) in $(ZONE)..."
	gcloud alpha compute tpus tpu-vm create $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--type=v5litepod \
		--topology=1x1 \
		--version=v2-alpha-tpuv5-lite

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
	gcloud compute tpus tpu-vm delete $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--quiet

destroy: destroy-tpu

# Check status of the TPU VM
status:
	@echo "🔍 Checking status of TPU VM $(SERVICE_NAME) in $(ZONE)..."
	gcloud compute tpus tpu-vm describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE)

# Get the IP address of the deployed TPU VM
endpoint:
	@gcloud compute tpus tpu-vm describe $(SERVICE_NAME) \
		--project=$(PROJECT_ID) \
		--zone=$(ZONE) \
		--format='value(networkEndpoints[0].ipAddress)'

# Run the performance benchmarking suite
benchmark:
	@echo "📊 Discovering endpoint for $(SERVICE_NAME)..."
	@IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].accessConfig.externalIp)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].ipAddress)'); fi; \
	if [ -z "$$IP" ]; then echo "❌ Error: Could not find IP for $(SERVICE_NAME)"; exit 1; fi; \
	echo "🚀 Starting benchmark against http://$$IP:8000"; \
	python benchmarking_suite.py --url http://$$IP:8000 --model $(MODEL_PATH)

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model via the TPU VM endpoint (assuming it's running)
query:
	@echo "🔍 Querying $(SERVICE_NAME) with prompt: \"$(PROMPT)\"..."
	@IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].accessConfig.externalIp)'); \
	if [ -z "$$IP" ]; then IP=$$(gcloud compute tpus tpu-vm describe $(SERVICE_NAME) --project=$(PROJECT_ID) --zone=$(ZONE) --format='value(networkEndpoints[0].ipAddress)'); fi; \
	curl -s -X POST "http://$$IP:8000/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test lint format tools clean deploy-tpu deploy destroy-tpu destroy status endpoint benchmark run-container query help

help:
	@echo "Make targets:"
	@echo ""
	@echo "  install        - Install Python dependencies"
	@echo "  run            - Start the MCP server on stdio"
	@echo "  test           - Run the test suite (unittest)"
	@echo "  lint           - ruff check + ruff format --check + mypy"
	@echo "  format         - Apply ruff formatting and autofixes"
	@echo "  tools          - Regenerate GemmaTools.md from server.py"
	@echo "  clean          - Remove caches and logs"
	@echo ""
	@echo "  deploy-tpu     - Create the TPU v5e instance ($(SERVICE_NAME) in $(ZONE))"
	@echo "  destroy-tpu    - Delete the TPU v5e instance"
	@echo "  status         - Describe the TPU instance"
	@echo "  endpoint       - Print the TPU instance IP"
	@echo "  run-container  - Print the docker run command to paste on the TPU VM"
	@echo "  benchmark      - Run benchmarking_suite.py against the live endpoint"
	@echo "  query          - Send one completion request (override with PROMPT=\"...\")"
	@echo ""
	@echo "For the MCP tool catalog see GemmaTools.md, or call the get_help tool."

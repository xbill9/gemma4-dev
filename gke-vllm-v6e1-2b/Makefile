# Configuration - tpu.env is the single source of truth; server.py and mcp-run.sh read
# the same file. Override per-invocation with e.g. `make status ZONE=europe-west4-a`.
-include tpu.env

PROJECT_ID ?= $(GOOGLE_CLOUD_PROJECT)
REGION ?= $(GOOGLE_CLOUD_REGION)
ZONE ?= $(GOOGLE_CLOUD_ZONE)
MODEL_PATH ?= $(MODEL_NAME)
BUCKET_NAME ?= $(PROJECT_ID)-bucket
# The hand-provisioned tpu-vm is named after this rig directory, matching the RESOURCE_ID
# default server.py derives the same way. Set SERVICE_NAME in tpu.env (or on the command
# line) to point these targets at a VM whose name predates that, e.g.
#   make status SERVICE_NAME=tpu-2B-v6e1-devops-agent
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

# ── The Compute Engine instance path is gone from this rig ───────────────────
#
# It lived here until 2026-08-25, inherited from the gce-vllm-v6e1-2b fork:
# deploy-tpu / deploy-tpu-spot / deploy-tpu-ondemand / deploy-tpu-flex,
# run-container, destroy-tpu, and a `status`/`endpoint`/`query` trio that shelled
# to `gcloud compute tpus tpu-vm` and could not have found a ct6e instance anyway.
# server.py now provisions GKE node pools, so a second Makefile vocabulary for a
# second control plane is exactly the trap the fork's own comment celebrated
# avoiding. The VM path still exists, in the rig that is named for it.
#
# The generic targets below are aliases for the gke-* ones, so `make deploy`,
# `make status` and `make query` mean what they mean in every sibling rig.

deploy: gke-up gke-deploy

# `destroy` releases the chip and keeps the cluster — the cheap, reversible half.
# Use `make destroy-cluster` to remove everything.
destroy: gke-down-pool

destroy-cluster: gke-down

status: gke-status

endpoint: gke-endpoint

benchmark: gke-benchmark

PROMPT ?= "What is Site Reliability Engineering?"

query: gke-query

# ── GKE: one cluster, one node, one v6e chip ─────────────────────────────────
# The path this rig is named for. Everything above provisions a Compute Engine
# instance (inherited from the gce-vllm-v6e1-2b fork); everything below asks GKE
# for a single-host TPU node pool and runs vLLM as a Pod on it. The two never
# see each other's resources — a node pool node is not a `compute instances`
# object you created, and `kubectl` is the only way to the endpoint.
GKE_CLUSTER_NAME ?= $(notdir $(CURDIR))
GKE_NODE_POOL ?= tpu-v6e-1
GKE_LOCATION ?= $(ZONE)
KUBECTL ?= kubectl

# kubectl and the GKE auth plugin are separate installs from gcloud itself.
gke-preflight:
	@command -v gcloud >/dev/null || { echo "❌ gcloud not found"; exit 1; }
	@command -v $(KUBECTL) >/dev/null || { echo "❌ kubectl not found. Install:"; \
		echo "   sudo apt-get install -y kubectl google-cloud-cli-gke-gcloud-auth-plugin"; exit 1; }
	@command -v gke-gcloud-auth-plugin >/dev/null || { echo "❌ gke-gcloud-auth-plugin not found. Install:"; \
		echo "   sudo apt-get install -y google-cloud-cli-gke-gcloud-auth-plugin"; exit 1; }
	@command -v envsubst >/dev/null || { echo "❌ envsubst not found (apt-get install gettext-base)"; exit 1; }
	@gcloud auth print-access-token >/dev/null 2>&1 || { echo "❌ gcloud credentials expired — run: gcloud auth login"; exit 1; }
	@echo "✅ kubectl, gke-gcloud-auth-plugin, envsubst and gcloud credentials all present"

# Creates the cluster and the TPU node pool. Idempotent.
gke-up:
	./gke/gke-up.sh

# Applies the vLLM Deployment + Service. Idempotent.
gke-deploy: gke-preflight
	./gke/gke-deploy.sh

gke-status:
	@$(KUBECTL) get nodes -l cloud.google.com/gke-tpu-accelerator=$(GKE_TPU_ACCELERATOR) -o wide || true
	@echo ""
	@$(KUBECTL) get pods -l app=vllm-gemma4 -o wide
	@echo ""
	@$(KUBECTL) get svc vllm-gemma4

gke-logs:
	$(KUBECTL) logs -f -l app=vllm-gemma4 --tail=200

# External IP of the LoadBalancer Service. Empty until GCP assigns one.
gke-endpoint:
	@$(KUBECTL) get svc vllm-gemma4 -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
	@echo ""

gke-port-forward:
	@echo "🔌 http://localhost:8000 -> vllm-gemma4:8000 (Ctrl-C to stop)"
	$(KUBECTL) port-forward svc/vllm-gemma4 8000:8000

# Same chat-completions smoke test the sibling rigs use. Raw /v1/completions returns an
# empty completion on -it models, so this one deliberately uses the chat endpoint.
# The prompt is JSON-encoded in python rather than interpolated into the -d string: the
# PROMPT default carries its own quotes, so the sibling `query` target's "$(PROMPT)" form
# emits invalid JSON when you don't override it.
gke-query:
	@IP=$$($(KUBECTL) get svc vllm-gemma4 -o jsonpath='{.status.loadBalancer.ingress[0].ip}'); \
	if [ -z "$$IP" ]; then echo "❌ no LoadBalancer IP yet — make gke-status"; exit 1; fi; \
	echo "🔍 querying http://$$IP:8000"; \
	python3 -c 'import json,sys,urllib.request; \
ip, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3].strip(chr(34)); \
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0.2}).encode(); \
req = urllib.request.Request(f"http://{ip}:8000/v1/chat/completions", body, {"Content-Type": "application/json"}); \
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=180)), indent=2))' \
		"$$IP" "$(MODEL_PATH)" "$(PROMPT)"

gke-benchmark:
	@IP=$$($(KUBECTL) get svc vllm-gemma4 -o jsonpath='{.status.loadBalancer.ingress[0].ip}'); \
	if [ -z "$$IP" ]; then echo "❌ no LoadBalancer IP yet — make gke-status"; exit 1; fi; \
	echo "🚀 benchmarking http://$$IP:8000"; \
	python benchmarking_suite.py --url http://$$IP:8000 --model $(MODEL_PATH)

# Deletes the TPU node pool only — releases the chip, keeps the cluster.
gke-down-pool:
	./gke/gke-down.sh --pool-only

# Deletes the whole cluster.
gke-down:
	./gke/gke-down.sh

.PHONY: gke-preflight gke-up gke-deploy gke-status gke-logs gke-endpoint gke-port-forward gke-query gke-benchmark gke-down-pool gke-down install run test lint format tools mcp-config clean deploy destroy destroy-cluster status endpoint benchmark query help

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
	@echo "  deploy         - gke-up + gke-deploy (cluster, node pool, model)"
	@echo "  destroy        - Delete the TPU node pool (releases the chip)"
	@echo "  destroy-cluster- Delete the whole cluster"
	@echo "  status/endpoint/query/benchmark - aliases for the gke-* targets below"
	@echo ""
	@echo "  GKE path (this rig's own — one cluster, one node, one v6e chip):"
	@echo "  gke-preflight  - Check kubectl, the GKE auth plugin, envsubst and gcloud creds"
	@echo "  gke-up         - Create cluster $(GKE_CLUSTER_NAME) + TPU node pool $(GKE_NODE_POOL) in $(GKE_LOCATION)"
	@echo "  gke-deploy     - Apply the vLLM Deployment + Service to it"
	@echo "  gke-status     - Show the TPU node, the pod and the Service"
	@echo "  gke-logs       - Follow the vLLM pod logs"
	@echo "  gke-endpoint   - Print the LoadBalancer external IP"
	@echo "  gke-query      - Chat-completions smoke test against that IP"
	@echo "  gke-benchmark  - Run benchmarking_suite.py against that IP"
	@echo "  gke-down-pool  - Delete the TPU node pool only (releases the chip)"
	@echo "  gke-down       - Delete the whole cluster"
	@echo "  benchmark      - Run benchmarking_suite.py against the live endpoint"
	@echo "  query          - Send one completion request (override with PROMPT=\"...\")"
	@echo ""
	@echo "For the MCP tool catalog see GemmaTools.md, or call the get_help tool."

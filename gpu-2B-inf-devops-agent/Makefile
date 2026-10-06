# Configuration - Update these or override via environment variables
SERVICE_NAME ?= inferentia-2b-devops-agent
MODEL_PATH ?= google/gemma-4-E2B-it
AWS_REGION ?= us-east-1

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

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Helper to deploy using local python script / server deployment tools
deploy:
	@echo "🚀 Deploying vLLM to AWS EC2 (Inferentia) for $(SERVICE_NAME)..."
	python3 -c "import asyncio, server; asyncio.run(server.deploy_vllm(service_name='$(SERVICE_NAME)', model_path='$(MODEL_PATH)', instance_type='inf2.8xlarge', market_type='spot'))"

status:
	@echo "🔍 Checking status of AWS EC2 instances for service $(SERVICE_NAME)..."
	python3 -c "import server; print(server.status_vllm(service_name='$(SERVICE_NAME)'))"

destroy:
	@echo "🗑️  Destroying/stopping AWS EC2 instances for service $(SERVICE_NAME)..."
	python3 -c "import asyncio, server; asyncio.run(server.destroy_vllm(service_name='$(SERVICE_NAME)'))"

# Cache active AWS SSO credentials to .aws_creds for use by other targets
auth:
	@echo "🔐 Caching AWS credentials to .aws_creds..."
	@bash save-aws-creds.sh

.PHONY: install run test clean deploy status destroy auth

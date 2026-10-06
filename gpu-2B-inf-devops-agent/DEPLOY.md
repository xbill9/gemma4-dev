# Deployment Guide: Self-Hosted vLLM on AWS Inferentia

This document summarizes the deployment configurations, architectures, and commands for the self-hosted vLLM inference server running on AWS EC2 using **AWS Inferentia** hardware accelerators.

---

## 🚀 AWS Inferentia Stack (EC2 inf2.xlarge Spot Instance)

*   **Instance Type:** `inf2.xlarge` (or larger up to `inf2.48xlarge`)
*   **Neuron Accelerator:** 1x AWS Inferentia2 Device (2 Neuron Cores)
*   **Operating System / AMI:** Deep Learning AMI (DLAMI) Neuron (Ubuntu 22.04)
*   **Container Image:** `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04`
*   **Model:** `google/gemma-4-E2B-it` (Unquantized)

### Neuron Run Command
On AWS Inferentia, models must be compiled for Neuron cores, and devices must be exposed to Docker using `--device`.

```bash
docker run -d --name vllm-server \
  --device /dev/neuron0 \
  --ipc=host \
  --restart always \
  -p 8080:8080 \
  -e HF_TOKEN="<your-hf-token>" \
  -e NEURON_CC_FLAGS="--model-type transformer" \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  -v /home/ubuntu/.cache/neuron:/root/.cache/neuron \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.16.0-neuronx-py312-sdk2.30.0-ubuntu24.04 \
  python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-E2B-it \
  --quantization neuron_quant \
  --max-model-len 16384 \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8080
```

---

## 🧩 Frameworks, Ecosystems & optimum-neuron

AWS Inferentia support is managed primarily through the **AWS Neuron SDK**, which provides seamless integration with popular deep learning frameworks like PyTorch, TensorFlow, and JAX.

### 📦 Ecosystem & Integration Paths
*   **PyTorch & TensorFlow**: Execute standard models with minimal code changes. For optimized deployment of transformer-based models, AWS provides the **Optimum Neuron** library.
*   **Libraries**: Neuron supports modern high-throughput deployment tools like **vLLM** and agentic AI frameworks / libraries from Hugging Face.
*   **Model Support**: State-of-the-art models, including Google Gemma 4, are natively supported.

---

## 🔗 Integration with SRE Agent

To connect the DevOps/SRE Agent to the newly deployed AWS endpoint:

1. Discover the public IP of your EC2 instance (e.g. `54.1.2.3`).
2. Export the endpoint URL and model name in your environment:
   ```bash
   export VLLM_BASE_URL="http://54.1.2.3:8080"
   export MODEL_NAME="google/gemma-4-E2B-it"
   ```
3. Start the agent:
   ```bash
   make run
   ```

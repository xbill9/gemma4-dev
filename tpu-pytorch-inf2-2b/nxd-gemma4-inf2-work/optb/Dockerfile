FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY optb_server.py /app/optb_server.py
COPY kv_pre_big.pt /workspace/kv_pre_big.pt
COPY kv_dec_big.pt /workspace/kv_dec_big.pt
COPY real-gemma4-E2B-it /workspace/real-gemma4-E2B-it
COPY opt_neuron /opt/aws/neuron
RUN ldconfig /opt/aws/neuron/lib || true
COPY torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl /tmp/torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl
RUN pip install "/tmp/torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl" && \
    pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
      "torch-neuronx==2.8.0.2.12.22436" "neuronx-cc==2.23.6484.0" "libneuronxla==2.2.15515.0" "transformers==5.13.0" && \
    rm "/tmp/torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl"
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib \
    PATH=/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin \
    KV_MAX=256 KV_BUCKET=64 KV_PRE_OUT=/workspace/kv_pre_big.pt KV_DEC_OUT=/workspace/kv_dec_big.pt \
    NEURON_RT_VISIBLE_CORES=0,1 PORT=8080
EXPOSE 8080
CMD ["python","/app/optb_server.py"]

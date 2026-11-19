# syntax=docker/dockerfile:1.10
# gpu-pytorch-g6-2b-q4w4a16's int4 engine on AWS's Lambda PyTorch image, with both checkpoints in the
# hub cache. engine_ext.py (an extension) starts ENGINE_CMD during Init; handler.py proxies each invocation.
ARG BASE_IMAGE
FROM python:3.13-slim AS fetch
RUN pip install --no-cache-dir huggingface_hub==1.31.0 hf_xet
COPY fetch.py /fetch.py
ARG MODEL_NAME
ARG MODEL_REVISION
ARG VOCAB_TABLES
ARG VOCAB_TABLES_REVISION
RUN python3 /fetch.py cache "$MODEL_NAME" "$MODEL_REVISION" /opt/hf \
 && python3 /fetch.py cache "$VOCAB_TABLES" "$VOCAB_TABLES_REVISION" /opt/hf \
 && rm -rf /opt/hf/hub/.locks

FROM ${BASE_IMAGE}
ARG TORCH_PIP_SPEC
# Hold torch at the image's build: the engine's pins must not replace it.
RUN python3 -c "import torch; print('torch==' + torch.__version__)" > /tmp/hold.txt \
 && pip install --no-cache-dir -c /tmp/hold.txt $TORCH_PIP_SPEC \
 && python3 -c "import torch, triton, transformers, compressed_tensors; print(torch.__version__, triton.__version__, transformers.__version__)" \
 && pip freeze > /opt/pip-freeze.txt
COPY --from=fetch /opt/hf /opt/hf
COPY engine/ /opt/engine/
COPY engine_ext.py /opt/engine-ext/engine_ext.py
COPY engine-ext /opt/extensions/pytorch-engine
COPY handler.py /var/task/handler.py
# The image is read-only on Lambda except /tmp: the hub cache is read offline, caches go to /tmp.
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE=/tmp/hf_modules \
    XDG_CACHE_HOME=/tmp/cache TRITON_CACHE_DIR=/tmp/triton HOME=/tmp

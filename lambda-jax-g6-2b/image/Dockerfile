# syntax=docker/dockerfile:1.10
# gpu-jax-g6-2b's pure-JAX Gemma 4 port on AWS's Lambda base image, with the reference checkpoint in the
# hub cache. engine_ext.py (an extension) starts ENGINE_CMD during Init; handler.py proxies each invocation.
ARG BASE_IMAGE
FROM python:3.13-slim AS fetch
RUN pip install --no-cache-dir huggingface_hub==1.31.0 hf_xet
COPY fetch.py /fetch.py
ARG MODEL_NAME
ARG MODEL_REVISION
RUN --mount=type=secret,id=hf,env=HF_TOKEN python3 /fetch.py cache "$MODEL_NAME" "$MODEL_REVISION" /opt/hf \
      "*.json" "*.jinja" "*.safetensors" \
 && rm -rf /opt/hf/hub/.locks

FROM ${BASE_IMAGE}
ARG JAX_PIP_SPEC
RUN pip install --no-cache-dir $JAX_PIP_SPEC \
 && python3 -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" \
 && pip freeze > /opt/pip-freeze.txt
COPY --from=fetch /opt/hf /opt/hf
COPY engine/ /opt/engine/
COPY engine_ext.py /opt/engine-ext/engine_ext.py
COPY engine-ext /opt/extensions/jax-engine
COPY handler.py /var/task/handler.py
# The image is read-only on Lambda except /tmp: the hub cache is read offline, caches go to /tmp.
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_MODULES_CACHE=/tmp/hf_modules \
    XDG_CACHE_HOME=/tmp/cache HOME=/tmp

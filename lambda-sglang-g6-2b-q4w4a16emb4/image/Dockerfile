# syntax=docker/dockerfile:1.10
# AWS's Lambda SGLang image with gpu-sglang-g6-2b-q4w4a16emb4's two patches and the checkpoint baked in.
# The image's engine extension starts sglang.launch_server during Init and its handler proxies each
# invocation; lmi.py sets every serving value as a function environment variable from tpu.env.
ARG BASE_IMAGE
FROM python:3.13-slim AS fetch
RUN pip install --no-cache-dir huggingface_hub==1.31.0 hf_xet
COPY fetch.py /fetch.py
ARG MODEL_NAME
ARG MODEL_REVISION
RUN --mount=type=secret,id=hf,env=HF_TOKEN python3 /fetch.py dir "$MODEL_NAME" "$MODEL_REVISION" /opt/model

FROM ${BASE_IMAGE}
COPY patches/ /tmp/patches/
# The patches fix SGLang's text-only Gemma 4 config and its int4 vocabulary tables; fail the build if
# either no longer applies.
RUN dnf install -y patch && dnf clean all \
 && SITE=$(python3 -c 'import os, sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))') \
 && patch -p1 --forward -d "$SITE" < /tmp/patches/text-config-per-layer.diff \
 && patch -p1 --forward -d "$SITE" < /tmp/patches/int4-embeddings.diff \
 && rm -rf /tmp/patches \
 && python3 -c "import sglang; print('sglang', sglang.__version__)"
COPY --from=fetch /opt/model /opt/model
ENV MODEL_ID=/opt/model

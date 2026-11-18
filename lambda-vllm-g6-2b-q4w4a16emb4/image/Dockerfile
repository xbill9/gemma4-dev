# AWS's Lambda vLLM image with the E2B checkpoint baked in. The image's own handler proxies each
# invocation to the vLLM server its engine extension starts during Init; lmi.py sets every serving
# value as a function environment variable from tpu.env.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY model/ /opt/model/
ENV MODEL_ID=/opt/model

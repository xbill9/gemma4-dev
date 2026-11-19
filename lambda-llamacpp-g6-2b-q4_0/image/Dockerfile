# syntax=docker/dockerfile:1.10
# llama-server (CUDA, sm_89) on AWS's Lambda base image, with the exact Q4_0 GGUF and its drafter.
# engine_ext.py (an extension) starts ENGINE_CMD during Init; handler.py proxies each invocation.
ARG BASE_IMAGE
FROM python:3.13-slim AS fetch
RUN pip install --no-cache-dir huggingface_hub==1.31.0 hf_xet
COPY fetch.py /fetch.py
ARG MODEL_NAME
ARG MODEL_REVISION
ARG MODEL_FILE
ARG MODEL_SHA256
ARG SPEC_DRAFT_HF
ARG SPEC_DRAFT_HF_REVISION
ARG SPEC_DRAFT_FILE
ARG SPEC_DRAFT_SHA256
RUN python3 /fetch.py file "$MODEL_NAME" "$MODEL_REVISION" "$MODEL_FILE" /opt/models "$MODEL_SHA256" \
 && python3 /fetch.py file "$SPEC_DRAFT_HF" "$SPEC_DRAFT_HF_REVISION" "$SPEC_DRAFT_FILE" /opt/models "$SPEC_DRAFT_SHA256" \
 && rm -rf /opt/models/.cache

FROM ${BASE_IMAGE} AS build
RUN dnf install -y --setopt=install_weak_deps=False cmake git gcc-c++ make \
      cuda-nvcc-13-0 cuda-cudart-devel-13-0 libcublas-devel-13-0 cuda-driver-devel-13-0 \
 && dnf clean all
ARG LLAMA_CPP_COMMIT
ARG CUDA_ARCH
RUN git clone -q https://github.com/ggml-org/llama.cpp /src && git -C /src checkout -q "$LLAMA_CPP_COMMIT"
# GGML_NATIVE=OFF with AVX2: the builder's CPU (Intel, AVX-512) is not the g6's (AMD EPYC).
RUN cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
      -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_HTML=OFF \
 && cmake --build /src/build -j "$(nproc)" --target llama-server \
 && ldd /src/build/bin/llama-server

FROM ${BASE_IMAGE}
COPY --from=build /src/build/bin/llama-server /opt/llama/bin/llama-server
COPY --from=build /usr/lib64/libgomp.so.1 /usr/lib64/
COPY --from=fetch /opt/models /opt/models
COPY engine_ext.py /opt/engine-ext/engine_ext.py
COPY engine-ext /opt/extensions/llamacpp-engine
COPY handler.py /var/task/handler.py
RUN ldd /opt/llama/bin/llama-server | grep -v "libcuda.so" | (! grep "not found")

# Self-hosted MinerU VLM parser for RunPod serverless.
# Based on MinerU's official image (its tested vLLM + dependency setup), plus the
# RunPod handler. cu129 base for broad RunPod GPU/driver compatibility.
FROM vllm/vllm-openai:v0.21.0-cu129

RUN apt-get update && \
    apt-get install -y fonts-noto-core fonts-noto-cjk fontconfig libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U 'mineru[core]>=3.4.0' --break-system-packages && \
    python3 -m pip install -U runpod boto3 requests --break-system-packages && \
    python3 -m pip cache purge

# Bake the MinerU VLM weights (MinerU2.5-Pro) into the image so cold workers do not
# download gigabytes on first request. Only the VLM backend is used, so skip the
# pipeline-only models (DocLayout-YOLO/PaddleOCR/UniMERNet). Weights are public.
RUN /bin/bash -c "mineru-models-download -s huggingface -m vlm"

ENV MINERU_MODEL_SOURCE=local

COPY handler.py /app/handler.py
WORKDIR /app

# Override the base image's vLLM-server entrypoint so our handler runs instead.
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Bake a CUDA 12.8 torch build (kernels for sm_70..sm_120) so the image runs on
# both older GPUs and Blackwell (RTX 5090, sm_120) with NO runtime install.
RUN pip install --break-system-packages --no-cache-dir --force-reinstall \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Install SAM3 dependencies WITHOUT touching torch/torchvision/cuda
# Using --no-deps for sam3 to prevent it from pulling CPU torch
RUN pip install --break-system-packages --no-cache-dir \
    boto3 \
    runpod \
    huggingface_hub \
    ftfy \
    regex \
    iopath \
    portalocker \
    safetensors \
    timm \
    einops \
    pycocotools \
    opencv-python-headless \
    Pillow \
    requests \
    decord \
    imageio-ffmpeg \
    && pip install --break-system-packages --no-cache-dir --no-deps sam3

# Create app directory
RUN mkdir -p /app

# Copy handler
COPY handler.py /app/handler.py

# Copy boot script (lightweight - just sets env and launches handler)
COPY boot_docker.py /app/boot.py

CMD ["python3", "-u", "/app/boot.py"]

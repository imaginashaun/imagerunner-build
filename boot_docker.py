#!/usr/bin/env python3
"""
Docker boot script for SAM 3.1 RunPod Serverless Worker.
All deps are already installed in the Docker image.
This just sets up HF auth and launches the handler.
"""
import os
import sys
import time

start = time.time()
print("[BOOT] SAM 3.1 Docker Boot", flush=True)

# Check workspace (network volume)
CACHE = "/workspace/.sam3_cache"
HF_CACHE = os.path.join(CACHE, "hf_hub")

if os.path.exists(HF_CACHE):
    os.environ["HF_HOME"] = HF_CACHE
    print(f"[BOOT] HF_HOME={HF_CACHE} (cached)", flush=True)
else:
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    os.makedirs("/tmp/hf_cache", exist_ok=True)
    print("[BOOT] HF_HOME=/tmp/hf_cache (will download)", flush=True)

# HuggingFace login
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    try:
        from huggingface_hub import login
        login(token=hf_token)
        print("[BOOT] HF login OK", flush=True)
    except Exception as e:
        print(f"[BOOT] HF login: {e}", flush=True)

# Check CUDA
try:
    import torch
    print(f"[BOOT] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[BOOT] GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"[BOOT] torch check: {e}", flush=True)

elapsed = time.time() - start
print(f"[BOOT] Ready in {elapsed:.1f}s. Launching handler...", flush=True)

# Launch handler
os.chdir("/app")
os.execv(sys.executable, [sys.executable, "-u", "/app/handler.py"])

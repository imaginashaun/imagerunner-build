"""RunPod serverless handler for self-hosted MinerU document/diagram parsing.

Input  (job["input"]):
  - image_url            : URL to fetch the page/diagram image, OR
  - image_base64         : base64 PNG/JPEG bytes
  - backend  (optional)  : MinerU backend, default "vlm-engine" (in-process VLM).
                           "hybrid-engine" = VLM + OCR (highest quality).
  - lesson_item_id (opt) : used only to namespace uploaded image crops.

Output:
  - middle        : MinerU middle.json (hierarchical; bboxes in ABSOLUTE pixels,
                    reading order via each block's `index`).
  - content_list  : MinerU content_list.json (flat, reading order, bbox 0-1000).
  - images        : { original_filename: public_s3_url } for every extracted crop.
  - backend       : the backend that was used.
On failure: { "error": str, ... }.
"""
import base64
import glob
import json
import os
import shutil
import subprocess
import tempfile
import traceback
import uuid

import boto3
import requests
import runpod

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET = os.environ.get("S3_SECRET_KEY", "")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC = os.environ.get("S3_PUBLIC_URL", "")


def _s3():
    return boto3.client(
        "s3", region_name=S3_REGION, aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET, endpoint_url=S3_ENDPOINT or None,
    )


def _upload(path, key, content_type="image/jpeg"):
    extra = {"ContentType": content_type}
    if not S3_ENDPOINT or "r2.cloudflarestorage" not in S3_ENDPOINT:
        extra["ACL"] = "public-read"
    with open(path, "rb") as f:
        _s3().upload_fileobj(f, S3_BUCKET, key, ExtraArgs=extra)
    if S3_PUBLIC:
        return f"{S3_PUBLIC.rstrip('/')}/{key}"
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def _download(url, path):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def _find(out_dir, *suffixes):
    for suffix in suffixes:
        hits = sorted(glob.glob(f"{out_dir}/**/*{suffix}", recursive=True))
        if hits:
            return hits[0]
    return None


def detect_shapes(image_path):
    """Detect candidate WHOLE-ELEMENT boxes in a diagram via classical CV, so the
    semantic detector's rough boxes can be snapped to real element extents (a box
    outline, a filled block, an icon). Two passes — outlined boxes (Canny) and filled
    shapes (non-white) — union'd. Returns [[x1,y1,x2,y2], ...] in ORIGINAL pixels.
    Best-effort: returns [] if OpenCV is unavailable.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return [], None
    im = cv2.imread(image_path)
    if im is None:
        return [], None
    H, W = im.shape[:2]
    area = float(W * H)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    boxes = []

    def collect(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            a = w * h
            if a < 0.004 * area or a > 0.45 * area:
                continue
            if w < 24 or h < 18 or (w / float(h)) > 20:
                continue
            boxes.append([int(x), int(y), int(x + w), int(y + h)])

    edges = cv2.dilate(cv2.Canny(gray, 30, 120), np.ones((3, 3), np.uint8), iterations=2)
    collect(edges)
    nonwhite = cv2.morphologyEx((gray < 238).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    collect(nonwhite)
    return boxes, [W, H]


def handler(job):
    work = None
    try:
        inp = job.get("input", {}) or {}
        backend = str(inp.get("backend") or "vlm-engine")
        lid = inp.get("lesson_item_id")

        work = tempfile.mkdtemp(prefix="mineru_")
        img = os.path.join(work, "input.png")
        if inp.get("image_url"):
            _download(inp["image_url"], img)
        elif inp.get("image_base64"):
            with open(img, "wb") as f:
                f.write(base64.b64decode(inp["image_base64"]))
        else:
            return {"error": "missing image_url or image_base64"}

        out = os.path.join(work, "out")
        os.makedirs(out, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("MINERU_MODEL_SOURCE", "local")

        proc = subprocess.run(
            ["mineru", "-p", img, "-o", out, "-b", backend],
            capture_output=True, text=True, env=env, timeout=900,
        )
        if proc.returncode != 0:
            return {
                "error": "mineru_failed",
                "stderr": (proc.stderr or "")[-2500:],
                "stdout": (proc.stdout or "")[-1000:],
                "backend": backend,
            }

        mid_p = _find(out, "_middle.json", "middle.json")
        cl_p = _find(out, "_content_list.json", "content_list.json")
        middle = json.load(open(mid_p)) if mid_p else None
        content_list = json.load(open(cl_p)) if cl_p else None

        images = {}
        for p in glob.glob(f"{out}/**/images/*", recursive=True):
            if not os.path.isfile(p):
                continue
            name = os.path.basename(p)
            key = f"mineru/{lid or 'x'}/{uuid.uuid4().hex}_{name}"
            try:
                images[name] = _upload(p, key)
            except Exception as e:  # noqa: BLE001 - best-effort per crop
                images[name] = None
                print(f"[MINERU] upload failed for {name}: {e}", flush=True)

        shape_boxes, image_size = detect_shapes(img)

        return {
            "middle": middle,
            "content_list": content_list,
            "images": images,
            "shape_boxes": shape_boxes,
            "image_size": image_size,
            "backend": backend,
        }
    except subprocess.TimeoutExpired:
        return {"error": "mineru_timeout"}
    except Exception as e:  # noqa: BLE001 - report any failure to the caller
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()[-1500:]}
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})

"""
SAM 3.1 RunPod Serverless Handler (Diagnostic Mode)
=====================================================
Wraps model loading in try/except for debugging.
"""

import io
import json
import os
import sys
import traceback
import uuid

print("[HANDLER] Starting handler.py...", flush=True)
print(f"[HANDLER] Python: {sys.version}", flush=True)
print(f"[HANDLER] CWD: {os.getcwd()}", flush=True)
print(f"[HANDLER] HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}", flush=True)

# Check if network volume cache is accessible
CACHE = "/workspace/.sam3_cache"
print(f"[HANDLER] Cache exists: {os.path.exists(CACHE)}", flush=True)
if os.path.exists(CACHE):
    print(f"[HANDLER] Cache contents: {os.listdir(CACHE)}", flush=True)

# Import basic deps
try:
    import boto3
    import numpy as np
    import requests as http_requests
    import torch
    from botocore.config import Config as BotoConfig
    from PIL import Image
    print(f"[HANDLER] Base imports OK. torch={torch.__version__}, cuda={torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[HANDLER] BASE IMPORT ERROR: {e}", flush=True)
    traceback.print_exc()

# ── S3/R2 Configuration ─────────────────────────────────────────────────────
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL", "")
MASK_PREFIX = os.environ.get("MASK_PREFIX", "masks")

s3_client = None


def _get_s3():
    global s3_client
    if s3_client is None:
        kwargs = {
            "region_name": S3_REGION,
            "aws_access_key_id": S3_ACCESS_KEY,
            "aws_secret_access_key": S3_SECRET_KEY,
            "config": BotoConfig(
                connect_timeout=15,
                read_timeout=60,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        }
        if S3_ENDPOINT:
            kwargs["endpoint_url"] = S3_ENDPOINT
        s3_client = boto3.client("s3", **kwargs)
    return s3_client


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Model Loading (lazy) ─────────────────────────────────────────────────────
# SAM3 is only needed for segmentation, so it is loaded on first segmentation
# request — video rendering skips the multi-GB model download entirely.
sam3_model = None
sam3_processor = None
model_error = None
_model_init_done = False


def ensure_model():
    global sam3_model, sam3_processor, model_error, _model_init_done
    if _model_init_done:
        return
    _model_init_done = True
    try:
        print("[INIT] Setting up CUDA...", flush=True)
        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            props = torch.cuda.get_device_properties(0)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            print(f"[INIT] GPU: {props.name}, {vram // 1024**2} MB VRAM", flush=True)

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            print("[INIT] HuggingFace authenticated", flush=True)

        print("[INIT] Importing SAM3...", flush=True)
        import sam3 as sam3_module
        print(f"[INIT] SAM3 module loaded from: {sam3_module.__file__}", flush=True)

        bpe_path = os.path.join(os.path.dirname(sam3_module.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            bpe_path = "/tmp/bpe_simple_vocab_16e6.txt.gz"
            if not os.path.exists(bpe_path):
                print("[INIT] BPE file not found locally, downloading from OpenAI...", flush=True)
                import urllib.request
                url = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
                urllib.request.urlretrieve(url, bpe_path)
                print("[INIT] BPE file downloaded successfully", flush=True)

        print("[INIT] Building SAM3 model...", flush=True)
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        sam3_processor = Sam3Processor(sam3_model, confidence_threshold=0.3)
        print("[INIT] ✅ SAM 3.1 model loaded and ready", flush=True)

    except Exception as e:
        model_error = str(e)
        print(f"[INIT] ❌ MODEL LOADING FAILED: {e}", flush=True)
        traceback.print_exc()
        print("[INIT] Handler will run in diagnostic mode", flush=True)


# ── Inference Functions ──────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    "diagram component", "labeled part", "arrow", "chart element",
    "cell structure", "molecule", "organ", "graph axis",
    "illustration", "symbol", "figure", "equation",
    "organism", "anatomical structure", "chemical bond", "process step",
]


def _calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def auto_segment_image(image, custom_prompts=None):
    prompts = custom_prompts or DEFAULT_PROMPTS
    all_objects = []
    seen_boxes = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = sam3_processor.set_image(image)

        for prompt_text in prompts:
            try:
                sam3_processor.reset_all_prompts(inference_state)
                state = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt_text)

                masks = state.get("masks")
                boxes = state.get("boxes")
                scores = state.get("scores")

                if masks is None or len(masks) == 0:
                    continue

                for i in range(len(masks)):
                    score = float(scores[i]) if scores is not None else 0.5
                    if score < 0.3:
                        continue

                    bbox = boxes[i].cpu().tolist()
                    is_dup = any(_calc_iou(bbox, sb) > 0.7 for sb in seen_boxes)
                    if is_dup:
                        continue

                    mask = masks[i].squeeze(0)
                    seen_boxes.append(bbox)
                    all_objects.append({
                        "label": prompt_text,
                        "bbox": bbox,
                        "score": score,
                        "mask": mask,
                    })

            except Exception as e:
                print(f"[WARN] Prompt '{prompt_text}' failed: {e}")
                continue

    all_objects.sort(key=lambda x: x["score"], reverse=True)
    return all_objects


def create_masked_crop(image, mask, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    mask_np = mask.cpu().numpy().astype(np.uint8) if isinstance(mask, torch.Tensor) else np.array(mask, dtype=np.uint8)

    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_mask = mask_np[y1:y2, x1:x2]
    cropped_rgba = cropped_image.convert("RGBA")
    data = np.array(cropped_rgba)
    data[:, :, 3] = cropped_mask * 255
    return Image.fromarray(data, "RGBA")


def upload_mask_to_s3(masked_img):
    buffer = io.BytesIO()
    masked_img.save(buffer, format="PNG")
    buffer.seek(0)

    s3_key = f"{MASK_PREFIX}/{uuid.uuid4()}.png"
    s3 = _get_s3()

    extra_args = {"ContentType": "image/png"}
    if not S3_ENDPOINT or "r2.cloudflarestorage" not in S3_ENDPOINT:
        extra_args["ACL"] = "public-read"

    s3.upload_fileobj(buffer, S3_BUCKET_NAME, s3_key, ExtraArgs=extra_args)

    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL.rstrip('/')}/{s3_key}"
    elif S3_ENDPOINT and "r2.cloudflarestorage" in S3_ENDPOINT:
        return f"https://{S3_BUCKET_NAME}.r2.cloudflarestorage.com/{s3_key}"
    else:
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"


# ── Box-prompted segmentation (Grounded-SAM) ─────────────────────────────────
# A caller (e.g. a VLM that read the whole diagram) provides a box per region.
# SAM 3 turns each box into a precise alpha mask via the native geometric prompt,
# so complex objects lift/separate cleanly instead of being rectangular crops.

def _rect_object(image, idx, label, bbox, description):
    """Fallback: a rectangular RGBA crop of the box (used if SAM returns nothing)."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image.crop((x1, y1, x2, y2)).convert("RGBA")
    url = upload_mask_to_s3(crop)
    return {
        "id": f"obj_{idx + 1}", "label": label, "bbox": [x1, y1, x2, y2],
        "masked_image_url": url, "score": 0.3, "description": description, "mask_kind": "rect",
    }


def segment_with_boxes(image, regions):
    """Box-prompted SAM 3 segmentation.

    regions: [{"label": str, "bbox": [x1,y1,x2,y2] (pixels), "description"?: str}].
    For each box we call ``add_geometric_prompt`` (box as normalized cx,cy,w,h) and
    keep the returned mask whose box best matches the requested region. Falls back
    to a rectangular crop when SAM returns nothing for a box.
    """
    W, H = image.size
    objects = []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam3_processor.set_image(image)

        for idx, region in enumerate(regions):
            label = str(region.get("label") or f"region {idx + 1}")
            description = str(region.get("description") or "")
            bb = region.get("bbox") or []
            if len(bb) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bb]
            except (TypeError, ValueError):
                continue

            # Clamp + order the requested box to the image.
            x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue

            requested = [x1, y1, x2, y2]
            # SAM 3 geometric prompt wants a normalized [cx, cy, w, h] box.
            norm_box = [((x1 + x2) / 2.0) / W, ((y1 + y2) / 2.0) / H, (x2 - x1) / W, (y2 - y1) / H]

            try:
                sam3_processor.reset_all_prompts(state)
                out = sam3_processor.add_geometric_prompt(box=norm_box, label=True, state=state)
                masks = out.get("masks")
                boxes = out.get("boxes")
                scores = out.get("scores")

                if masks is None or len(masks) == 0:
                    obj = _rect_object(image, idx, label, requested, description)
                    if obj:
                        objects.append(obj)
                    continue

                # Pick the returned mask whose box best overlaps the requested box.
                best_i, best_key = 0, -1.0
                for i in range(len(masks)):
                    ob = boxes[i].cpu().tolist() if hasattr(boxes[i], "cpu") else list(boxes[i])
                    score = float(scores[i]) if scores is not None else 0.5
                    key = _calc_iou(requested, ob) + 0.001 * score
                    if key > best_key:
                        best_key, best_i = key, i

                obox = boxes[best_i].cpu().tolist() if hasattr(boxes[best_i], "cpu") else list(boxes[best_i])
                mask = masks[best_i].squeeze(0)
                masked = create_masked_crop(image, mask, obox)
                if masked is None:
                    obj = _rect_object(image, idx, label, requested, description)
                    if obj:
                        objects.append(obj)
                    continue

                url = upload_mask_to_s3(masked)
                objects.append({
                    "id": f"obj_{idx + 1}",
                    "label": label,
                    "bbox": [float(v) for v in obox],
                    "masked_image_url": url,
                    "score": round(float(scores[best_i]) if scores is not None else 0.5, 4),
                    "description": description,
                    "mask_kind": "sam",
                })

            except Exception as e:
                print(f"[BOX] region {idx} ('{label}') failed: {e}", flush=True)
                try:
                    obj = _rect_object(image, idx, label, requested, description)
                    if obj:
                        objects.append(obj)
                except Exception:
                    pass

    return objects


# ══════════════════════════════════════════════════════════════════════════════
# Image-narration VIDEO rendering
# ══════════════════════════════════════════════════════════════════════════════
# Renders the animated "image narration" (segmentation masks animated over the
# base image, synced to narration audio) into an MP4. Pure CPU compositing with
# Pillow + a static ffmpeg (imageio-ffmpeg); no browser, no system packages.

import math
import re
import shutil
import subprocess
import tempfile

# ── Premium narration animation aesthetic ─────────────────────────────────────
# Colour is reserved for the pointer + annotation; content cards keep their true
# colours with soft shadows, so the result reads as elegant rather than garish.
# Cropping uses a general ROUNDED-RECT shape from the base image (the jagged SAM
# alpha mask is set aside) and every highlight "lifts" the card off the page.
V_ACCENT = (255, 193, 84)        # warm amber — pointer + annotation accent
V_ACCENT_HI = (255, 226, 170)    # pointer highlight tint
V_ACCENT_EDGE = (120, 78, 12)    # pointer dark edge
V_INK = (17, 19, 26)             # dark glass for the annotation pill

# Timing (seconds)
V_FADE = 0.32                    # fade-in for a gently-lifted card
V_SETTLE = 0.55                  # gentle in-place lift settle
V_BORDER_DUR = 0.42             # hero: brief in-place beat before it flies up
V_LIFT_DUR = 0.85               # hero: travel-to-centre duration
V_GLIDE = 0.55                  # pointer glide between targets
V_ANNO_IN, V_ANNO_HOLD, V_ANNO_OUT = 0.3, 1.7, 0.55   # brief annotation envelope

# Geometry
V_RADIUS_FRAC = 0.05            # card corner radius (fraction of min side)
V_RADIUS_MIN, V_RADIUS_MAX = 12, 48
V_GENTLE_SCALE = 0.085          # extra scale when a card lifts in place
V_GENTLE_RISE = 0.02            # how far it rises (fraction of H)
V_RAISE_FILL = 0.62             # hero card target fill of the stage
V_RAISE_MIN, V_RAISE_MAX = 0.5, 2.6
V_DIM_GENTLE = 0.72            # base dim while a card is gently highlighted
V_DIM_HERO = 0.26              # base dim under a hero lift
V_AMBIENT = 0.02              # slow ambient breath/pan so frames are never static


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _v_download(url, path):
    r = http_requests.get(url, timeout=90)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def _v_load_image(url):
    r = http_requests.get(url, timeout=90)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def _media_duration(path, ffmpeg):
    """Parse a media file's duration (seconds) from ffmpeg's stderr banner."""
    res = subprocess.run([ffmpeg, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", res.stderr or "")
    if m:
        h, mn, s = m.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)
    return 0.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _ease_out_back(p):
    p = _clamp(p, 0.0, 1.0)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def _ease_out(p):
    p = _clamp(p, 0.0, 1.0)
    return 1 - (1 - p) ** 3


def _ease_in_out(p):
    p = _clamp(p, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def _with_opacity(img, op):
    if op >= 1.0:
        return img
    op = max(0.0, op)
    a = img.split()[3].point(lambda v: int(v * op))
    out = img.copy()
    out.putalpha(a)
    return out


def _brightness(img, factor):
    from PIL import ImageEnhance
    rgb = Image.merge("RGB", img.split()[:3])
    rgb = ImageEnhance.Brightness(rgb).enhance(factor)
    out = Image.merge("RGBA", (*rgb.split(), img.split()[3]))
    return out


# ── Fonts (bundled DejaVu, with system fallbacks) ────────────────────────────
_FONT_CACHE = {}


def _font(size, bold=False):
    key = (int(size), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "assets", name),
        os.path.join("/app/assets", name),
        "/usr/share/fonts/truetype/dejavu/" + name,
        "/System/Library/Fonts/Supplemental/Arial%s.ttf" % (" Bold" if bold else ""),
    ]
    fnt = None
    for c in candidates:
        try:
            fnt = ImageFont.truetype(c, int(size))
            break
        except Exception:
            continue
    if fnt is None:
        fnt = ImageFont.load_default()
    _FONT_CACHE[key] = fnt
    return fnt


# ── Shape + card helpers ─────────────────────────────────────────────────────
def _vgrad(size, top_color, bottom_color):
    w, h = size
    strip = Image.new("RGBA", (1, max(1, h)))
    for y in range(max(1, h)):
        f = y / max(1, h - 1)
        strip.putpixel((0, y), (
            int(top_color[0] + (bottom_color[0] - top_color[0]) * f),
            int(top_color[1] + (bottom_color[1] - top_color[1]) * f),
            int(top_color[2] + (bottom_color[2] - top_color[2]) * f),
            255,
        ))
    return strip.resize((w, h))


def _rounded_mask(w, h, radius):
    from PIL import ImageDraw
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return m


def _card_radius(w, h):
    return int(_clamp(min(w, h) * V_RADIUS_FRAC, V_RADIUS_MIN, V_RADIUS_MAX))


def _make_card(base, x1, y1, x2, y2):
    """A general ROUNDED-RECT crop of the base image (the lifted 'card')."""
    crop = base.crop((x1, y1, x2, y2)).convert("RGBA")
    w, h = crop.size
    crop.putalpha(_rounded_mask(w, h, _card_radius(w, h)))
    return crop


def _shadow_for(card_img, blur=18):
    """Soft black silhouette used as the card's drop shadow."""
    from PIL import ImageFilter
    a = card_img.split()[3].filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", card_img.size, (0, 0, 0, 255))
    layer.putalpha(a)
    return layer


def _rim_layer(card_img, strength=0.5):
    """A subtle light rim around the card edge to sell its lift off the page."""
    from PIL import ImageChops, ImageFilter
    a = card_img.split()[3]
    edge = ImageChops.subtract(a.filter(ImageFilter.MaxFilter(5)), a)
    layer = Image.new("RGBA", card_img.size, (255, 255, 255, 255))
    layer.putalpha(edge.point(lambda v: int(v * strength)))
    return layer


def _paste_card(frame, card, left, top, op, shadow=0.5, shadow_blur=20, shadow_dy=16, rim=True):
    sh = _with_opacity(_shadow_for(card, shadow_blur), shadow * op)
    frame.alpha_composite(sh, (left, top + shadow_dy))
    if rim:
        frame.alpha_composite(_with_opacity(_rim_layer(card), 0.9 * op), (left, top))
    frame.alpha_composite(_with_opacity(card, op), (left, top))


# ── Scene -> render inputs ───────────────────────────────────────────────────
def _build_objs(scene, base, W, H):
    """Rounded-rect base crops for every object (no SAM masks)."""
    objs = {}
    for o in scene.get("objects", []):
        x1, y1, x2, y2 = [float(v) for v in o["bbox"]]
        x1 = int(_clamp(round(x1), 0, W - 2))
        y1 = int(_clamp(round(y1), 0, H - 2))
        x2 = int(_clamp(round(x2), x1 + 1, W))
        y2 = int(_clamp(round(y2), y1 + 1, H))
        objs[o["id"]] = {
            "img": _make_card(base, x1, y1, x2, y2),
            "left": x1, "top": y1, "w": x2 - x1, "h": y2 - y1,
            "label": str(o.get("label") or ""),
        }
    return objs


def _build_cues(scene, offsets, total):
    cues = []
    for c in scene.get("cues", []):
        if c.get("time") is not None and c.get("audio_index") is not None and c["audio_index"] < len(offsets):
            tt = offsets[c["audio_index"]] + float(c["time"])
        else:
            tt = float(c.get("progress", 0)) * total
        cues.append({**c, "t": tt})
    cues.sort(key=lambda c: c["t"])
    by_obj = {}
    for c in cues:
        by_obj.setdefault(c["object_id"], []).append(c)
    focus_seq = [c for c in cues if c.get("action") == "in"]
    return by_obj, focus_seq


def _active_state(t, by_obj):
    active = {}
    for oid, seq in by_obj.items():
        a = None
        for c in seq:
            if c["t"] <= t + 1e-6:
                a = c
            else:
                break
        if a and a["action"] == "in":
            active[oid] = a
    presented, pres_cue = None, None
    for oid, c in active.items():
        if c["type"] == "raise" and (pres_cue is None or c["t"] >= pres_cue["t"]):
            pres_cue, presented = c, oid
    return active, presented, pres_cue


# ── Card geometry over time ──────────────────────────────────────────────────
def _gentle_rect(o, elapsed, t, H):
    p = _ease_out(_clamp(elapsed / V_SETTLE, 0.0, 1.0))
    scale = 1.0 + V_GENTLE_SCALE * p
    rise = (V_GENTLE_RISE * H) * p
    bob = math.sin(t * 1.5) * (H * 0.0035) * p
    nw = max(1, int(round(o["w"] * scale)))
    nh = max(1, int(round(o["h"] * scale)))
    cx = o["left"] + o["w"] / 2
    cy = o["top"] + o["h"] / 2 - rise + bob
    return nw, nh, int(round(cx - nw / 2)), int(round(cy - nh / 2))


def _hero_rect(o, elapsed, t, W, H):
    gw, gh = o["w"] / W, o["h"] / H
    target = _clamp(min(V_RAISE_FILL / gw, V_RAISE_FILL / gh), V_RAISE_MIN, V_RAISE_MAX)
    lp = _ease_out_back(_clamp((elapsed - V_BORDER_DUR) / V_LIFT_DUR, 0.0, 1.0))
    scale = 1.0 + (target - 1.0) * lp
    start_cx, start_cy = o["left"] + o["w"] / 2, o["top"] + o["h"] / 2
    settled = max(0.0, elapsed - (V_BORDER_DUR + V_LIFT_DUR))
    bob = math.sin(settled * 1.4) * (H * 0.01)
    end_cx, end_cy = W / 2, H * 0.54 + bob
    cx = start_cx + (end_cx - start_cx) * lp
    cy = start_cy + (end_cy - start_cy) * lp
    nw = max(1, int(round(o["w"] * scale)))
    nh = max(1, int(round(o["h"] * scale)))
    return nw, nh, int(round(cx - nw / 2)), int(round(cy - nh / 2))


# ── Floating 3D pointer ──────────────────────────────────────────────────────
_ARROW_CACHE = {}


def _arrow_img(size):
    """A glossy, bevelled arrow pointing DOWN (tip at bottom centre), cached."""
    size = int(size)
    if size in _ARROW_CACHE:
        return _ARROW_CACHE[size]
    from PIL import ImageChops, ImageDraw, ImageFilter
    Wd = size
    Hd = int(size * 1.3)
    pad = int(size * 0.45)
    canvas = (Wd + pad * 2, Hd + pad * 2)
    ox, oy = pad, pad
    shaft_w = Wd * 0.42
    shaft_h = Hd * 0.46
    pts = [
        (ox + Wd / 2 - shaft_w / 2, oy),
        (ox + Wd / 2 + shaft_w / 2, oy),
        (ox + Wd / 2 + shaft_w / 2, oy + shaft_h),
        (ox + Wd, oy + shaft_h),
        (ox + Wd / 2, oy + Hd),
        (ox, oy + shaft_h),
        (ox + Wd / 2 - shaft_w / 2, oy + shaft_h),
    ]
    sil = Image.new("L", canvas, 0)
    ImageDraw.Draw(sil).polygon(pts, fill=255)

    body = Image.composite(_vgrad(canvas, V_ACCENT_HI, V_ACCENT), Image.new("RGBA", canvas, (0, 0, 0, 0)), sil)

    edge = ImageChops.subtract(sil.filter(ImageFilter.MaxFilter(5)), sil)
    outline = Image.new("RGBA", canvas, V_ACCENT_EDGE + (255,))
    outline.putalpha(edge)

    spec = Image.new("L", canvas, 0)
    ImageDraw.Draw(spec).polygon([
        (ox + Wd / 2 - shaft_w / 2 + 2, oy + 2),
        (ox + Wd / 2 - 2, oy + 2),
        (ox + Wd / 2 - 2, oy + shaft_h),
        (ox + Wd / 2 - shaft_w / 2 + 2, oy + shaft_h),
    ], fill=90)
    spec = ImageChops.multiply(spec, sil)
    specL = Image.new("RGBA", canvas, (255, 255, 255, 255))
    specL.putalpha(spec.filter(ImageFilter.GaussianBlur(1)))

    sh = sil.filter(ImageFilter.GaussianBlur(int(size * 0.13)))
    shadow = Image.new("RGBA", canvas, (0, 0, 0, 255))
    shadow.putalpha(sh.point(lambda v: int(v * 0.5)))

    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    out.alpha_composite(shadow, (0, int(size * 0.12)))
    out.alpha_composite(outline)
    out.alpha_composite(body)
    out.alpha_composite(specL)
    # tip offset from the image's top-left (used to anchor the tip on target).
    meta = {"tip": (canvas[0] / 2, oy + Hd), "size": canvas}
    _ARROW_CACHE[size] = (out, meta)
    return _ARROW_CACHE[size]


def _focus_at(t, focus_seq):
    cur, prev = None, None
    for c in focus_seq:
        if c["t"] <= t + 1e-6:
            prev, cur = cur, c
        else:
            break
    return cur, prev


def _target_top_center(oid, objs, by_obj, t, W, H):
    """Where the pointer should aim: the top-centre of the object's card NOW."""
    o = objs.get(oid)
    if not o:
        return (W / 2, H * 0.2)
    cue = None
    for c in by_obj.get(oid, []):
        if c["t"] <= t + 1e-6:
            cue = c
        else:
            break
    e = (t - cue["t"]) if cue else 0.0
    if cue and cue["type"] == "raise" and e >= V_BORDER_DUR:
        nw, nh, left, top = _hero_rect(o, e, t, W, H)
    else:
        nw, nh, left, top = _gentle_rect(o, e, t, H)
    return (left + nw / 2, top)


def _draw_pointer(frame, focus_seq, objs, by_obj, t, W, H):
    cur, prev = _focus_at(t, focus_seq)
    if not cur:
        return None
    arrow, meta = _arrow_img(int(_clamp(min(W, H) * 0.058, 34, 112)))
    aw, ah = meta["size"]
    tipx_off, tipy_off = meta["tip"]

    cur_pt = _target_top_center(cur["object_id"], objs, by_obj, t, W, H)
    if prev:
        prev_pt = _target_top_center(prev["object_id"], objs, by_obj, t, W, H)
        gp = _ease_in_out(_clamp((t - cur["t"]) / V_GLIDE, 0.0, 1.0))
    else:
        prev_pt, gp = cur_pt, 1.0

    tip_x = prev_pt[0] + (cur_pt[0] - prev_pt[0]) * gp
    tip_y = prev_pt[1] + (cur_pt[1] - prev_pt[1]) * gp
    gap = ah * 0.10 + math.sin(t * 2.3) * (ah * 0.05)   # hover above + idle bob
    tip_y -= gap

    left = int(round(tip_x - tipx_off))
    top = int(round(tip_y - tipy_off))
    op = _clamp((t - cur["t"]) / 0.3, 0.0, 1.0) if not prev else 1.0
    frame.alpha_composite(_with_opacity(arrow, op), (left, top))
    return (tip_x, top)   # anchor for the annotation (arrow body top)


def _draw_annotation(frame, text, anchor, t, t0, W, H):
    e = t - t0
    if e < 0:
        return
    if e < V_ANNO_IN:
        op = e / V_ANNO_IN
    elif e < V_ANNO_IN + V_ANNO_HOLD:
        op = 1.0
    elif e < V_ANNO_IN + V_ANNO_HOLD + V_ANNO_OUT:
        op = 1.0 - (e - V_ANNO_IN - V_ANNO_HOLD) / V_ANNO_OUT
    else:
        return
    text = text.strip()
    if not text:
        return
    from PIL import ImageDraw
    fs = int(_clamp(min(W, H) * 0.03, 16, 42))
    fnt = _font(fs, bold=True)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tb = d.textbbox((0, 0), text, font=fnt)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    dot = int(fs * 0.42)
    padx, pady = int(fs * 0.7), int(fs * 0.5)
    gap = int(fs * 0.45)
    pillw = padx + dot + gap + tw + padx
    pillh = th + pady * 2
    anchor_x, arrow_top = anchor
    x0 = int(_clamp(anchor_x - pillw / 2, 8, W - pillw - 8))
    y0 = int(arrow_top - pillh - int(fs * 0.45))
    if y0 < 8:
        y0 = int(_clamp(arrow_top + int(fs * 0.45), 8, H - pillh - 8))
    d.rounded_rectangle([x0, y0, x0 + pillw, y0 + pillh], radius=int(pillh / 2), fill=V_INK + (236,))
    d.rounded_rectangle([x0, y0, x0 + pillw, y0 + pillh], radius=int(pillh / 2), outline=V_ACCENT + (90,), width=2)
    cy = y0 + pillh / 2
    d.ellipse([x0 + padx, cy - dot / 2, x0 + padx + dot, cy + dot / 2], fill=V_ACCENT + (255,))
    d.text((x0 + padx + dot + gap, y0 + pady - tb[1]), text, font=fnt, fill=(248, 249, 252, 255))
    frame.alpha_composite(_with_opacity(layer, op))


def _compose_frame(t, base, objs, by_obj, focus_seq, W, H):
    from PIL import ImageFilter
    active, presented, pres_cue = _active_state(t, by_obj)
    cur, _prev = _focus_at(t, focus_seq)

    # A hero (raise) holds centre-stage, but must YIELD as soon as a newer focus
    # begins — otherwise it lingers while the pointer has already moved on. When
    # superseded it cross-fades out over V_FADE instead of cutting.
    superseded = (presented is not None and cur is not None
                  and cur["object_id"] != presented and cur["t"] > pres_cue["t"] + 1e-6)
    pres_elapsed = (t - pres_cue["t"]) if pres_cue else 0.0
    hero = presented is not None and pres_elapsed >= V_BORDER_DUR and not superseded

    if hero:
        frame = _brightness(base.convert("RGBA").filter(ImageFilter.GaussianBlur(2)), V_DIM_HERO)
    elif active:
        frame = _brightness(base.convert("RGBA"), V_DIM_GENTLE)
    else:
        frame = base.convert("RGBA").copy()

    def fade(c):
        return _clamp((t - c["t"]) / V_FADE, 0.0, 1.0) if V_FADE > 0 else 1.0

    if hero:
        # Sole focus: the one card centre-stage.
        o = objs[presented]
        nw, nh, left, top = _hero_rect(o, pres_elapsed, t, W, H)
        _paste_card(frame, o["img"].resize((nw, nh)), left, top, fade(pres_cue),
                    shadow=0.62, shadow_blur=int(max(16, nh * 0.06)), shadow_dy=int(H * 0.05))
    else:
        # Gently-lifted cards for every active object (presented handled below).
        for oid, c in active.items():
            if oid == presented:
                continue
            o = objs[oid]
            nw, nh, left, top = _gentle_rect(o, t - c["t"], t, H)
            _paste_card(frame, o["img"].resize((nw, nh)), left, top, fade(c),
                        shadow=0.55, shadow_blur=int(max(10, nh * 0.06)), shadow_dy=int(max(8, nh * 0.06)))
        if presented is not None:
            o = objs[presented]
            if superseded:
                # Hero crossfades out from its lifted position as the next card arrives.
                op = 1.0 - _clamp((t - cur["t"]) / V_FADE, 0.0, 1.0)
                if op > 0.01:
                    nw, nh, left, top = _hero_rect(o, pres_elapsed, t, W, H)
                    _paste_card(frame, o["img"].resize((nw, nh)), left, top, op,
                                shadow=0.62, shadow_blur=int(max(16, nh * 0.06)), shadow_dy=int(H * 0.05))
            else:
                # Brief in-place beat before the lift to centre.
                nw, nh, left, top = _gentle_rect(o, pres_elapsed, t, H)
                _paste_card(frame, o["img"].resize((nw, nh)), left, top, fade(pres_cue),
                            shadow=0.5, shadow_blur=int(max(8, nh * 0.05)), shadow_dy=int(max(6, nh * 0.045)))

    # Floating pointer + brief annotation callout (audio-driven via focus_seq).
    anchor = _draw_pointer(frame, focus_seq, objs, by_obj, t, W, H)
    if cur and anchor:
        label = objs.get(cur["object_id"], {}).get("label", "")
        _draw_annotation(frame, label, anchor, t, cur["t"], W, H)

    return _ambient(frame, t, W, H).convert("RGB")


def _ambient(frame, t, W, H):
    """A very slow breath + drift on the whole composition so the frame is never
    fully static, without disturbing the object coordinates."""
    z = V_AMBIENT * (0.5 - 0.5 * math.cos(t * 2 * math.pi / 16.0))
    if z <= 0.0005:
        return frame
    nw, nh = int(round(W * (1 + z))), int(round(H * (1 + z)))
    big = frame.resize((nw, nh))
    px = math.sin(t * 2 * math.pi / 37.0) * 0.5 + 0.5
    py = math.cos(t * 2 * math.pi / 41.0) * 0.5 + 0.5
    ox, oy = int((nw - W) * px), int((nh - H) * py)
    return big.crop((ox, oy, ox + W, oy + H))



# Frame compositing is CPU-bound (Pillow); fan it out across all cores so a
# ~60s clip renders in seconds on the multi-core GPU box instead of minutes.
_MP = {}


def _mp_init(base, objs, by_obj, focus_seq, W, H, frames_dir):
    _MP.update(base=base, objs=objs, by_obj=by_obj, focus_seq=focus_seq, W=W, H=H, dir=frames_dir)


def _mp_frame(args):
    fi, t = args
    _compose_frame(t, _MP["base"], _MP["objs"], _MP["by_obj"], _MP["focus_seq"], _MP["W"], _MP["H"]).save(
        os.path.join(_MP["dir"], f"f{fi:05d}.png"))


def render_narration_video(scene, fps=24):
    ffmpeg = _ffmpeg_exe()
    work = tempfile.mkdtemp(prefix="narr_")
    try:
        base = _v_load_image(scene["image_url"]).convert("RGB")
        W, H = base.size
        W -= W % 2  # libx264 needs even dimensions
        H -= H % 2
        base = base.crop((0, 0, W, H))

        # Rounded-rect crops from the base image (general shapes that "lift"); the
        # jagged SAM alpha masks are set aside.
        objs = _build_objs(scene, base, W, H)

        tracks = scene.get("audio", {}).get("tracks", [])
        audio_files, durations = [], []
        for i, tr in enumerate(tracks):
            p = os.path.join(work, f"a{i}.mp3")
            _v_download(tr["url"], p)
            audio_files.append(p)
            durations.append(_media_duration(p, ffmpeg))

        offsets, acc = [], 0.0
        for d in durations:
            offsets.append(acc)
            acc += d
        total = acc if acc > 0 else 1.0

        by_obj, focus_seq = _build_cues(scene, offsets, total)

        nframes = max(1, int(math.ceil(total * fps)))
        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir)
        frame_args = [(fi, fi / fps) for fi in range(nframes)]
        nproc = min(max(1, (os.cpu_count() or 4)), 32)
        if nproc > 1 and nframes > 8:
            import multiprocessing as mp
            with mp.Pool(processes=nproc, initializer=_mp_init,
                         initargs=(base, objs, by_obj, focus_seq, W, H, frames_dir)) as pool:
                pool.map(_mp_frame, frame_args, chunksize=4)
        else:
            _mp_init(base, objs, by_obj, focus_seq, W, H, frames_dir)
            for a in frame_args:
                _mp_frame(a)

        # Concatenate the narration clips into one track.
        list_path = os.path.join(work, "audio.txt")
        with open(list_path, "w") as f:
            for p in audio_files:
                f.write(f"file '{p}'\n")
        audio_out = os.path.join(work, "audio.m4a")
        if audio_files:
            subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                            "-c:a", "aac", "-b:a", "160k", audio_out], check=True, capture_output=True)

        out = os.path.join(work, "out.mp4")
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "f%05d.png")]
        if audio_files:
            cmd += ["-i", audio_out]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"]
        if audio_files:
            cmd += ["-c:a", "aac", "-shortest"]
        cmd += ["-movflags", "+faststart", out]
        subprocess.run(cmd, check=True, capture_output=True)

        url = _upload_video(out, scene.get("lesson_item_id"))
        return {
            "video_url": url,
            "duration": round(total, 2),
            "frames": nframes,
            "fps": fps,
            "width": W,
            "height": H,
            "clips": len(audio_files),
            "objects": len(objs),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _upload_video(path, lesson_item_id=None):
    key_dir = f"videos/{lesson_item_id}" if lesson_item_id else "videos"
    s3_key = f"{key_dir}/{uuid.uuid4()}.mp4"
    s3 = _get_s3()
    extra_args = {"ContentType": "video/mp4"}
    if not S3_ENDPOINT or "r2.cloudflarestorage" not in S3_ENDPOINT:
        extra_args["ACL"] = "public-read"
    with open(path, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET_NAME, s3_key, ExtraArgs=extra_args)
    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL.rstrip('/')}/{s3_key}"
    return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"


# ── RunPod Serverless Handler ────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler.

    Modes:
      - segmentation (default): {"image_url": "...", "prompts": [...]}
      - video: {"mode": "video", "scene": {...}, "fps": 24}
    """
    inp = job["input"]

    # ── Video rendering mode (does not require the SAM3 model) ──
    if inp.get("mode") == "video" or inp.get("scene"):
        try:
            scene = inp.get("scene") or {}
            if not scene.get("image_url"):
                raise ValueError("video mode requires input.scene.image_url")
            fps = int(inp.get("fps", 24))
            print(f"[VIDEO] Rendering scene for lesson_item {scene.get('lesson_item_id')} @ {fps}fps", flush=True)
            result = render_narration_video(scene, fps=fps)
            print(f"[VIDEO] ✅ {result}", flush=True)
            return result
        except Exception as e:
            traceback.print_exc()
            return {"error": "video_render_failed", "message": str(e), "trace": traceback.format_exc()[-1500:]}

    # Segmentation path needs the SAM3 model — load it on first use.
    ensure_model()

    # Diagnostic mode: return system info if model failed to load
    if sam3_model is None or sam3_processor is None:
        return {
            "error": "Model not loaded",
            "model_error": model_error,
            "python_version": sys.version,
            "cuda_available": torch.cuda.is_available() if 'torch' in dir() else False,
            "hf_home": os.environ.get("HF_HOME", "NOT SET"),
            "cache_exists": os.path.exists(CACHE),
            "workspace_contents": os.listdir("/workspace") if os.path.exists("/workspace") else [],
        }

    # Test mode
    if inp.get("test"):
        return {"status": "ok", "model_loaded": True, "message": "SAM3 handler ready"}

    # Box-prompted mode: caller supplies a box per region (Grounded-SAM).
    if inp.get("mode") == "segment_boxes" or inp.get("regions"):
        image_url = inp.get("image_url")
        regions = inp.get("regions") or []
        if not image_url:
            raise ValueError("segment_boxes mode requires input.image_url")
        if not regions:
            raise ValueError("segment_boxes mode requires input.regions")

        print(f"[BOX] Processing {len(regions)} regions for {image_url}", flush=True)
        resp = http_requests.get(image_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size

        objects = segment_with_boxes(image, regions)
        unique_labels = list(dict.fromkeys(o["label"] for o in objects))
        caption = f"Image containing: {', '.join(unique_labels)}" if unique_labels else "Educational image"
        print(f"[BOX] ✅ Returning {len(objects)} masked objects", flush=True)
        return {"caption": caption, "image_width": w, "image_height": h, "objects": objects}

    image_url = inp.get("image_url")
    custom_prompts = inp.get("prompts")

    if not image_url:
        raise ValueError("Missing required input: image_url")

    print(f"[JOB] Processing: {image_url}", flush=True)

    resp = http_requests.get(image_url, timeout=30)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    w, h = image.size
    print(f"[JOB] Image size: {w}x{h}", flush=True)

    detected_objects = auto_segment_image(image, custom_prompts)
    print(f"[JOB] Detected {len(detected_objects)} objects", flush=True)

    objects = []
    for idx, obj in enumerate(detected_objects):
        masked_img = create_masked_crop(image, obj["mask"], obj["bbox"])
        if masked_img is None:
            continue

        mask_url = upload_mask_to_s3(masked_img)
        objects.append({
            "id": f"obj_{idx + 1}",
            "label": obj["label"],
            "bbox": obj["bbox"],
            "masked_image_url": mask_url,
            "score": round(obj["score"], 4),
        })

    unique_labels = list(dict.fromkeys(o["label"] for o in objects))
    caption = f"Image containing: {', '.join(unique_labels)}" if unique_labels else "Educational image"

    result = {
        "caption": caption,
        "image_width": w,
        "image_height": h,
        "objects": objects,
    }

    print(f"[JOB] ✅ Returning {len(objects)} objects", flush=True)
    return result


if __name__ == "__main__":
    import runpod
    print("[HANDLER] Starting RunPod serverless...", flush=True)
    runpod.serverless.start({"handler": handler})

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
V_DIM_SECTION = 0.4           # (unused) legacy blanket-spotlight dim
V_AMBIENT = 0.02              # (unused) legacy ambient breath
# Camera: the diagram sits zoomed-out with a background margin; the camera eases
# INTO a section only when it is a tight, contiguous block (then a dotted outline
# frames it). No whole-frame breathing — the frame stays stable.
V_MARGIN = 0.11             # background margin around the diagram (zoomed-out default)
V_SECTION_FILL = 0.8         # fraction of the frame a zoomed-in section fills
V_ZOOM_DUR = 0.9             # camera ease between framings (seconds)
V_TIGHT_MIN = 0.4            # min item-area / section-area to count as a tight block
V_SECTION_HOLD = 2.6        # how long the camera holds a section zoom before pulling back out
V_DASH_SPEED = 0.06         # marching-ants speed of the dotted outline (fraction of min side / s)
V_FLASH_DUR = 0.85          # flash pulse on each point-item as it is named (strong, visible)
V_SECTION_MAX_AREA = 0.4    # a section bigger than this fraction of the diagram gets NO zoom/outline (would blanket)
V_LABEL_IN, V_LABEL_HOLD, V_LABEL_OUT = 0.45, 2.4, 0.7   # brief chapter-label signpost envelope
# Spotlight LIFT: a border/raise item floats to the centre while the rest darkens,
# eased in AND out so it never pops in or vanishes.
V_LIFT_IN = 0.7             # grow-to-centre duration
V_LIFT_OUT = 0.5            # shrink-back duration before the item is done
V_LIFT_DIM = 0.82           # how much the background darkens under a lift (alpha of black)
V_LIFT_FILL = 0.52          # fraction of the frame the centred lifted card fills


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
def _v_detect_shapes(base):
    """Whole-element candidate boxes via classical CV, measured in the EXACT pixel
    space of the base image the renderer draws in — so snapping can NEVER introduce a
    coordinate mismatch (no padding/old-size/new-size ambiguity). Two passes (outlined
    boxes via Canny + filled shapes via non-white), unioned. Best-effort → [] on error.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    try:
        arr = np.array(base.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
        H, W = arr.shape[:2]
        area = float(W * H)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
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
                boxes.append((x, y, x + w, y + h))

        edges = cv2.dilate(cv2.Canny(gray, 30, 120), np.ones((3, 3), np.uint8), iterations=2)
        collect(edges)
        nonwhite = cv2.morphologyEx((gray < 238).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        collect(nonwhite)
        return boxes
    except Exception:
        return []


def _v_snap_objects(objects, shapes):
    """Snap each object's box to the real WHOLE-ELEMENT shape it sits in. A detector's
    box is often a fragment (just the word "Encoder"); snapping to the enclosing
    detected shape recovers the whole block + label. Picks the smallest shape that
    contains the box centre and is >=55% of the rough box area (so it never collapses
    onto a tiny token inside), else the smallest containing shape. Leaves the box
    unchanged when nothing contains it. Mutates `objects` in place."""
    if not shapes:
        return
    for o in objects:
        b = o.get("bbox")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rough = max(1.0, (x2 - x1) * (y2 - y1))
        cont = [s for s in shapes if s[0] - 2 <= cx <= s[2] + 2 and s[1] - 2 <= cy <= s[3] + 2]
        if not cont:
            continue
        big = [s for s in cont if (s[2] - s[0]) * (s[3] - s[1]) >= 0.55 * rough]
        pool = big or cont
        s = min(pool, key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
        o["bbox"] = [float(s[0]), float(s[1]), float(s[2]), float(s[3])]


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


def _build_sections(scene, W, H):
    """Pedagogical sections {id,label,role,item_ids,bbox(px, clamped)} -> the
    renderer spotlights the active one and shows its label as a signpost."""
    out = {}
    for s in scene.get("sections", []):
        bb = s.get("bbox") or []
        if len(bb) != 4:
            continue
        x1 = int(_clamp(round(float(bb[0])), 0, W - 2))
        y1 = int(_clamp(round(float(bb[1])), 0, H - 2))
        x2 = int(_clamp(round(float(bb[2])), x1 + 1, W))
        y2 = int(_clamp(round(float(bb[3])), y1 + 1, H))
        out[s["id"]] = {
            "id": s["id"], "label": str(s.get("label") or ""), "role": str(s.get("role") or "other"),
            "item_ids": list(s.get("item_ids") or []),
            "left": x1, "top": y1, "w": x2 - x1, "h": y2 - y1,
        }
    return out


def _section_of_item(oid, sections):
    for sid, s in sections.items():
        if oid in s["item_ids"]:
            return sid
    return None


def _build_section_timeline(focus_seq, sections):
    """Collapse the focus order into contiguous per-section RUNS, each with the
    time its first item is named. The active section persists for the whole run."""
    timeline = []
    for c in focus_seq:
        sid = _section_of_item(c["object_id"], sections)
        if sid is None:
            continue
        if not timeline or timeline[-1]["section_id"] != sid:
            timeline.append({"section_id": sid, "t": c["t"]})
    return timeline


def _active_section(t, timeline):
    cur = None
    for run in timeline:
        if run["t"] <= t + 1e-6:
            cur = run
        else:
            break
    return cur


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


# ── Camera + card geometry ───────────────────────────────────────────────────
# The diagram is composed onto a "stage" at its own coordinates, then a viewport
# CAMERA maps the stage to the output frame: zoomed OUT with a background margin by
# default (so the whole diagram is visible with breathing room), and eased INTO the
# current pedagogical section when that section is a tight, contiguous block. The
# arrow + pills + dotted section outline are drawn crisply in OUTPUT space.

def _lift_rect(o, elapsed, t, H, strong=False):
    """An in-place lift of a card off the page (no fly-to-centre)."""
    amt = 0.2 if strong else V_GENTLE_SCALE
    p = _ease_out(_clamp(elapsed / V_SETTLE, 0.0, 1.0))
    scale = 1.0 + amt * p
    rise = (V_GENTLE_RISE * (1.7 if strong else 1.0) * H) * p
    bob = math.sin(t * 1.5) * (H * 0.0035) * p
    nw = max(1, int(round(o["w"] * scale)))
    nh = max(1, int(round(o["h"] * scale)))
    cx = o["left"] + o["w"] / 2
    cy = o["top"] + o["h"] / 2 - rise + bob
    return nw, nh, int(round(cx - nw / 2)), int(round(cy - nh / 2))


def _bg_color(base):
    """The diagram's background colour, sampled from its border (for the margin)."""
    im = base.convert("RGB")
    w, h = im.size
    small = im.resize((min(w, 64), min(h, 64)))
    sw, sh = small.size
    px = small.load()
    border = []
    for x in range(sw):
        border.append(px[x, 0]); border.append(px[x, sh - 1])
    for y in range(sh):
        border.append(px[0, y]); border.append(px[sw - 1, y])
    border.sort(key=lambda c: c[0] + c[1] + c[2])
    return border[len(border) // 2]


def _default_viewport(W, H):
    # Stable, full-frame view: NO zoomed-out margin. The margin rendered white padding
    # around the diagram, which (a) read as confusing empty space and (b) made the
    # camera scale the diagram, so lift cards (placed through the camera transform)
    # appeared to drift. The diagram now fills the frame 1:1; lifts/outline/arrows are
    # crisp output-space overlays on top.
    return (0.0, 0.0, float(W), float(H))


def _fit_viewport(rect, W, H, fill):
    rx, ry, rw, rh = rect
    ar = W / H
    vw = max(rw / fill, (rh / fill) * ar)
    vh = vw / ar
    cx, cy = rx + rw / 2.0, ry + rh / 2.0
    return (cx - vw / 2.0, cy - vh / 2.0, vw, vh)


def _section_tight(sec, objs, W, H):
    """A section is zoomed + outlined only when it is a compact, contiguous block:
    its items fill a decent fraction of its bbox AND its bbox is not so large that
    the outline would blanket the diagram (a big/scattered section gets no frame —
    just the per-item arrow)."""
    area = sec["w"] * sec["h"]
    if area <= 1:
        return False
    if area > V_SECTION_MAX_AREA * (W * H):
        return False
    used = 0.0
    for oid in sec["item_ids"]:
        o = objs.get(oid)
        if o:
            used += o["w"] * o["h"]
    return (used / area) >= V_TIGHT_MIN


def _build_cam_runs(section_timeline, sections, objs, W, H):
    """No camera punch. Section emphasis is the marching dotted OUTLINE only (an
    output-space overlay). The previous per-section zoom-IN-then-pull-OUT collided
    with a lift's exit: the section pull-out (the whole region shrinking back) played
    at the same time as the lifted card shrinking back, so two things appeared to
    "animate out" at once. A single stable framing removes that entirely."""
    return [{"t": 0.0, "rect": None}]


def _camera(t, cam_runs, W, H):
    cur, prev = None, None
    for r in cam_runs:
        if r["t"] <= t + 1e-6:
            prev, cur = cur, r
        else:
            break

    def vp_of(run):
        if run is None or run["rect"] is None:
            return _default_viewport(W, H)
        return _fit_viewport(run["rect"], W, H, V_SECTION_FILL)

    cvp = vp_of(cur)
    if prev is None:
        return cvp
    pvp = vp_of(prev)
    p = _ease_in_out(_clamp((t - cur["t"]) / V_ZOOM_DUR, 0.0, 1.0))
    return tuple(pvp[i] + (cvp[i] - pvp[i]) * p for i in range(4))


def _apply_viewport(stage, vp, bg, W, H):
    vx, vy, vw, vh = vp
    iw, ih = max(1, int(round(vw))), max(1, int(round(vh)))
    canvas = Image.new("RGBA", (iw, ih), tuple(bg) + (255,))
    canvas.alpha_composite(stage, (int(round(-vx)), int(round(-vy))))
    return canvas.resize((W, H))


def _vp_map(sx, sy, vp, W, H):
    vx, vy, vw, vh = vp
    return ((sx - vx) / vw * W, (sy - vy) / vh * H)


def _draw_dotted_rect(d, x1, y1, x2, y2, color, width, dash, gap, phase=0.0):
    """A dashed rectangle whose dashes MARCH (offset by `phase`) so it animates."""
    period = dash + gap
    ph = phase % period

    def dline(xa, ya, xb, yb):
        L = math.hypot(xb - xa, yb - ya)
        if L < 1:
            return
        ux, uy = (xb - xa) / L, (yb - ya) / L
        pos = -ph
        while pos < L:
            a, bb = max(0.0, pos), min(L, pos + dash)
            if bb > a:
                d.line([(xa + ux * a, ya + uy * a), (xa + ux * bb, ya + uy * bb)], fill=color, width=width)
            pos += period
    dline(x1, y1, x2, y1)
    dline(x2, y1, x2, y2)
    dline(x2, y2, x1, y2)
    dline(x1, y2, x1, y1)


# ── Floating 3D pointer ──────────────────────────────────────────────────────
_ARROW_CACHE = {}


def _arrow_img(size, up=False):
    """A glossy, bevelled arrow pointing DOWN (tip at bottom centre), or UP when
    `up` is set (so an item near the top edge is pointed at from below), cached."""
    size = int(size)
    key = (size, bool(up))
    if key in _ARROW_CACHE:
        return _ARROW_CACHE[key]
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
    tip = (canvas[0] / 2, oy + Hd)
    if up:
        out = out.rotate(180)
        tip = (canvas[0] - tip[0], canvas[1] - tip[1])
    meta = {"tip": tip, "size": canvas}
    _ARROW_CACHE[key] = (out, meta)
    return _ARROW_CACHE[key]


def _focus_at(t, focus_seq):
    cur, prev = None, None
    for c in focus_seq:
        if c["t"] <= t + 1e-6:
            prev, cur = cur, c
        else:
            break
    return cur, prev


def _target_top_center(oid, objs, by_obj, t, H):
    """Where the pointer aims, in STAGE coords: the top-centre of the item (its
    lifted card top for border/raise, else its plain bbox top)."""
    o = objs.get(oid)
    if not o:
        return None
    cue = None
    for c in by_obj.get(oid, []):
        if c["t"] <= t + 1e-6:
            cue = c
        else:
            break
    if cue and cue["type"] in ("border", "raise"):
        e = t - cue["t"]
        nw, nh, left, top = _lift_rect(o, e, t, H, strong=(cue["type"] == "raise"))
        return (left + nw / 2, top)
    return (o["left"] + o["w"] / 2, o["top"])


def _item_rect_out(oid, objs, vp, W, H):
    """An object's bbox mapped into OUTPUT (post-camera) coordinates: [x1,y1,x2,y2]."""
    o = objs[oid]
    x1, y1 = _vp_map(o["left"], o["top"], vp, W, H)
    x2, y2 = _vp_map(o["left"] + o["w"], o["top"] + o["h"], vp, W, H)
    return [x1, y1, x2, y2]


def _draw_pointer(out, focus_seq, objs, by_obj, t, vp, W, H):
    """Lands the arrow ON the current item (glides from the previous one). Points
    DOWN from just above the item, or UP from just below it when the item hugs the
    top edge — so the arrow never detaches to the frame edge."""
    cur, prev = _focus_at(t, focus_seq)
    if not cur or cur["object_id"] not in objs:
        return None

    cur_rect = _item_rect_out(cur["object_id"], objs, vp, W, H)
    if prev and prev["object_id"] in objs:
        prev_rect = _item_rect_out(prev["object_id"], objs, vp, W, H)
        gp = _ease_in_out(_clamp((t - cur["t"]) / V_GLIDE, 0.0, 1.0))
    else:
        prev_rect, gp = cur_rect, 1.0
    rect = [prev_rect[i] + (cur_rect[i] - prev_rect[i]) * gp for i in range(4)]
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2.0

    size = int(_clamp(min(W, H) * 0.055, 32, 104))
    point_up = y1 < H * 0.2   # item hugs the top → point UP from below it
    arrow, meta = _arrow_img(size, up=point_up)
    aw, ah = meta["size"]
    tipx_off, tipy_off = meta["tip"]
    bob = math.sin(t * 2.3) * (ah * 0.05)
    if point_up:
        tip_x, tip_y = cx, y2 + ah * 0.12 + bob
    else:
        tip_x, tip_y = cx, y1 - ah * 0.12 - bob

    left = int(round(_clamp(tip_x - tipx_off, 4 - aw, W - 4)))
    top = int(round(tip_y - tipy_off))
    op = _clamp((t - cur["t"]) / 0.3, 0.0, 1.0) if not prev else 1.0
    out.alpha_composite(_with_opacity(arrow, op), (left, top))
    # Annotation anchor: just outside the arrow, on the side away from the item.
    return (tip_x, top if not point_up else top + ah, point_up)


def _draw_flash(out, o, in_t, t, vp, W, H):
    """A strong, clearly-visible pulse on a point-item as it is named: a glowing
    accent ring around it (far more visible than a faint brightness bump)."""
    if not o:
        return
    fe = t - in_t
    if not (0.0 <= fe < V_FLASH_DUR):
        return
    from PIL import ImageDraw, ImageFilter
    pulse = math.sin(fe / V_FLASH_DUR * math.pi)
    x1, y1 = _vp_map(o["left"], o["top"], vp, W, H)
    x2, y2 = _vp_map(o["left"] + o["w"], o["top"] + o["h"], vp, W, H)
    pad = max(4.0, min(W, H) * 0.012)
    x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
    rad = int(_clamp(min(x2 - x1, y2 - y1) * 0.12, 6, 44))
    wdt = max(4, int(min(W, H) * 0.011))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle([x1, y1, x2, y2], radius=rad, outline=V_ACCENT + (255,), width=wdt)
    glow = glow.filter(ImageFilter.GaussianBlur(int(min(W, H) * 0.014)))
    out.alpha_composite(_with_opacity(glow, pulse))
    ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(ring).rounded_rectangle([x1, y1, x2, y2], radius=rad, outline=V_ACCENT_HI + (255,), width=max(3, wdt // 2))
    out.alpha_composite(_with_opacity(ring, pulse * 0.95))


def _draw_lift(out, o, obj_cues, cur, t, vp, W, H, hold_until=None):
    """Spotlight LIFT: darken the whole frame and float the item to the centre,
    enlarged, eased IN and OUT so it grows in and shrinks back (never pops).

    The lift stays up for the WHOLE time the narration is discussing the item — i.e.
    until the NEXT item is named (`hold_until`) — not merely until its short name
    finishes. The model tends to close an item's `out` cue right after the name, so
    relying on that alone made the card drop back down while still being talked about.
    Falls back to the item's own `out` cue for the final item (nothing follows it)."""
    in_t = cur["t"]
    own_out = None
    for c in obj_cues:
        if c.get("action") == "out" and c["t"] > in_t + 1e-6:
            own_out = c["t"]
            break
    end_t = hold_until if hold_until is not None else own_out
    e = t - in_t
    p_in = _ease_out(_clamp(e / V_LIFT_IN, 0.0, 1.0))
    p_out = _ease_out(_clamp((end_t - t) / V_LIFT_OUT, 0.0, 1.0)) if end_t is not None else 1.0
    p = max(0.0, min(p_in, p_out))
    if p <= 0.01:
        return None

    dim = Image.new("RGBA", (W, H), (0, 0, 0, int(255 * V_LIFT_DIM)))
    out.alpha_composite(_with_opacity(dim, p))

    x1, y1 = _vp_map(o["left"], o["top"], vp, W, H)
    x2, y2 = _vp_map(o["left"] + o["w"], o["top"] + o["h"], vp, W, H)
    sw, sh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    scx, scy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = _clamp(min(V_LIFT_FILL * W / sw, V_LIFT_FILL * H / sh), 1.0, 5.0)
    nw = max(1, int(round(sw * (1.0 + (s - 1.0) * p))))
    nh = max(1, int(round(sh * (1.0 + (s - 1.0) * p))))
    cx = scx + (W / 2.0 - scx) * p
    cy = scy + (H * 0.5 - scy) * p
    card = o["img"].resize((nw, nh), Image.LANCZOS)
    left, top = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    _paste_card(out, card, left, top, 1.0,
                shadow=0.6 * p, shadow_blur=int(max(14, nh * 0.06)), shadow_dy=int(max(10, nh * 0.05)))
    return (cx, top, False)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _draw_annotation(frame, text, anchor, t, t0, W, H, avoid=None):
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
    anchor_x, anchor_y = anchor[0], anchor[1]
    below = bool(anchor[2]) if len(anchor) > 2 else False
    x0 = int(_clamp(anchor_x - pillw / 2, 8, W - pillw - 8))
    above_y = int(anchor_y - pillh - int(fs * 0.45))
    below_y = int(_clamp(anchor_y + int(fs * 1.4), 8, H - pillh - 8))
    y0 = below_y if below else above_y
    if y0 < 8:
        y0 = below_y
    # Never collide with the fixed section-label pill — drop below the arrow instead.
    if avoid is not None and _rects_overlap((x0, y0, x0 + pillw, y0 + pillh), avoid):
        y0 = below_y
    d.rounded_rectangle([x0, y0, x0 + pillw, y0 + pillh], radius=int(pillh / 2), fill=V_INK + (236,))
    d.rounded_rectangle([x0, y0, x0 + pillw, y0 + pillh], radius=int(pillh / 2), outline=V_ACCENT + (90,), width=2)
    cy = y0 + pillh / 2
    d.ellipse([x0 + padx, cy - dot / 2, x0 + padx + dot, cy + dot / 2], fill=V_ACCENT + (255,))
    d.text((x0 + padx + dot + gap, y0 + pady - tb[1]), text, font=fnt, fill=(248, 249, 252, 255))
    frame.alpha_composite(_with_opacity(layer, op))


def _draw_section_label(frame, text, op, W, H):
    """A fixed 'chapter' pill at the top-left (a stable signpost that never moves
    around or collides with the per-item annotation). Returns its bounding rect."""
    text = (text or "").strip()
    if not text:
        return None
    from PIL import ImageDraw
    fs = int(_clamp(min(W, H) * 0.03, 16, 38))
    fnt = _font(fs, bold=True)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tb = d.textbbox((0, 0), text, font=fnt)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    dot = int(fs * 0.4)
    padx, pady = int(fs * 0.6), int(fs * 0.42)
    gap = int(fs * 0.4)
    pw, ph = padx + dot + gap + tw + padx, th + pady * 2
    m = int(min(W, H) * 0.03)
    px, py = m, m
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=int(ph / 2), fill=V_INK + (238,))
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=int(ph / 2), outline=V_ACCENT + (170,), width=2)
    cy = py + ph / 2
    d.ellipse([px + padx, cy - dot / 2, px + padx + dot, cy + dot / 2], fill=V_ACCENT + (255,))
    d.text((px + padx + dot + gap, py + pady - tb[1]), text, font=fnt, fill=(248, 249, 252, 255))
    frame.alpha_composite(_with_opacity(layer, op))
    return (px, py, px + pw, py + ph)


def _hover_viewport(t, phase_t, W, H):
    """The default (full-diagram) viewport, but gently RAISED toward the viewer (a
    slight eased zoom-in) and HOVERING (a soft vertical bob) — used for the intro and
    outro so the whole diagram floats while it is introduced / wrapped up."""
    vx, vy, vw, vh = _default_viewport(W, H)
    e = max(0.0, t - phase_t)
    rise = _ease_out(_clamp(e / 0.8, 0.0, 1.0))
    zoom = 1.0 - 0.035 * rise                      # up to ~3.5% larger = "raised"
    bob = math.sin(t * 1.3) * (H * 0.012) * rise   # gentle hover
    cx, cy = vx + vw / 2.0, vy + vh / 2.0
    nw, nh = vw * zoom, vh * zoom
    return (cx - nw / 2.0, cy - nh / 2.0 + bob, nw, nh)


def _draw_whole_outline(stage, whole_bbox, t, intro_end, outro_start, in_intro, bg, W, H):
    """INTRO / OUTRO treatment: present the ENTIRE diagram in full view, gently raised
    and hovering, with a marching dotted border tracing all the way around it. Used
    before the first item is named (intro) and during the closing takeaway (outro)."""
    from PIL import ImageDraw
    phase_t = 0.0 if in_intro else outro_start
    vp = _hover_viewport(t, phase_t, W, H)
    out = _apply_viewport(stage, vp, bg, W, H)

    if in_intro:
        if t < 0.6:
            op = _ease_out(_clamp(t / 0.6, 0.0, 1.0))
        elif t < intro_end - 0.4:
            op = 1.0
        else:
            op = _clamp((intro_end - t) / 0.4, 0.0, 1.0)
    else:
        op = _ease_out(_clamp((t - outro_start) / 0.5, 0.0, 1.0))
    if op <= 0.01:
        return out

    bx1, by1 = _vp_map(whole_bbox[0], whole_bbox[1], vp, W, H)
    bx2, by2 = _vp_map(whole_bbox[2], whole_bbox[3], vp, W, H)
    pad = min(W, H) * 0.022
    bx1 = max(2.0, bx1 - pad); by1 = max(2.0, by1 - pad)
    bx2 = min(W - 2.0, bx2 + pad); by2 = min(H - 2.0, by2 + pad)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    phase = t * V_DASH_SPEED * min(W, H)
    _draw_dotted_rect(ImageDraw.Draw(layer), bx1, by1, bx2, by2, V_ACCENT + (255,),
                      max(4, int(min(W, H) * 0.0075)), int(min(W, H) * 0.028),
                      int(min(W, H) * 0.02), phase)
    out.alpha_composite(_with_opacity(layer, op * 0.95))
    return out


def _compose_frame(t, base, objs, by_obj, focus_seq, sections, section_timeline, cam_runs, bg, W, H,
                   intro_end=0.0, outro_start=None, whole_bbox=None):
    cur, _prev = _focus_at(t, focus_seq)

    # INTRO (before the first item is named) and OUTRO (during the closing takeaway,
    # after the last item's `out`) get the whole-diagram float + marching border so
    # the opening and conclusion are never a static frame.
    stage_rgba = base.convert("RGBA")
    in_intro = bool(focus_seq) and t < intro_end - 0.05
    in_outro = outro_start is not None and t > outro_start + 0.05
    if (in_intro or in_outro) and whole_bbox is not None:
        return _draw_whole_outline(stage_rgba, whole_bbox, t, intro_end, outro_start,
                                   in_intro, bg, W, H).convert("RGB")

    # STAGE = the diagram only. Lifts/flashes/arrows are OUTPUT-space overlays drawn
    # AFTER the camera, so they stay crisp and correctly placed at any zoom.
    stage = stage_rgba

    # CAMERA: zoomed out with margin by default; eased into the current tight
    # section briefly, then back out so the whole diagram is visible again.
    vp = _camera(t, cam_runs, W, H)
    out = _apply_viewport(stage, vp, bg, W, H)

    # Animated dotted outline + fixed chapter label for the current section (only a
    # tight, contiguous block — never a blanket over scattered content). The dashes
    # MARCH so the outline reads as a moving, attention-drawing frame.
    sec_run = _active_section(t, section_timeline) if sections else None
    sec = sections.get(sec_run["section_id"]) if sec_run else None
    label_rect = None
    if sec and _section_tight(sec, objs, W, H):
        from PIL import ImageDraw
        elapsed = t - sec_run["t"]
        op = _ease_out(_clamp(elapsed / 0.45, 0.0, 1.0))   # outline stays while the section is active
        x1, y1 = _vp_map(sec["left"], sec["top"], vp, W, H)
        x2, y2 = _vp_map(sec["left"] + sec["w"], sec["top"] + sec["h"], vp, W, H)
        pad = min(W, H) * 0.016
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        phase = t * V_DASH_SPEED * min(W, H)
        _draw_dotted_rect(ImageDraw.Draw(layer), x1, y1, x2, y2, V_ACCENT + (255,),
                          max(3, int(min(W, H) * 0.006)), int(min(W, H) * 0.026),
                          int(min(W, H) * 0.018), phase)
        out.alpha_composite(_with_opacity(layer, op * 0.95))
        # The chapter label is a BRIEF signpost (fades out) so it never lingers and
        # stacks under every per-item annotation.
        if elapsed < V_LABEL_IN + V_LABEL_HOLD:
            lab_op = elapsed / V_LABEL_IN if elapsed < V_LABEL_IN else 1.0
        elif elapsed < V_LABEL_IN + V_LABEL_HOLD + V_LABEL_OUT:
            lab_op = 1.0 - (elapsed - V_LABEL_IN - V_LABEL_HOLD) / V_LABEL_OUT
        else:
            lab_op = 0.0
        if lab_op > 0.01:
            label_rect = _draw_section_label(out, sec["label"], lab_op, W, H)

    # The current item: a border/raise LIFTS (darken the rest + float to centre);
    # a point flashes + gets the gliding arrow. Drawn over the section outline.
    lift = cur is not None and cur.get("type") in ("border", "raise") and cur["object_id"] in objs
    if lift:
        next_focus_t = None
        for c in focus_seq:
            if c["t"] > cur["t"] + 1e-6:
                next_focus_t = c["t"]
                break
        anchor = _draw_lift(out, objs[cur["object_id"]], by_obj.get(cur["object_id"], []), cur, t, vp, W, H,
                            hold_until=next_focus_t)
        if anchor:
            _draw_annotation(out, objs[cur["object_id"]].get("label", ""), anchor, t, cur["t"], W, H)
    else:
        if cur and cur["object_id"] in objs:
            _draw_flash(out, objs[cur["object_id"]], cur["t"], t, vp, W, H)
        anchor = _draw_pointer(out, focus_seq, objs, by_obj, t, vp, W, H)
        if cur and anchor:
            _draw_annotation(out, objs.get(cur["object_id"], {}).get("label", ""),
                             anchor, t, cur["t"], W, H, avoid=label_rect)

    return out.convert("RGB")




# Frame compositing is CPU-bound (Pillow); fan it out across all cores so a
# ~60s clip renders in seconds on the multi-core GPU box instead of minutes.
_MP = {}


def _mp_init(base, objs, by_obj, focus_seq, sections, section_timeline, cam_runs, bg, W, H, frames_dir,
             intro_end, outro_start, whole_bbox):
    _MP.update(base=base, objs=objs, by_obj=by_obj, focus_seq=focus_seq,
               sections=sections, section_timeline=section_timeline,
               cam_runs=cam_runs, bg=bg, W=W, H=H, dir=frames_dir,
               intro_end=intro_end, outro_start=outro_start, whole_bbox=whole_bbox)


def _mp_frame(args):
    fi, t = args
    _compose_frame(t, _MP["base"], _MP["objs"], _MP["by_obj"], _MP["focus_seq"],
                   _MP["sections"], _MP["section_timeline"], _MP["cam_runs"], _MP["bg"],
                   _MP["W"], _MP["H"], _MP["intro_end"], _MP["outro_start"], _MP["whole_bbox"]).save(
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

        # SNAP each object's box to its real whole-element shape, measured on THIS base
        # image (exact render space) — so a lift shows the whole block + label, never a
        # text fragment, and there is no coordinate-space mismatch. On by default.
        if scene.get("snap_to_shapes", True):
            _v_snap_objects(scene.get("objects", []), _v_detect_shapes(base))

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
        sections = _build_sections(scene, W, H)
        section_timeline = _build_section_timeline(focus_seq, sections)
        cam_runs = _build_cam_runs(section_timeline, sections, objs, W, H)
        bg = _bg_color(base)

        # Intro = before the first item is named; outro = after the last item's `out`
        # (the closing takeaway). Both get the whole-diagram float + marching border.
        intro_end = focus_seq[0]["t"] if focus_seq else 0.0
        out_times = [c["t"] for seq in by_obj.values() for c in seq
                     if c.get("action") == "out" and c.get("t") is not None]
        outro_start = max(out_times) if out_times else total
        if objs:
            whole_bbox = (
                min(o["left"] for o in objs.values()),
                min(o["top"] for o in objs.values()),
                max(o["left"] + o["w"] for o in objs.values()),
                max(o["top"] + o["h"] for o in objs.values()),
            )
        else:
            whole_bbox = (0, 0, W, H)

        nframes = max(1, int(math.ceil(total * fps)))
        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir)
        frame_args = [(fi, fi / fps) for fi in range(nframes)]
        nproc = min(max(1, (os.cpu_count() or 4)), 32)
        if nproc > 1 and nframes > 8:
            import multiprocessing as mp
            with mp.Pool(processes=nproc, initializer=_mp_init,
                         initargs=(base, objs, by_obj, focus_seq, sections, section_timeline, cam_runs, bg, W, H, frames_dir,
                                   intro_end, outro_start, whole_bbox)) as pool:
                pool.map(_mp_frame, frame_args, chunksize=4)
        else:
            _mp_init(base, objs, by_obj, focus_seq, sections, section_timeline, cam_runs, bg, W, H, frames_dir,
                     intro_end, outro_start, whole_bbox)
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

    # ── Shape-detection mode (classical CV; no SAM3/model load, returns fast) ──
    # Returns RAW candidate element boxes. The caller (PHP) cleans/dedups them, renders
    # a numbered overlay, and has the VLM LABEL the boxes — the VLM never invents
    # coordinates, which is what was producing tiled/duplicate/oversized boxes.
    if inp.get("mode") == "detect_shapes":
        image_url = inp.get("image_url")
        if not image_url:
            raise ValueError("detect_shapes mode requires input.image_url")
        resp = http_requests.get(image_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size
        boxes = _v_detect_shapes(image)
        print(f"[SHAPES] {len(boxes)} candidate boxes for {w}x{h}", flush=True)
        return {"shape_boxes": [list(b) for b in boxes], "image_width": w, "image_height": h}

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

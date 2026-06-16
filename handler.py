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

# Animation constants — mirror the browser renderer (image-narration-test.blade.php).
V_BORDER_DUR = 0.6
V_LIFT_DUR = 0.7
V_SHIMMER_START = V_BORDER_DUR + V_LIFT_DUR
V_SHIMMER_DUR = 1.2
V_SHIMMER_END = V_SHIMMER_START + V_SHIMMER_DUR
V_FADE = 0.4
V_RAISE_FILL = 0.68
V_RAISE_MIN = 0.45
V_RAISE_MAX = 3.0
V_ACCENT = (255, 107, 53)


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


def _outline_layer(mask_img, color, px=3):
    """Build a colored outline that follows the mask's alpha shape."""
    from PIL import ImageChops, ImageFilter
    a = mask_img.split()[3]
    dil = a.filter(ImageFilter.MaxFilter(px * 2 + 1))
    edge = ImageChops.subtract(dil, a)
    layer = Image.new("RGBA", mask_img.size, color + (255,))
    layer.putalpha(edge)
    return layer


def _glow_layer(mask_img, color, blur=10, strength=1.0):
    from PIL import ImageFilter
    a = mask_img.split()[3].filter(ImageFilter.GaussianBlur(blur))
    a = a.point(lambda v: int(min(255, v * strength)))
    layer = Image.new("RGBA", mask_img.size, color + (255,))
    layer.putalpha(a)
    return layer


def _shadow_for(mask_img, blur=18):
    """Soft black silhouette used as a drop/contact shadow."""
    from PIL import ImageFilter
    a = mask_img.split()[3].filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", mask_img.size, (0, 0, 0, 255))
    layer.putalpha(a)
    return layer


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


def _compose_frame(t, base, objs, by_obj, W, H):
    from PIL import ImageDraw
    active, presented, pres_cue = _active_state(t, by_obj)
    pres_elapsed = (t - pres_cue["t"]) if pres_cue else 0.0
    presenting = presented is not None and pres_elapsed >= V_BORDER_DUR

    frame = base
    if presenting:
        from PIL import ImageFilter
        frame = _brightness(base.convert("RGBA").filter(ImageFilter.GaussianBlur(2)), 0.18)
    else:
        frame = base.convert("RGBA").copy()

    def fade(c):
        return _clamp((t - c["t"]) / V_FADE, 0.0, 1.0) if V_FADE > 0 else 1.0

    def paste_in_place(oid, op, layer_extra=None):
        o = objs[oid]
        img = _with_opacity(o["img"], op)
        if layer_extra is not None:
            le = _with_opacity(layer_extra, op)
            frame.alpha_composite(le, (o["left"], o["top"]))
        frame.alpha_composite(img, (o["left"], o["top"]))

    # --- Non-presented objects: in-place effects ---
    for oid, c in active.items():
        if oid == presented:
            continue
        if presenting:
            paste_in_place(oid, 0.12)  # receded behind the lifted object
            continue
        op = fade(c)
        typ = c["type"]
        o = objs[oid]
        if typ == "border":
            paste_in_place(oid, op, _outline_layer(o["img"], V_ACCENT, 3))
        elif typ == "flash":
            pulse = 1.0 + 0.4 * abs(math.sin(t * 6.0))
            paste_in_place_img(frame, _brightness(o["img"], pulse), o, op)
        elif typ == "arrow":
            paste_in_place(oid, op, _glow_layer(o["img"], V_ACCENT, 8, 0.8))
            _draw_arrow(frame, o, op)
        else:
            paste_in_place(oid, op, _glow_layer(o["img"], (255, 255, 255), 8, 0.4))

    # --- Presented object: border-in-place -> lift -> shimmer/hold ---
    if presented is not None:
        o = objs[presented]
        op = fade(pres_cue)
        if not presenting:
            # Border draws around the object where it sits.
            paste_in_place(presented, op, _outline_layer(o["img"], V_ACCENT, 3))
        else:
            _paste_lifted(frame, o, pres_elapsed, t, W, H, op)

    return frame.convert("RGB")


def paste_in_place_img(frame, img, o, op):
    frame.alpha_composite(_with_opacity(img, op), (o["left"], o["top"]))


def _draw_arrow(frame, o, op):
    from PIL import ImageDraw
    cx = o["left"] + o["w"] / 2
    top = o["top"]
    size = max(14, int(o["w"] * 0.12))
    y = max(2, top - size - 6)
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.polygon([(cx - size / 2, y), (cx + size / 2, y), (cx, y + size)],
              fill=V_ACCENT + (int(255 * op),))
    frame.alpha_composite(layer)


def _paste_lifted(frame, o, elapsed, t, W, H, op):
    # Lift progress with overshoot from in-place (scale 1 @ bbox centre) to target.
    gw, gh = o["w"] / W, o["h"] / H
    target_scale = _clamp(min(V_RAISE_FILL / gw, V_RAISE_FILL / gh), V_RAISE_MIN, V_RAISE_MAX)
    lp = _ease_out_back((elapsed - V_BORDER_DUR) / V_LIFT_DUR)
    scale = 1.0 + (target_scale - 1.0) * lp

    start_cx, start_cy = o["left"] + o["w"] / 2, o["top"] + o["h"] / 2
    # Gentle hover once settled.
    settled = max(0.0, elapsed - V_SHIMMER_START)
    bob = math.sin(settled * 1.7) * (H * 0.012)
    end_cx, end_cy = W / 2, H / 2 + bob
    cx = start_cx + (end_cx - start_cx) * lp
    cy = start_cy + (end_cy - start_cy) * lp

    new_w = max(1, int(round(o["w"] * scale)))
    new_h = max(1, int(round(o["h"] * scale)))
    mask = o["img"].resize((new_w, new_h))

    # Shimmer burst (≈3 pulses) right after enlarging.
    if V_SHIMMER_START <= elapsed < V_SHIMMER_END:
        pulse = 1.0 + 0.5 * abs(math.sin((elapsed - V_SHIMMER_START) * math.pi / 0.4))
        mask = _brightness(mask, pulse)

    left = int(round(cx - new_w / 2))
    top = int(round(cy - new_h / 2))

    # Separated drop shadow (sells the height off the page).
    shadow = _shadow_for(mask, blur=22)
    shadow = _with_opacity(shadow, 0.55 * op)
    frame.alpha_composite(shadow, (left, top + int(H * 0.06)))

    # Slow border shimmer carried with the lift.
    glow = max(0.4, 0.7 + 0.3 * math.sin(t * 2.8))
    frame.alpha_composite(_with_opacity(_glow_layer(mask, V_ACCENT, 12, glow), op), (left, top))
    frame.alpha_composite(_with_opacity(_outline_layer(mask, V_ACCENT, 3), op), (left, top))
    frame.alpha_composite(_with_opacity(mask, op), (left, top))


# Frame compositing is CPU-bound (Pillow); fan it out across all cores so a
# ~60s clip renders in seconds on the multi-core GPU box instead of minutes.
_MP = {}


def _mp_init(base, objs, by_obj, W, H, frames_dir):
    _MP.update(base=base, objs=objs, by_obj=by_obj, W=W, H=H, dir=frames_dir)


def _mp_frame(args):
    fi, t = args
    _compose_frame(t, _MP["base"], _MP["objs"], _MP["by_obj"], _MP["W"], _MP["H"]).save(
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

        objs = {}
        for o in scene.get("objects", []):
            x1, y1, x2, y2 = [float(v) for v in o["bbox"]]
            w = max(1, int(round(x2 - x1)))
            h = max(1, int(round(y2 - y1)))
            img = _v_load_image(o["masked_image_url"]).convert("RGBA").resize((w, h))
            objs[o["id"]] = {"img": img, "left": int(round(x1)), "top": int(round(y1)), "w": w, "h": h}

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

        nframes = max(1, int(math.ceil(total * fps)))
        frames_dir = os.path.join(work, "frames")
        os.makedirs(frames_dir)
        frame_args = [(fi, fi / fps) for fi in range(nframes)]
        nproc = min(max(1, (os.cpu_count() or 4)), 32)
        if nproc > 1 and nframes > 8:
            import multiprocessing as mp
            with mp.Pool(processes=nproc, initializer=_mp_init,
                         initargs=(base, objs, by_obj, W, H, frames_dir)) as pool:
                pool.map(_mp_frame, frame_args, chunksize=4)
        else:
            _mp_init(base, objs, by_obj, W, H, frames_dir)
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

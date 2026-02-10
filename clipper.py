import os
from collections import Counter
from statistics import median

import ffmpeg
import config
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None

from config import CLIP_PADDING_BEFORE, CLIP_PADDING_AFTER

MIN_CLIP_SEC = getattr(config, "MIN_CLIP_SEC", 10.0)
MAX_CLIP_SEC = getattr(config, "MAX_CLIP_SEC", 180.0)
SPEECH_PADDING_BEFORE_SEC = getattr(config, "SPEECH_PADDING_BEFORE_SEC", 0.0)
SPEECH_PADDING_AFTER_SEC = getattr(config, "SPEECH_PADDING_AFTER_SEC", 0.0)

PHONE_WIDTH = 1080
PHONE_HEIGHT = 1920
OUTER_MARGIN = 24
CONTENT_WIDTH = PHONE_WIDTH - (OUTER_MARGIN * 2)
CONTENT_HEIGHT = PHONE_HEIGHT - (OUTER_MARGIN * 2)
CAMERA_HEIGHT = 800
FACECAM_EDGE_TRIM_PX = 2
GAMEPLAY_HEIGHT_SCALE = 0.6
GAMEPLAY_SOURCE_WIDTH_RATIO = 0.6
CHAT_OVERLAY_WIDTH = CONTENT_WIDTH
CHAT_OVERLAY_HEIGHT = 500

LONGFORM_WIDTH = 1920
LONGFORM_HEIGHT = 1080
LONGFORM_MARGIN = 24
LONGFORM_CHAT_MAX_HEIGHT = 360
LONGFORM_CHAT_MAX_WIDTH = 720


def _parse_highlight_item(item):
    if isinstance(item, dict):
        start = float(item["start"])
        end = float(item["end"])
        return start, end

    start, end = item
    return float(start), float(end)


def _clamp_int(value, low, high):
    return int(max(low, min(high, value)))


def _build_crop_from_face(frame_w, frame_h, face_x, face_y, face_w, face_h):
    target_aspect = CONTENT_WIDTH / float(CAMERA_HEIGHT)

    # Keep crop tighter around the face so gameplay/background is minimized.
    crop_h = int(max(face_h * 2.8, frame_h * 0.12))
    crop_h = min(crop_h, int(frame_h * 0.42))

    crop_w = int(crop_h * target_aspect)
    if crop_w < int(face_w * 2.0):
        crop_w = int(face_w * 2.0)
        crop_h = int(crop_w / target_aspect)

    crop_w = min(crop_w, frame_w)
    crop_h = min(crop_h, frame_h)

    center_x = face_x + (face_w / 2.0)
    center_y = face_y + (face_h * 0.52)

    crop_x = int(round(center_x - (crop_w / 2.0)))
    crop_y = int(round(center_y - (crop_h * 0.50)))

    crop_x = _clamp_int(crop_x, 0, max(0, frame_w - crop_w))
    crop_y = _clamp_int(crop_y, 0, max(0, frame_h - crop_h))

    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def _default_facecam_crop(frame_w, frame_h):
    target_aspect = CONTENT_WIDTH / float(CAMERA_HEIGHT)
    crop_w = int(frame_w * 0.24)
    crop_h = int(crop_w / target_aspect)

    if crop_h > int(frame_h * 0.35):
        crop_h = int(frame_h * 0.35)
        crop_w = int(crop_h * target_aspect)

    crop_w = max(1, min(crop_w, frame_w))
    crop_h = max(1, min(crop_h, frame_h))
    margin = 10

    crop_x = max(0, frame_w - crop_w - margin)
    crop_y = margin
    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def _weighted_peak_coord(profile, coords, target, span):
    if profile.size == 0:
        return None, None

    coords = coords.astype(np.float32)
    weights = 1.0 / (1.0 + (np.abs(coords - float(target)) / max(float(span), 1.0)))
    weighted = profile * weights
    idx = int(np.argmax(weighted))
    return int(coords[idx]), float(weighted[idx])


def _edge_border_strength(edges, x, y, w, h):
    if w < 4 or h < 4:
        return 0.0

    edge_map = (edges > 0).astype(np.float32)
    t = max(1, int(min(w, h) * 0.02))

    top = edge_map[y:y + t, x:x + w].mean()
    bottom = edge_map[y + h - t:y + h, x:x + w].mean()
    left = edge_map[y:y + h, x:x + t].mean()
    right = edge_map[y:y + h, x + w - t:x + w].mean()
    return float((top + bottom + left + right) / 4.0)


def _detect_rect_panel_from_contours(frame, face_x, face_y, face_w, face_h):
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    median_luma = float(np.median(gray))
    canny_low = int(max(20, min(120, median_luma * 0.66)))
    canny_high = int(max(canny_low + 20, min(240, median_luma * 1.33)))
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(frame_w * frame_h)
    face_cx = face_x + (face_w / 2.0)
    face_cy = face_y + (face_h / 2.0)
    face_area = float(face_w * face_h)
    best = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.0015:
            continue

        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        if len(approx) < 4 or len(approx) > 10:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < int(face_w * 1.7) or h < int(face_h * 1.7):
            continue
        if w > int(frame_w * 0.75) or h > int(frame_h * 0.75):
            continue

        ar = w / float(max(h, 1))
        if ar < 0.65 or ar > 2.8:
            continue

        if not (x <= face_cx <= (x + w) and y <= face_cy <= (y + h)):
            continue

        face_overlap = (
            max(0, min(x + w, face_x + face_w) - max(x, face_x)) *
            max(0, min(y + h, face_y + face_h) - max(y, face_y))
        )
        if face_overlap < face_area * 0.65:
            continue

        rectangularity = area / float(max(w * h, 1))
        if rectangularity < 0.45:
            continue

        border_strength = _edge_border_strength(edges, x, y, w, h)
        if border_strength < 0.06:
            continue

        area_ratio = (w * h) / frame_area
        if area_ratio < 0.015 or area_ratio > 0.45:
            continue

        # Favors well-defined rectangular borders with realistic camera-panel size.
        score = (
            (border_strength * 5.5) +
            (rectangularity * 2.0) -
            (abs(ar - 1.45) * 0.35) -
            (abs(area_ratio - 0.08) * 1.4)
        )
        if best is None or score > best[0]:
            best = (score, x, y, w, h)

    if best is None:
        return None

    _, x, y, w, h = best
    pad_x = int(round(w * 0.02))
    pad_y = int(round(h * 0.02))
    x = _clamp_int(x - pad_x, 0, frame_w - 2)
    y = _clamp_int(y - pad_y, 0, frame_h - 2)
    w = _clamp_int(w + (pad_x * 2), 2, frame_w - x)
    h = _clamp_int(h + (pad_y * 2), 2, frame_h - y)
    return {"x": x, "y": y, "w": w, "h": h}


def _detect_panel_box(frame, face_x, face_y, face_w, face_h):
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).astype(np.float32)
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).astype(np.float32)
    edges = cv2.Canny(gray, 80, 160).astype(np.float32)

    vy0 = _clamp_int(face_y - int(face_h * 2.5), 0, frame_h - 2)
    vy1 = _clamp_int(face_y + int(face_h * 3.0), vy0 + 2, frame_h)

    left_x0 = _clamp_int(face_x - int(face_w * 5.5), 1, frame_w - 3)
    left_x1 = _clamp_int(face_x - int(face_w * 0.15), left_x0 + 1, frame_w - 2)
    if left_x1 <= left_x0:
        return None

    left_coords = np.arange(left_x0, left_x1 + 1)
    left_profile = (
        grad_x[vy0:vy1, left_x0:left_x1 + 1].mean(axis=0) +
        (edges[vy0:vy1, left_x0:left_x1 + 1].mean(axis=0) * 0.8)
    )
    left_target = face_x - int(face_w * 1.25)
    left_x, left_score = _weighted_peak_coord(left_profile, left_coords, left_target, face_w * 2.5)
    if left_x is None:
        return None

    right_x0 = _clamp_int(face_x + int(face_w * 1.05), left_x + 2, frame_w - 2)
    right_x1 = _clamp_int(face_x + int(face_w * 7.0), right_x0 + 1, frame_w - 2)
    if right_x1 <= right_x0:
        return None

    right_coords = np.arange(right_x0, right_x1 + 1)
    right_profile = (
        grad_x[vy0:vy1, right_x0:right_x1 + 1].mean(axis=0) +
        (edges[vy0:vy1, right_x0:right_x1 + 1].mean(axis=0) * 0.8)
    )
    right_target = face_x + int(face_w * 2.25)
    right_x, right_score = _weighted_peak_coord(right_profile, right_coords, right_target, face_w * 3.0)
    if right_x is None:
        return None

    panel_w = right_x - left_x
    if panel_w < int(face_w * 1.8) or panel_w > int(face_w * 10.0):
        return None

    hx0 = _clamp_int(left_x - int(face_w * 0.4), 0, frame_w - 2)
    hx1 = _clamp_int(right_x + int(face_w * 0.4), hx0 + 2, frame_w)

    top_y0 = _clamp_int(face_y - int(face_h * 4.5), 1, frame_h - 3)
    top_y1 = _clamp_int(face_y - int(face_h * 0.08), top_y0 + 1, frame_h - 2)
    if top_y1 <= top_y0:
        return None

    top_coords = np.arange(top_y0, top_y1 + 1)
    top_profile = (
        grad_y[top_y0:top_y1 + 1, hx0:hx1].mean(axis=1) +
        (edges[top_y0:top_y1 + 1, hx0:hx1].mean(axis=1) * 0.8)
    )
    top_target = face_y - int(face_h * 1.3)
    top_y, top_score = _weighted_peak_coord(top_profile, top_coords, top_target, face_h * 2.0)
    if top_y is None:
        return None

    bot_y0 = _clamp_int(face_y + int(face_h * 1.05), top_y + 2, frame_h - 2)
    bot_y1 = _clamp_int(face_y + int(face_h * 7.0), bot_y0 + 1, frame_h - 2)
    if bot_y1 <= bot_y0:
        return None

    bot_coords = np.arange(bot_y0, bot_y1 + 1)
    bot_profile = (
        grad_y[bot_y0:bot_y1 + 1, hx0:hx1].mean(axis=1) +
        (edges[bot_y0:bot_y1 + 1, hx0:hx1].mean(axis=1) * 0.8)
    )
    bot_target = face_y + int(face_h * 2.4)
    bottom_y, bottom_score = _weighted_peak_coord(bot_profile, bot_coords, bot_target, face_h * 3.0)
    if bottom_y is None:
        return None

    panel_h = bottom_y - top_y
    if panel_h < int(face_h * 1.8) or panel_h > int(face_h * 10.0):
        return None

    # Reject low-confidence panel picks.
    if min(left_score, right_score, top_score, bottom_score) < 12.0:
        return None

    x = _clamp_int(left_x, 0, frame_w - 2)
    y = _clamp_int(top_y, 0, frame_h - 2)
    w = _clamp_int(panel_w, 2, frame_w - x)
    h = _clamp_int(panel_h, 2, frame_h - y)
    return {"x": x, "y": y, "w": w, "h": h}


def detect_facecam_crop(video_path, sample_count=180, grid_size=6):
    if cv2 is None:
        print("OpenCV unavailable; using fallback top-right crop for facecam.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open VOD for facecam detection; using fallback crop.")
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    duration_sec = (frame_count / fps) if fps > 0 else 0

    if frame_w <= 0 or frame_h <= 0 or duration_sec <= 0:
        cap.release()
        print("Missing video metadata for facecam detection; using fallback crop.")
        return None

    classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    detections = []
    for i in range(sample_count):
        t = duration_sec * ((i + 1) / float(sample_count + 1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_face = max(24, int(min(frame_w, frame_h) * 0.03))
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )
        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        panel_box = _detect_rect_panel_from_contours(frame, x, y, w, h)
        if panel_box is None:
            panel_box = _detect_panel_box(frame, x, y, w, h)
        if panel_box is None:
            panel_box = _build_crop_from_face(frame_w, frame_h, x, y, w, h)

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        gx = int(min(grid_size - 1, max(0, (cx / frame_w) * grid_size)))
        gy = int(min(grid_size - 1, max(0, (cy / frame_h) * grid_size)))
        detections.append((
            gx, gy,
            x, y, w, h,
            panel_box["x"], panel_box["y"], panel_box["w"], panel_box["h"],
        ))

    cap.release()

    if not detections:
        print("No face detections found; using fallback top-right crop.")
        return _default_facecam_crop(frame_w, frame_h)

    bucket_counts = Counter((d[0], d[1]) for d in detections)
    best_bucket, best_count = bucket_counts.most_common(1)[0]
    bucket_detections = [d for d in detections if (d[0], d[1]) == best_bucket]

    # If detections are too fragmented, use all detections.
    if best_count < max(4, int(len(detections) * 0.25)):
        bucket_detections = detections

    panel_xs = [d[6] for d in bucket_detections]
    panel_ys = [d[7] for d in bucket_detections]
    panel_ws = [d[8] for d in bucket_detections]
    panel_hs = [d[9] for d in bucket_detections]

    crop = {
        "x": _clamp_int(int(median(panel_xs)), 0, frame_w - 2),
        "y": _clamp_int(int(median(panel_ys)), 0, frame_h - 2),
        "w": _clamp_int(int(median(panel_ws)), 2, frame_w),
        "h": _clamp_int(int(median(panel_hs)), 2, frame_h),
    }
    crop["w"] = min(crop["w"], frame_w - crop["x"])
    crop["h"] = min(crop["h"], frame_h - crop["y"])
    print(
        "Detected facecam crop:",
        f"x={crop['x']}, y={crop['y']}, w={crop['w']}, h={crop['h']}",
        f"(detections={len(detections)})",
    )
    return crop


def _render_vertical_clip(video_path, chat_path, chat_mask_path, out_path, start, end, crop):
    vod = ffmpeg.input(video_path, ss=start, to=end)
    chat = ffmpeg.input(chat_path, ss=start, to=end)
    chat_mask = None
    if chat_mask_path and os.path.exists(chat_mask_path):
        chat_mask = ffmpeg.input(chat_mask_path, ss=start, to=end)

    panel_x = int((PHONE_WIDTH - CONTENT_WIDTH) / 2)

    # Keep gameplay behind the facecam, but pin gameplay to the bottom.
    gameplay_width = CONTENT_WIDTH
    gameplay_height = _clamp_int(
        int(round(CONTENT_HEIGHT * GAMEPLAY_HEIGHT_SCALE)),
        2,
        CONTENT_HEIGHT,
    )
    gameplay_x = panel_x
    gameplay_y = PHONE_HEIGHT - OUTER_MARGIN - gameplay_height

    # Use a wider gameplay source slice so more scene width remains visible in portrait.
    gameplay = vod.video.filter(
        "crop",
        f"floor(iw*{GAMEPLAY_SOURCE_WIDTH_RATIO}/2)*2",
        "ih",
        f"floor(iw*{(1.0 - GAMEPLAY_SOURCE_WIDTH_RATIO) / 2.0}/2)*2",
        "0",
    )
    gameplay = gameplay.filter(
        "scale",
        gameplay_width,
        gameplay_height,
        force_original_aspect_ratio="increase",
    )
    gameplay = gameplay.filter("crop", gameplay_width, gameplay_height)
    gameplay = gameplay.filter(
        "pad",
        PHONE_WIDTH,
        PHONE_HEIGHT,
        str(gameplay_x),
        str(gameplay_y),
        color="black",
    )

    trim_x = min(FACECAM_EDGE_TRIM_PX, max(0, (crop["w"] - 2) // 2))
    trim_y = min(FACECAM_EDGE_TRIM_PX, max(0, (crop["h"] - 2) // 2))
    face_w = max(2, crop["w"] - (trim_x * 2))
    face_h = max(2, crop["h"] - (trim_y * 2))
    face_x_src = crop["x"] + trim_x
    face_y_src = crop["y"] + trim_y

    face = vod.video.filter("crop", face_w, face_h, face_x_src, face_y_src)
    face = face.filter(
        "scale",
        -2,
        CAMERA_HEIGHT,
    )

    chat_video = chat.video.filter("format", "rgba").filter(
        "scale",
        CHAT_OVERLAY_WIDTH,
        CHAT_OVERLAY_HEIGHT,
        force_original_aspect_ratio="decrease",
    )
    chat_video = chat_video.filter(
        "pad",
        CHAT_OVERLAY_WIDTH,
        CHAT_OVERLAY_HEIGHT,
        "0",
        "(oh-ih)/2",
        color="black@0.0",
    )
    if chat_mask is not None:
        chat_alpha = chat_mask.video.filter(
            "scale",
            CHAT_OVERLAY_WIDTH,
            CHAT_OVERLAY_HEIGHT,
            force_original_aspect_ratio="decrease",
        )
        chat_alpha = chat_alpha.filter(
            "pad",
            CHAT_OVERLAY_WIDTH,
            CHAT_OVERLAY_HEIGHT,
            "0",
            "(oh-ih)/2",
            color="black",
        )
        chat_alpha = chat_alpha.filter("format", "gray")
        chat_overlay = ffmpeg.filter([chat_video, chat_alpha], "alphamerge")
    else:
        # Fallback when no mask file is available.
        chat_overlay = chat_video.filter("colorchannelmixer", aa=0.90)

    face_x = "(main_w-overlay_w)/2"
    face_y = OUTER_MARGIN
    chat_x = gameplay_x
    chat_y = gameplay_y

    with_chat = ffmpeg.overlay(gameplay, chat_overlay, x=chat_x, y=chat_y, eof_action="pass")
    composed = ffmpeg.overlay(with_chat, face, x=face_x, y=face_y, eof_action="pass")
    (
        ffmpeg
        .output(
            composed,
            vod.audio,
            out_path,
            vcodec="libx264",
            acodec="aac",
            preset="veryfast",
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        .run(overwrite_output=True, quiet=True)
    )


def render_longform_with_chat(video_path, highlights, out_path, chat_path=None, chat_mask_path=None):
    if not highlights:
        print("Skipping long-form render: no clips to combine.")
        return False

    chat_enabled = bool(chat_path and os.path.exists(chat_path))
    mask_enabled = bool(chat_mask_path and os.path.exists(chat_mask_path))
    streams = []

    for item in highlights:
        start, end = _parse_highlight_item(item)
        if end <= start:
            continue

        vod_seg = ffmpeg.input(video_path, ss=start, to=end)
        base = vod_seg.video.filter(
            "scale",
            LONGFORM_WIDTH,
            LONGFORM_HEIGHT,
            force_original_aspect_ratio="decrease",
        )
        base = base.filter(
            "pad",
            LONGFORM_WIDTH,
            LONGFORM_HEIGHT,
            "(ow-iw)/2",
            "(oh-ih)/2",
            color="black",
        )

        if chat_enabled:
            chat_seg = ffmpeg.input(chat_path, ss=start, to=end)
            chat_video = chat_seg.video.filter("format", "rgba")
            chat_video = chat_video.filter(
                "scale",
                LONGFORM_CHAT_MAX_WIDTH,
                LONGFORM_CHAT_MAX_HEIGHT,
                force_original_aspect_ratio="decrease",
            )

            if mask_enabled:
                chat_mask_seg = ffmpeg.input(chat_mask_path, ss=start, to=end)
                chat_alpha = chat_mask_seg.video.filter(
                    "scale",
                    LONGFORM_CHAT_MAX_WIDTH,
                    LONGFORM_CHAT_MAX_HEIGHT,
                    force_original_aspect_ratio="decrease",
                )
                chat_alpha = chat_alpha.filter("format", "gray")
                chat_overlay = ffmpeg.filter([chat_video, chat_alpha], "alphamerge")
            else:
                chat_overlay = chat_video.filter("colorchannelmixer", aa=0.90)

            base = ffmpeg.overlay(
                base,
                chat_overlay,
                x=LONGFORM_MARGIN,
                y=f"main_h-overlay_h-{LONGFORM_MARGIN}",
                eof_action="pass",
            )

        streams.extend([base, vod_seg.audio])

    if not streams:
        print("Skipping long-form render: no valid clip ranges to combine.")
        return False

    joined = ffmpeg.concat(*streams, v=1, a=1).node
    vout = joined[0]
    aout = joined[1]
    (
        ffmpeg
        .output(
            vout,
            aout,
            out_path,
            vcodec="libx264",
            acodec="aac",
            preset="veryfast",
            pix_fmt="yuv420p",
            movflags="+faststart",
        )
        .run(overwrite_output=True, quiet=True)
    )
    return True


def cut_clips(video_path, highlights, out_dir, chat_path=None, chat_mask_path=None):
    written = 0
    prev_written_end = 0.0
    if chat_path and not os.path.exists(chat_path):
        raise FileNotFoundError(
            f"Expected chat video at {chat_path} for vertical mode, but it was not found."
        )

    vertical_mode = bool(chat_path)
    facecam_crop = None

    if vertical_mode:
        facecam_crop = detect_facecam_crop(video_path)
        if not facecam_crop:
            probe = ffmpeg.probe(video_path)
            v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
            frame_w = int(v_stream["width"])
            frame_h = int(v_stream["height"])
            facecam_crop = _default_facecam_crop(frame_w, frame_h)
            print(
                "Using fallback facecam crop:",
                f"x={facecam_crop['x']}, y={facecam_crop['y']}, "
                f"w={facecam_crop['w']}, h={facecam_crop['h']}",
            )
        if chat_mask_path and os.path.exists(chat_mask_path):
            print("Vertical output mode enabled (facecam top + bottom-left transparent chat overlay).")
        else:
            print("Vertical output mode enabled (facecam top + bottom-left chat overlay; mask missing).")
    else:
        print("Vertical mode disabled (no chat video provided). Producing standard clips.")

    for i, item in enumerate(highlights):
        base_start, base_end = _parse_highlight_item(item)
        base_duration = base_end - base_start
        if base_duration < MIN_CLIP_SEC or base_duration > MAX_CLIP_SEC:
            print(
                f"Skipping clip candidate {i}: base duration {base_duration:.1f}s "
                f"is outside {MIN_CLIP_SEC:.0f}s-{MAX_CLIP_SEC:.0f}s"
            )
            continue

        available_extra = max(0.0, MAX_CLIP_SEC - base_duration)
        pad_before = min(
            float(CLIP_PADDING_BEFORE),
            float(SPEECH_PADDING_BEFORE_SEC),
            available_extra / 2.0,
            base_start,
        )
        remaining_extra = max(0.0, available_extra - pad_before)
        pad_after = min(
            float(CLIP_PADDING_AFTER),
            float(SPEECH_PADDING_AFTER_SEC),
            remaining_extra,
        )

        s = max(0.0, base_start - pad_before)
        e = base_end + pad_after

        # Hard guard: outputs must never overlap.
        if s < prev_written_end:
            s = prev_written_end

        final_duration = e - s
        if final_duration < MIN_CLIP_SEC or final_duration > MAX_CLIP_SEC:
            print(
                f"Skipping clip candidate {i}: final duration {final_duration:.1f}s "
                f"is outside {MIN_CLIP_SEC:.0f}s-{MAX_CLIP_SEC:.0f}s"
            )
            continue

        filename = f"clip_{written:02d}.mp4"

        print(f"Cutting clip {written}: {s:.2f}s to {e:.2f}s ({final_duration:.1f}s)")

        if vertical_mode:
            _render_vertical_clip(
                video_path=video_path,
                chat_path=chat_path,
                chat_mask_path=chat_mask_path,
                out_path=f"{out_dir}/{filename}",
                start=s,
                end=e,
                crop=facecam_crop,
            )
        else:
            (
                ffmpeg
                .input(video_path, ss=s, to=e)
                .output(
                    f"{out_dir}/{filename}",
                    vcodec="libx264",
                    acodec="aac",
                    preset="veryfast",
                    movflags="+faststart",
                )
                .run(overwrite_output=True, quiet=True)
            )

        written += 1
        prev_written_end = e

    return written

import concurrent.futures
import difflib
import json
import math
import os
import statistics
import time

import cv2
from ultralytics import YOLO

from price_sources import estimate_value_groq

IGNORED_LABELS = {
    "person", "chair", "couch", "bed", "dining table", "tv",
    "keyboard", "mouse", "book", "potted plant",
}

CONSENSUS_N = 3
CONSENSUS_SPAN_SEC = 0.5
CONSENSUS_MAX_WAIT_SEC = 3.0

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
BUFFER_FLUSH_FRAMES = 30  # discard up to ~1s of stale frames after Groq call

os.makedirs("snapshots", exist_ok=True)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def flush_camera_buffer(cap, n=BUFFER_FLUSH_FRAMES):
    for _ in range(n):
        if not cap.grab():
            return


def call_one(args):
    crop, full, hint, used_fb = args
    try:
        r = estimate_value_groq(crop, yolo_hint=hint, full_frame_path=full)
        r["_used_fallback"] = used_fb
        r["_yolo_hint"] = hint
        return r
    except KeyError:
        return {"error": "GROQ_API_KEY missing"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


_motion_state = {"prev": None}


def compute_motion_mask(frame):
    small = cv2.resize(frame, (160, 120))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    prev = _motion_state["prev"]
    _motion_state["prev"] = gray
    if prev is None:
        return None
    return cv2.absdiff(gray, prev).astype("float32") / 255.0


def _motion_factor(motion_mask, box, frame_w, frame_h):
    if motion_mask is None:
        return 0.0
    mh, mw = motion_mask.shape
    x1, y1, x2, y2 = box
    mx1 = max(0, int(x1 * mw / frame_w))
    my1 = max(0, int(y1 * mh / frame_h))
    mx2 = min(mw, int(x2 * mw / frame_w))
    my2 = min(mh, int(y2 * mh / frame_h))
    if mx2 <= mx1 or my2 <= my1:
        return 0.0
    region = motion_mask[my1:my2, mx1:mx2]
    return float(min(1.0, region.mean() * 5.0))


def score_detection(d, frame_w, frame_h, motion_mask=None):
    x1, y1, x2, y2 = d["box"]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx = (cx - frame_w / 2) / (frame_w / 2)
    dy = (cy - frame_h / 2) / (frame_h / 2)
    centrality = max(0.0, 1.0 - math.hypot(dx, dy))
    area_ratio = (bw * bh) / (frame_w * frame_h)
    if area_ratio < 0.005:
        return 0.0
    score = d["confidence"] * centrality
    score *= 1.0 + _motion_factor(motion_mask, d["box"], frame_w, frame_h)
    return score


def center_crop_box(frame_w, frame_h, ratio=0.6):
    cw, ch = int(frame_w * ratio), int(frame_h * ratio)
    x1 = (frame_w - cw) // 2
    y1 = (frame_h - ch) // 2
    return (x1, y1, x1 + cw, y1 + ch)


def pad_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(frame_w, x2 + pad),
        min(frame_h, y2 + pad),
    )


def cluster_strings(strings):
    clean = [s for s in strings if s]
    if not clean:
        return None, 0
    clusters = []
    for s in clean:
        placed = False
        for cl in clusters:
            if difflib.SequenceMatcher(None, s.lower(), cl[0].lower()).ratio() >= 0.6:
                cl.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    clusters.sort(key=len, reverse=True)
    return clusters[0][0], len(clusters[0])


def aggregate_consensus(results):
    by_conf = sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)
    item, agree = cluster_strings([r.get("item") for r in results])
    if agree < 2:
        item = by_conf[0].get("item", "unknown")
    brand, _ = cluster_strings([r.get("brand") for r in results])
    model_str, _ = cluster_strings([r.get("model") for r in results])
    condition, _ = cluster_strings([r.get("condition") for r in results])
    lows = [r.get("low", 0) for r in results if isinstance(r.get("low"), (int, float))]
    highs = [r.get("high", 0) for r in results if isinstance(r.get("high"), (int, float))]
    confs = [r.get("confidence", 0) for r in results if isinstance(r.get("confidence"), (int, float))]
    notes = by_conf[0].get("notes", "")
    return {
        "item": item,
        "brand": brand,
        "model": model_str,
        "condition": condition,
        "confidence": round(statistics.mean(confs), 2) if confs else 0,
        "low": int(statistics.median(lows)) if lows else 0,
        "high": int(statistics.median(highs)) if highs else 0,
        "agreement": f"{agree}/{len(results)}",
        "notes": notes,
    }


def draw_frame(frame, detections, best_idx, groq_result, fallback_crop):
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        is_best = i == best_idx
        label = d["label"]
        if is_best and groq_result and groq_result.get("item"):
            label = groq_result["item"]
        color = (0, 255, 0) if is_best else (0, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if fallback_crop and best_idx == -1:
        x1, y1, x2, y2 = fallback_crop
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(frame, "center-crop fallback (no YOLO lock)", (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)

    if groq_result and groq_result.get("item"):
        low = groq_result.get("low", "?")
        high = groq_result.get("high", "?")
        cond = groq_result.get("condition", "?")
        conf = groq_result.get("confidence", 0)
        agree = groq_result.get("agreement", "")
        banner = f"ID: {groq_result['item']} [{cond}, conf {conf}, agree {agree}]  |  ${low}-${high}"
    else:
        banner = "Hold an item up and press 's' for price estimate  |  'q' to quit"

    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(frame, banner, (10, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def _draw_status_overlay(frame, text, color=(0, 180, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), color, 8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return frame


def capture_consensus_crops(cap, detection_fn, window_name):
    captured = []
    start = time.time()
    timestamp = int(start)
    next_capture_target = 0.0

    while len(captured) < CONSENSUS_N and (time.time() - start) < CONSENSUS_MAX_WAIT_SEC:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time() - start
        status = f"CAPTURING {len(captured) + 1}/{CONSENSUS_N}..."
        cv2.imshow(window_name, _draw_status_overlay(frame.copy(), status))
        cv2.waitKey(1)

        if now < next_capture_target:
            continue

        best_idx, detections, fallback_crop = detection_fn(frame)
        h, w = frame.shape[:2]

        if best_idx >= 0:
            box = pad_box(detections[best_idx]["box"], w, h)
            yolo_hint = detections[best_idx]["label"]
            used_fallback = False
        else:
            box = fallback_crop
            yolo_hint = None
            used_fallback = True

        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        idx = len(captured)
        crop_path = f"snapshots/snapshot_{timestamp}_{idx}.jpg"
        full_path = f"snapshots/snapshot_{timestamp}_{idx}_full.jpg"
        cv2.imwrite(crop_path, crop)
        cv2.imwrite(full_path, frame)
        captured.append((crop_path, full_path, yolo_hint, used_fallback))
        next_capture_target = now + (CONSENSUS_SPAN_SEC / max(1, CONSENSUS_N - 1))

    return captured


def run_detection(frame):
    h, w = frame.shape[:2]
    motion_mask = compute_motion_mask(frame)
    results = model(frame, verbose=False, conf=0.15)
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        if label in IGNORED_LABELS:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "box": (int(x1), int(y1), int(x2), int(y2)),
        })
    scored = [(score_detection(d, w, h, motion_mask), i) for i, d in enumerate(detections)]
    scored = [t for t in scored if t[0] >= 0.05]
    best_idx = max(scored)[1] if scored else -1
    fallback_crop = center_crop_box(w, h) if best_idx == -1 else None
    return best_idx, detections, fallback_crop


WINDOW_NAME = "Object Detector - press 's' for price"

print(f"Camera index: {CAMERA_INDEX} (override with CAMERA_INDEX env var)")
print("Hold an item up to the camera and press 's' to capture + price it. 'q' to quit.")

last_seen = set()
groq_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    best_idx, detections, fallback_crop = run_detection(frame)

    current = {d["label"] for d in detections if d["confidence"] > 0.5}
    if current != last_seen:
        print(f"Now seeing (YOLO): {current}")
        last_seen = current

    if not detections and best_idx == -1 and not fallback_crop:
        groq_result = None

    annotated = draw_frame(frame.copy(), detections, best_idx, groq_result, fallback_crop)
    cv2.imshow(WINDOW_NAME, annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        print(f"\n[capture] collecting {CONSENSUS_N} frames across ~{CONSENSUS_SPAN_SEC}s...")
        captured = capture_consensus_crops(cap, run_detection, WINDOW_NAME)
        if not captured:
            print("couldn't capture any frames - check the camera.")
            continue
        print(f"[capture] got {len(captured)} frames. Firing {len(captured)} Groq calls in parallel...")

        ret_a, last_frame = cap.read()
        if ret_a:
            cv2.imshow(WINDOW_NAME, _draw_status_overlay(last_frame.copy(), "ANALYZING...", color=(0, 200, 0)))
            cv2.waitKey(1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONSENSUS_N) as ex:
            raw_results = list(ex.map(call_one, captured))

        for i, r in enumerate(raw_results):
            print(f"\n  [{i}] {json.dumps(r, indent=2)}")

        valid = [r for r in raw_results if "error" not in r]
        if not valid:
            print("\nall Groq calls failed. Check your API key and network.")
            flush_camera_buffer(cap)
            _motion_state["prev"] = None
            continue

        consensus = aggregate_consensus(valid)
        print(f"\n=== CONSENSUS ===\n{json.dumps(consensus, indent=2)}\n")
        groq_result = consensus

        flush_camera_buffer(cap)
        _motion_state["prev"] = None

cap.release()
cv2.destroyAllWindows()

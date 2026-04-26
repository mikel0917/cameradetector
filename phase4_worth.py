"""Phase 4 — Real-time webcam object detector with Groq-powered price estimates.

Press 's' to capture three snapshots, identify the item with Groq, and aggregate
into a consensus price range. 'q' or window-X to quit. CAMERA_INDEX env var
selects the webcam (default 0).
"""

import concurrent.futures
import difflib
import json
import math
import os
import statistics
import time

import cv2
import numpy as np
from ultralytics import YOLO

from price_sources import estimate_value_groq

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Detection scoring (per-frame)
YOLO_CONF_THRESHOLD = 0.15
MIN_AREA_RATIO = 0.005          # boxes smaller than this fraction of the frame are noise
COMPOSITE_SCORE_FLOOR = 0.05    # below this, treat as no-lock and use fallback
MOTION_SCALE = 5.0              # multiplier applied to mean motion intensity per box
CROP_PAD_RATIO = 0.15           # pad each crop by this fraction of its longest side
CENTER_CROP_RATIO = 0.6         # if all else fails, crop the central 60% of the frame
MOTION_FALLBACK_AREA = 0.01     # min motion-blob area (vs full frame) to be a usable fallback
MOTION_FALLBACK_THRESHOLD = 0.05  # per-pixel motion intensity to count as "moving"

# Multi-frame consensus
CONSENSUS_N = 3
CONSENSUS_SPAN_SEC = 0.5
CONSENSUS_MAX_WAIT_SEC = 3.0

# Camera
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
BUFFER_FLUSH_FRAMES = 30        # discard up to ~1s of stale frames after Groq call

# UI
WINDOW_NAME = "Object Detector - press 's' for price"
FONT = cv2.FONT_HERSHEY_SIMPLEX
ERROR_OVERLAY_MS = 1500         # how long to flash an error message after a failed call

IGNORED_LABELS = frozenset({
    "person", "chair", "couch", "bed", "dining table", "tv",
    "keyboard", "mouse", "book", "potted plant",
})

SNAPSHOT_DIR = "snapshots"


# ---------------------------------------------------------------------------
# Motion tracking
# ---------------------------------------------------------------------------

class MotionTracker:
    """Per-pixel frame-difference, downsampled for speed."""

    def __init__(self, target_size=(160, 120)):
        self.target_size = target_size
        self.prev = None

    def update(self, frame):
        small = cv2.resize(frame, self.target_size)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prev = self.prev
        self.prev = gray
        if prev is None:
            return None
        return cv2.absdiff(gray, prev).astype("float32") / 255.0

    def reset(self):
        self.prev = None


# ---------------------------------------------------------------------------
# Box / scoring helpers
# ---------------------------------------------------------------------------

def motion_factor(motion_mask, box, frame_w, frame_h):
    """Mean motion intensity inside the box, normalized 0-1."""
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
    return float(min(1.0, region.mean() * MOTION_SCALE))


def score_detection(d, frame_w, frame_h, motion_mask):
    """confidence × centrality × (1 + motion). Returns 0 to reject."""
    x1, y1, x2, y2 = d["box"]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx = (cx - frame_w / 2) / (frame_w / 2)
    dy = (cy - frame_h / 2) / (frame_h / 2)
    centrality = max(0.0, 1.0 - math.hypot(dx, dy))
    if (bw * bh) / (frame_w * frame_h) < MIN_AREA_RATIO:
        return 0.0
    return d["confidence"] * centrality * (1.0 + motion_factor(motion_mask, d["box"], frame_w, frame_h))


def center_crop_box(frame_w, frame_h, ratio=CENTER_CROP_RATIO):
    cw, ch = int(frame_w * ratio), int(frame_h * ratio)
    x1 = (frame_w - cw) // 2
    y1 = (frame_h - ch) // 2
    return (x1, y1, x1 + cw, y1 + ch)


def motion_bbox(motion_mask, frame_w, frame_h):
    """Bounding box of the largest moving region. Returns None if no significant motion."""
    if motion_mask is None:
        return None
    binary = (motion_mask > MOTION_FALLBACK_THRESHOLD).astype("uint8") * 255
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MOTION_FALLBACK_AREA * binary.size:
        return None
    mh, mw = motion_mask.shape
    x, y, w, h = cv2.boundingRect(largest)
    return (
        int(x * frame_w / mw),
        int(y * frame_h / mh),
        int((x + w) * frame_w / mw),
        int((y + h) * frame_h / mh),
    )


def pad_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    pad = int(CROP_PAD_RATIO * max(x2 - x1, y2 - y1))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(frame_w, x2 + pad),
        min(frame_h, y2 + pad),
    )


# ---------------------------------------------------------------------------
# Consensus aggregation
# ---------------------------------------------------------------------------

def cluster_strings(strings, threshold=0.6):
    """Group similar strings; return (representative, cluster_size) of the largest cluster."""
    clean = [s for s in strings if s]
    if not clean:
        return None, 0
    clusters = []
    for s in clean:
        for cl in clusters:
            if difflib.SequenceMatcher(None, s.lower(), cl[0].lower()).ratio() >= threshold:
                cl.append(s)
                break
        else:
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
    return {
        "item": item,
        "brand": brand,
        "model": model_str,
        "condition": condition,
        "confidence": round(statistics.mean(confs), 2) if confs else 0,
        "low": int(statistics.median(lows)) if lows else 0,
        "high": int(statistics.median(highs)) if highs else 0,
        "agreement": f"{agree}/{len(results)}",
        "notes": by_conf[0].get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hud(frame, detections, best_idx, groq_result, fallback_box, fallback_kind):
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        is_best = i == best_idx
        label = d["label"]
        if is_best and groq_result and groq_result.get("item"):
            label = groq_result["item"]
        color = (0, 255, 0) if is_best else (0, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 6), FONT, 0.6, (0, 0, 0), 2)

    if best_idx == -1 and fallback_box is not None:
        x1, y1, x2, y2 = fallback_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        note = "motion-region fallback" if fallback_kind == "motion" else "center-crop fallback"
        cv2.putText(frame, note, (x1 + 5, y1 + 20), FONT, 0.5, (0, 180, 255), 2)

    if groq_result and groq_result.get("item"):
        low = groq_result.get("low", "?")
        high = groq_result.get("high", "?")
        cond = groq_result.get("condition", "?")
        conf = groq_result.get("confidence", 0)
        agree = groq_result.get("agreement", "")
        banner = f"ID: {groq_result['item']} [{cond}, conf {conf}, agree {agree}]  |  ${low}-${high}"
    else:
        banner = "Hold an item up and press 's' for price estimate  |  'q' to quit"

    (tw, th), _ = cv2.getTextSize(banner, FONT, 0.6, 2)
    cv2.rectangle(frame, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(frame, banner, (10, th + 8), FONT, 0.6, (255, 255, 255), 2)
    return frame


def draw_status_overlay(frame, text, color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), color, 8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.0, 3)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, 1.0, color, 3)
    return frame


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_detection(model, motion_tracker, frame):
    """Returns (best_idx, detections, fallback_box, fallback_kind).
    fallback_kind is '' when YOLO has a winner, else 'motion' or 'center'.
    """
    h, w = frame.shape[:2]
    motion_mask = motion_tracker.update(frame)
    results = model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)
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
    scored = [t for t in scored if t[0] >= COMPOSITE_SCORE_FLOOR]
    if scored:
        return max(scored)[1], detections, None, ""

    mb = motion_bbox(motion_mask, w, h)
    if mb is not None:
        return -1, detections, mb, "motion"
    return -1, detections, center_crop_box(w, h), "center"


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def capture_consensus_crops(cap, model, motion_tracker, window_name):
    captured = []
    start = time.time()
    timestamp = int(start)
    next_capture_target = 0.0

    while len(captured) < CONSENSUS_N and (time.time() - start) < CONSENSUS_MAX_WAIT_SEC:
        ret, frame = cap.read()
        if not ret:
            continue

        status = f"CAPTURING {len(captured) + 1}/{CONSENSUS_N}..."
        cv2.imshow(window_name, draw_status_overlay(frame.copy(), status, (0, 180, 255)))
        cv2.waitKey(1)

        now = time.time() - start
        if now < next_capture_target:
            continue

        best_idx, detections, fallback_box, _ = run_detection(model, motion_tracker, frame)
        h, w = frame.shape[:2]

        if best_idx >= 0:
            box = pad_box(detections[best_idx]["box"], w, h)
            yolo_hint = detections[best_idx]["label"]
            used_fallback = False
        else:
            box = fallback_box
            yolo_hint = None
            used_fallback = True

        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        idx = len(captured)
        crop_path = f"{SNAPSHOT_DIR}/snapshot_{timestamp}_{idx}.jpg"
        full_path = f"{SNAPSHOT_DIR}/snapshot_{timestamp}_{idx}_full.jpg"
        cv2.imwrite(crop_path, crop)
        cv2.imwrite(full_path, frame)
        captured.append((crop_path, full_path, yolo_hint, used_fallback))
        next_capture_target = now + (CONSENSUS_SPAN_SEC / max(1, CONSENSUS_N - 1))

    return captured


def fire_groq_calls(captured):
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONSENSUS_N) as ex:
        return list(ex.map(call_one, captured))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    motion_tracker = MotionTracker()

    print(f"Camera index: {CAMERA_INDEX} (override with CAMERA_INDEX env var)")
    print("Hold an item up to the camera and press 's' to capture + price it. 'q' or window-X to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    groq_result = None
    analyzing = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("camera read failed - exiting.")
                break

            best_idx, detections, fallback_box, fallback_kind = run_detection(model, motion_tracker, frame)

            if not detections and best_idx == -1 and fallback_box is None:
                groq_result = None

            annotated = draw_hud(frame.copy(), detections, best_idx, groq_result, fallback_box, fallback_kind)
            cv2.imshow(WINDOW_NAME, annotated)

            # Window-X handling
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and not analyzing:
                analyzing = True
                try:
                    groq_result = run_consensus(cap, model, motion_tracker) or groq_result
                finally:
                    analyzing = False
                    flush_camera_buffer(cap)
                    motion_tracker.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_consensus(cap, model, motion_tracker):
    """Capture + Groq + aggregate. Returns the consensus dict, or None on failure."""
    print(f"\n[capture] collecting {CONSENSUS_N} frames across ~{CONSENSUS_SPAN_SEC}s...")
    captured = capture_consensus_crops(cap, model, motion_tracker, WINDOW_NAME)
    if not captured:
        flash_error("CAMERA ERROR")
        print("couldn't capture any frames - check the camera.")
        return None

    print(f"[capture] got {len(captured)} frames. Firing {len(captured)} Groq calls in parallel...")

    ret_a, last_frame = cap.read()
    if ret_a:
        cv2.imshow(WINDOW_NAME, draw_status_overlay(last_frame.copy(), "ANALYZING...", (0, 200, 0)))
        cv2.waitKey(1)

    raw_results = fire_groq_calls(captured)
    for i, r in enumerate(raw_results):
        print(f"\n  [{i}] {json.dumps(r, indent=2)}")

    valid = [r for r in raw_results if "error" not in r]
    if not valid:
        first_err = next((r["error"] for r in raw_results if "error" in r), "unknown")
        flash_error(f"GROQ FAILED: {first_err[:60]}")
        print(f"\nall Groq calls failed: {first_err}")
        return None

    consensus = aggregate_consensus(valid)
    print(f"\n=== CONSENSUS ===\n{json.dumps(consensus, indent=2)}\n")
    return consensus


def flash_error(text):
    """Show a red full-frame error overlay for ERROR_OVERLAY_MS."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, draw_status_overlay(frame, text, (0, 0, 255)))
    cv2.waitKey(ERROR_OVERLAY_MS)


if __name__ == "__main__":
    main()

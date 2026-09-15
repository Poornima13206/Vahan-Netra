"""
Real detection + OCR pipeline — runs on actual video frames.

Unlike app.py's seeded demo data, nothing here is faked: this reads real
video frames, finds plate-shaped regions using classical computer vision
(Canny edges + contour geometry — the same family of technique many
production ANPR systems use as a first-pass localizer before/instead of a
deep detector), crops them, and runs them through a real OCR engine
(Tesseract) to extract text with a genuine confidence score.

This is intentionally NOT YOLOv8. YOLOv8 needs a trained/fine-tuned model
and labeled plate data to be more accurate than classical CV on real
photos — without that, an untrained YOLO is not actually better here, and
shipping it would just add a large dependency with no real accuracy gain
for a demo. What this file demonstrates instead is the exact set of
stages any real ANPR system needs (localize -> crop -> OCR -> confidence
-> filter/store), running against real pixels instead of a database seed.
Swapping the localizer for a fine-tuned YOLOv8 model later is a one-
function change: replace `find_plate_candidates()`; everything after it
(OCR, confidence scoring, DB insert) stays the same.

Usage:
    python3 detect_plates.py sample_traffic.mp4 --camera CAM-05 --annotate out.mp4
"""

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract

DB_PATH = Path(__file__).parent / "anpr.db"
PLATE_RE = re.compile(r"^[A-Z0-9]{6,10}$")


def find_plate_candidates(frame):
    """Classical plate localization: edges -> contours -> rectangle
    geometry filter. Returns a list of (x, y, w, h) boxes likely to
    contain a license plate.

    Thresholds here are deliberately generous -- this is a first-pass
    localizer, not the final accuracy gate (OCR confidence + a plate-
    format regex, applied later in the pipeline, do the real filtering).
    Real footage varies a lot in lighting/background/camera angle, so
    expect to tune these two things first if a real clip under- or
    over-detects:
      - aspect ratio bounds (plates look wider/narrower depending on
        camera angle)
      - brightness_delta (how much brighter than its surroundings a
        plate needs to look -- lower this if your footage is well-lit
        and plates don't stand out much, raise it if you're getting
        false positives on other bright surfaces like headlights/sky)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray_smooth, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    frame_mean_brightness = gray.mean()
    brightness_delta = 15  # how much brighter than the frame average a plate region should be

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0 or w == 0:
            continue
        aspect = w / h
        area_ratio = (w * h) / frame_area
        # Widened from the original synthetic-video-only bounds: plates can look
        # closer to square at an angle, and can be a smaller or larger fraction
        # of frame depending on how close the camera is to the road.
        if 1.5 <= aspect <= 6.0 and 0.0003 <= area_ratio <= 0.06:
            roi = gray[y:y + h, x:x + w]
            if roi.size and roi.mean() > frame_mean_brightness + brightness_delta:
                boxes.append((x, y, w, h))
    return boxes


NAMED_COLORS = {
    "white": (240, 240, 240), "silver": (192, 192, 192), "grey": (128, 128, 128),
    "black": (30, 30, 30), "red": (180, 40, 40), "blue": (40, 70, 160),
}


def extract_vehicle_color(frame, box):
    """Samples the region above the plate (the vehicle body) and classifies
    it to the nearest named colour. This is the signal the re-ID fallback
    (reid.py) uses when OCR can't read the plate at all -- a real, if
    weak on its own, way to keep a vehicle's trajectory alive across an
    unreadable sighting instead of losing it."""
    x, y, w, h = box
    body_y0 = max(0, y - h * 3)
    body = frame[body_y0:y, x:x + w]
    if body.size == 0:
        return None
    avg_bgr = body.reshape(-1, 3).mean(axis=0)
    avg_rgb = (avg_bgr[2], avg_bgr[1], avg_bgr[0])
    best = min(NAMED_COLORS, key=lambda name: sum((a - b) ** 2 for a, b in zip(avg_rgb, NAMED_COLORS[name])))
    return best


def ocr_crop(frame, box):
    """Real OCR: crop, upscale, threshold, run Tesseract, return (text, confidence)."""
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return "", 0.0

    crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

    text = "".join(t for t in data["text"] if t.strip()).upper().replace(" ", "")
    confs = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    confidence = (sum(confs) / len(confs) / 100) if confs else 0.0
    return text, round(confidence, 3)


def insert_detection(plate, confidence, camera_id, timestamp=None, vehicle_color=None, vehicle_type=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO detections
           (plate, true_plate, confidence, camera_id, vehicle_color, vehicle_type, timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (plate, plate, confidence, camera_id, vehicle_color, vehicle_type,
         (timestamp or datetime.now()).isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def process_video(path, camera_id="CAM-DEMO", annotate_path=None, sample_every=3,
                   store_to_db=False, min_confidence=0.2):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    writer = None
    if annotate_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(annotate_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    results = []
    seen_recent = {}  # plate -> last frame index, to avoid spamming duplicate reads
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every == 0:
            for box in find_plate_candidates(frame):
                text, conf = ocr_crop(frame, box)
                accepted_text = None

                if PLATE_RE.match(text) and conf >= min_confidence:
                    accepted_text = text
                    last_seen = seen_recent.get(text, -999)
                    if frame_idx - last_seen > int(fps):  # dedupe within ~1s
                        seen_recent[text] = frame_idx
                        results.append({"frame": frame_idx, "plate": text, "confidence": conf, "box": box})
                        if store_to_db:
                            insert_detection(text, conf, camera_id)
                else:
                    # OCR failed or came back low-confidence, but we still
                    # localized a plate-shaped region -- fall back to the
                    # vehicle's colour so re-ID can keep the trajectory
                    # alive instead of losing this sighting entirely.
                    color = extract_vehicle_color(frame, box)
                    last_seen = seen_recent.get("UNREADABLE", -999)
                    if color and frame_idx - last_seen > int(fps) * 2:
                        seen_recent["UNREADABLE"] = frame_idx
                        results.append({"frame": frame_idx, "plate": "UNREADABLE",
                                         "confidence": conf, "box": box, "vehicle_color": color})
                        if store_to_db:
                            insert_detection("UNREADABLE", conf, camera_id, vehicle_color=color)

                if writer:
                    x, y, w, h = box
                    color_box = (0, 200, 255) if accepted_text else (100, 100, 100)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, 2)
                    label = f"{accepted_text} {conf:.2f}" if accepted_text else "unreadable"
                    cv2.putText(frame, label, (x, max(y - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--camera", default="CAM-DEMO")
    ap.add_argument("--annotate", default=None, help="path to write an annotated output video")
    ap.add_argument("--store", action="store_true", help="insert real reads into anpr.db")
    args = ap.parse_args()

    results = process_video(args.video, camera_id=args.camera,
                             annotate_path=args.annotate, store_to_db=args.store)

    print(f"\n{len(results)} plate reads accepted (confidence-filtered, deduplicated):\n")
    for r in results:
        print(f"  frame {r['frame']:>4}  plate={r['plate']:<12}  confidence={r['confidence']:.2f}")

    if args.store:
        print(f"\nInserted into {DB_PATH} -- open the dashboard, these show up in the live analytics.")
    if args.annotate:
        print(f"Annotated video written to {args.annotate}")


if __name__ == "__main__":
    main()

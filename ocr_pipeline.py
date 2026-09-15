"""
Real detection pipeline — SWAP-IN POINT for when you have actual camera
footage instead of the synthetic seed data in app.py.

This file is intentionally not wired into app.py yet. The idea is:
run this offline (or as a background worker per camera) against real
video, and have it INSERT rows into the same `detections` table that
app.py already reads from. Nothing in the API, frontend, trajectory
engine, or analytics needs to change — they only care about rows in
that table, not where they came from.

Pipeline stages:
    1. Plate localization  -> YOLOv8 (pretrained on a license-plate dataset)
    2. Plate crop + OCR     -> PaddleOCR / EasyOCR
    3. Confidence + fuzzy correction -> reuse noisy_ocr_read()'s inverse:
       here you'd apply a plate-format regex (e.g. Indian format
       AA-00-AA-0000) to auto-correct obviously wrong characters
       (letter position -> can't be a digit-only OCR confusion, etc.)
    4. Vehicle attribute fallback -> a color/type classifier (e.g. a small
       CNN or even a CLIP zero-shot classifier) runs on the same crop
       whenever OCR confidence is below your threshold, so the vehicle
       can still be re-identified across cameras.

Install (only when you actually have footage to run this on):
    pip install ultralytics paddleocr opencv-python-headless --break-system-packages

Example skeleton (fill in with your trained/downloaded weights):
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "anpr.db"

INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$")


def insert_detection(plate: str, confidence: float, camera_id: str,
                      vehicle_color: str = None, vehicle_type: str = None):
    """Call this from your real pipeline for every accepted read."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO detections
           (plate, true_plate, confidence, camera_id, vehicle_color, vehicle_type, timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (plate, plate, confidence, camera_id, vehicle_color, vehicle_type,
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def looks_like_valid_plate(plate: str) -> bool:
    """Cheap sanity filter before trusting an OCR read enough to store it
    as high-confidence — reject strings that can't be a real plate format."""
    return bool(INDIAN_PLATE_RE.match(plate))


def process_frame(frame, camera_id: str, plate_detector, ocr_engine, attribute_classifier=None):
    """
    Pseudocode for one video frame — fill in with your actual model calls.

        boxes = plate_detector(frame)                     # YOLOv8 inference
        for box in boxes:
            crop = frame[box.y1:box.y2, box.x1:box.x2]
            text, ocr_confidence = ocr_engine(crop)        # PaddleOCR/EasyOCR

            if ocr_confidence < 0.5 or not looks_like_valid_plate(text):
                # Fall back to vehicle re-identification instead of
                # dropping the detection entirely.
                color, vtype = attribute_classifier(frame, box) if attribute_classifier else (None, None)
                insert_detection(text or "UNREADABLE", ocr_confidence, camera_id, color, vtype)
            else:
                insert_detection(text, ocr_confidence, camera_id)
    """
    raise NotImplementedError("Fill this in once you have real video + model weights.")

"""
Generates a short synthetic traffic video so we have real video frames to
run a real detection+OCR pipeline against. This is NOT a substitute for
actual camera footage in your final demo — swap this input for a real
clip (phone-recorded traffic, a dashcam video, a Kaggle ANPR dataset clip)
the moment you have one. Everything downstream (detect_plates.py) treats
video frames as video frames; it doesn't know or care that these were
rendered rather than filmed.

Three vehicles cross the frame under three difficulty conditions, mirroring
the exact conditions the SIH problem statement calls out:
  - vehicle 1: clean, front-on, static plate      (easy case)
  - vehicle 2: angled plate                        (angled shot)
  - vehicle 3: motion-blurred, fast-moving plate    (motion blur)
"""

import cv2
import numpy as np

WIDTH, HEIGHT = 960, 540
FPS = 24
OUT_PATH = "sample_traffic.mp4"

SCENES = [
    {"plate": "KA05MJ4218", "y": 160, "speed": 4,  "angle": 0,  "blur": 0, "frames": 90},
    {"plate": "KA51YL5994", "y": 300, "speed": 5,  "angle": 18, "blur": 0, "frames": 90},
    {"plate": "KA03RK6689", "y": 420, "speed": 14, "angle": 4,  "blur": 9, "frames": 60},
]


def draw_vehicle(frame, x, y, plate_text, angle=0, color=(90, 90, 90)):
    """Draws a simple car silhouette with a rectangular plate + text on it."""
    car_w, car_h = 160, 70
    plate_w, plate_h = 100, 28

    car_layer = np.zeros((car_h + 40, car_w + 40, 3), dtype=np.uint8)
    cx, cy = car_layer.shape[1] // 2, car_layer.shape[0] // 2

    cv2.rectangle(car_layer, (cx - car_w // 2, cy - car_h // 2),
                  (cx + car_w // 2, cy + car_h // 2), color, -1, cv2.LINE_AA)
    cv2.rectangle(car_layer, (cx - car_w // 2, cy - car_h // 2),
                  (cx + car_w // 2, cy + car_h // 2), (30, 30, 30), 2, cv2.LINE_AA)

    px0, py0 = cx - plate_w // 2, cy + car_h // 2 - plate_h - 6
    cv2.rectangle(car_layer, (px0, py0), (px0 + plate_w, py0 + plate_h), (235, 235, 235), -1)
    cv2.rectangle(car_layer, (px0, py0), (px0 + plate_w, py0 + plate_h), (20, 20, 20), 1)
    cv2.putText(car_layer, plate_text, (px0 + 4, py0 + plate_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1, cv2.LINE_AA)

    if angle != 0:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        car_layer = cv2.warpAffine(car_layer, M, (car_layer.shape[1], car_layer.shape[0]))

    h, w = car_layer.shape[:2]
    x0, y0 = int(x - w // 2), int(y - h // 2)
    x1, y1 = x0 + w, y0 + h
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return
    cx0, cy0 = fx0 - x0, fy0 - y0
    cx1, cy1 = cx0 + (fx1 - fx0), cy0 + (fy1 - fy0)

    roi = frame[fy0:fy1, fx0:fx1]
    patch = car_layer[cy0:cy1, cx0:cx1]
    mask = np.any(patch > 15, axis=2)
    roi[mask] = patch[mask]


def generate():
    writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    total_frames = max(s["frames"] for s in SCENES)

    for f in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), (58, 58, 58), dtype=np.uint8)
        cv2.rectangle(frame, (0, HEIGHT // 2 + 60), (WIDTH, HEIGHT), (40, 40, 40), -1)
        for lane_y in (HEIGHT // 2 + 60, HEIGHT // 2 + 200):
            for lx in range(0, WIDTH, 40):
                cv2.line(frame, (lx, lane_y), (lx + 20, lane_y), (200, 200, 0), 2)

        for scene in SCENES:
            if f >= scene["frames"]:
                continue
            x = 60 + f * scene["speed"]
            draw_vehicle(frame, x, scene["y"], scene["plate"], angle=scene["angle"])

        for scene in SCENES:
            if scene["blur"] > 0 and f < scene["frames"]:
                frame = cv2.GaussianBlur(frame, (scene["blur"] | 1, scene["blur"] | 1), 0)
                break

        writer.write(frame)

    writer.release()
    print(f"Wrote {OUT_PATH} ({total_frames} frames, {WIDTH}x{HEIGHT} @ {FPS}fps)")
    def label(s):
        tags = []
        if s["angle"]:
            tags.append("angled")
        if s["blur"]:
            tags.append("blurred")
        return ", ".join(tags) if tags else "clean"

    print("Scenes:", [(s["plate"], label(s)) for s in SCENES])


if __name__ == "__main__":
    generate()

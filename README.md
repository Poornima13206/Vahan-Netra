# Vahan Netra — City-Wide ANPR Trajectory Tracking & Traffic Analytics

A working prototype built for the Smart India Hackathon problem statement:
**"City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban
Traffic Analytics."** This links isolated per-camera plate detection into
one centralized system that tracks vehicle trajectories across a city's
entire camera network, and layers real analytics (origin-destination
flow, route anomaly detection, predictive movement) on top of it.

---

## 1. What this actually does

| Problem statement requirement | What's built |
|---|---|
| High-accuracy OCR engine (>90% target) | Real OCR pipeline (OpenCV + Tesseract) — works well on clean plates, honestly does **not** yet hit 90% on angled/blurred plates (see Limitations). Fuzzy matching compensates at the search layer. |
| Single plate trajectory tracking | Fully working — search any plate, see its complete route across every camera, animated on a live GIS map with timestamps |
| Macro traffic flow & movement analytics | Heatmap, Origin-Destination flow map, and statistical route anomaly detection |
| Alert system (blacklist + anomalies) | Real-time blacklist alerts via WebSocket + automated anomaly flagging |

## What's simulated

**Beyond the problem statement** (the project's actual differentiators):
- Vehicle Re-ID fallback for plates that are too dirty/damaged/blurred to read at all
- Predictive next-camera / ETA estimation from historical traffic patterns
- Real video upload — run the actual detection pipeline on your own footage, not just simulated data

---

## 2. Tech stack

**Backend:** Python, FastAPI (REST + WebSocket), SQLite
**Computer vision / OCR:** OpenCV (plate localization), Tesseract OCR (text extraction)
**Frontend:** Vanilla HTML/CSS/JavaScript (no build step), Leaflet.js on OpenStreetMap tiles
**No paid APIs anywhere** — everything runs locally, no vendor lock-in, no API keys needed

---

## 3. Project structure

```
vahan-netra/
├── README.md                    (this file)
├── backend/
│   ├── app.py                   Main FastAPI server — all API endpoints, DB setup, seed data, live feed
│   ├── anomaly.py                Route anomaly detection (impossible speed / unusual timing / rare route)
│   ├── reid.py                   Vehicle re-identification fallback for unreadable plates
│   ├── predict.py                Predictive next-camera / ETA estimation
│   ├── detect_plates.py          Real CV+OCR pipeline — runs on actual video (sample or uploaded)
│   ├── generate_sample_video.py  Generates a synthetic test video (no real footage available in dev)
│   ├── ocr_pipeline.py           Documentation/stub showing where a trained YOLOv8 detector would plug in
│   ├── requirements.txt
│   └── sample_traffic.mp4        Pre-generated test clip
└── frontend/
    └── index.html                 The entire dashboard — map, search, analytics, alerts, video upload
```

---

## 4. How to run it

### Step 1 — Install Tesseract (the OCR engine itself, not just the Python wrapper)
- **Linux:** `apt-get install tesseract-ocr`
- **Mac:** `brew install tesseract`
- **Windows:** install from https://github.com/UB-Mannheim/tesseract/wiki — if `pytesseract` can't find it afterward, add this near the top of `detect_plates.py`:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

### Step 2 — Install Python dependencies
```bash
cd backend
pip install -r requirements.txt --break-system-packages
```
(Drop `--break-system-packages` on Windows or inside a virtual environment.)

### Step 3 — Run the backend
```bash
uvicorn app:app --port 8000
```
Don't use `--reload` during an actual demo — it watches your whole project folder (including `venv/`, which has thousands of files) and will randomly restart the server mid-demo, killing the WebSocket connection. Only use `--reload` while actively editing code, and exclude your venv if you do: `uvicorn app:app --reload --port 8000 --reload-exclude "venv/*"`.

The first run auto-creates `anpr.db` with realistic seed data (commuter corridors, a few deliberately injected anomalies and re-ID cases). Delete this file any time to reseed fresh.

### Step 4 — Open the frontend
Use VS Code's **Live Server** extension (right-click `frontend/index.html` → "Open with Live Server") rather than double-clicking the file directly — opening it as a raw `file://` URL can cause the browser to block the API/WebSocket calls.

### Step 5 — Verify it's working
Check the top-right of the dashboard says "live feed connected" with a glowing cyan dot.

---

## 5. Full feature list

### Plate search (fuzzy matching)
Search any plate string. Uses edit-distance ranking, so a misread character (0/O, 1/I, 8/B, 5/S — the exact confusions real OCR makes) still surfaces the correct vehicle, ranked by how close the match is and how many times it's been seen.

### Trajectory reconstruction
Click a search result to see that vehicle's full route: every camera it hit, in time order, animated on the map with a moving marker. Each stop shows OCR confidence and the vehicle's recorded color/type.

### Live camera feed (simulated)
A WebSocket connection pushes a new fake detection every ~3 seconds, so the dashboard feels alive for demo purposes. **This is simulated, not real camera data** — say this plainly if asked. The real detection pipeline is the video upload feature below.

### Analyze your own footage (real, not simulated)
Upload any video in the left sidebar. This runs the actual CV+OCR pipeline (`detect_plates.py`) frame by frame — real plate localization, real OCR, real confidence scoring. Results get stored and immediately show up everywhere else in the dashboard (search, analytics, map). You also get a downloadable annotated video showing the detection boxes.

### City Snapshot analytics
Total detections, unique vehicles, average OCR confidence, and active alert count, aggregated live from the database.

### Heatmap
Per-camera detection density, shown as glowing circles sized by traffic volume.

### Top Routes (Origin-Destination flow)
Ranked list + weighted flow-lines on the map showing which camera-to-camera corridors carry the most traffic. This is the "origin-destination patterns" macro-analytics requirement from the problem statement.

### Route Anomaly Detection
Flags three kinds of suspicious routes, built on a baseline of normal travel times computed from all vehicles:
- **Impossible speed** (red) — implied speed between two cameras exceeds ~120 km/h
- **Unusual timing** (amber) — a statistical outlier versus how long that specific route normally takes
- **Rare route** (grey) — a camera-to-camera hop almost nobody else ever takes

Click a card to jump straight to that vehicle's trajectory.

### Vehicle Re-ID fallback
When a plate genuinely can't be read, the pipeline still extracts the vehicle's color from the body above the plate and stores it as an "UNREADABLE" detection instead of discarding the sighting. For any known vehicle, the system checks whether a same-colored unreadable sighting at a nearby camera, at a physically plausible speed, could be a continuation of its route — shown as a **ranked, scored candidate for human review, never auto-confirmed**.

### Predictive next-camera / ETA
For a vehicle's last known camera, looks at everyone else historically seen there and where they went next, turning that into a probability distribution plus an ETA. Shown as a violet dotted projection on the map. This is a population-level pattern ("vehicles here usually go there"), not a guarantee for one specific vehicle — the UI says this explicitly.

### Blacklist alerts
Add any plate to the blacklist; the moment it's seen again (including via the live simulated feed), an alert fires instantly in the sidebar.

---
## 6. Research & references

- Problem domain research: real-world ANPR/OCR accuracy under varying lighting, angle, motion blur, and plate degradation conditions
- Open-source tools: OpenCV documentation, Tesseract OCR, Ultralytics YOLOv8 (planned upgrade), Leaflet.js / OpenStreetMap
- Public datasets considered for future model training: Kaggle Indian vehicle number-plate datasets, CCPD (Chinese City Parking Dataset) as a methodology reference
- Policy reference: Digital Personal Data Protection Act, 2023 (India) — informs the proposed privacy-by-design architecture


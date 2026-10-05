<div align="center">

# 🚦 Vahan Netra
### City-Wide ANPR Trajectory Tracking & Urban Traffic Analytics

**Smart India Hackathon — Problem Statement:** *City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking and Urban Traffic Analytics*

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-setup--installation) • [Usage](#-usage) • [API Reference](#-api-reference) • [Limitations](#-known-limitations) • [Roadmap](#-roadmap)

</div>

---

## 📌 Overview

Modern cities run large networks of ANPR (Automatic Number Plate Recognition) cameras, but most deployments process each camera's feed in isolation — a plate is detected, logged, and forgotten. There is no system that links sightings across cameras into a usable trajectory, and no layer of analysis on top of the raw detections.

**Vahan Netra** is a centralized platform that solves this by treating every camera as a contributor to one shared, queryable dataset. On top of that shared dataset, it adds the intelligence layer that turns raw detections into operational insight: trajectory reconstruction, fuzzy OCR-error correction, vehicle re-identification for unreadable plates, origin-destination traffic flow analysis, automated anomaly detection, and predictive movement estimation.

---

## ✅ Problem Statement Coverage

| Requirement | Status | Notes |
|---|:---:|---|
| High-accuracy OCR engine (>90% target) | 🟡 Partial | Real OCR pipeline implemented (OpenCV + Tesseract); performs well on clear plates, does not yet meet the 90% target under angled/blurred conditions. See [Known Limitations](#-known-limitations). |
| Single plate trajectory tracking | ✅ Complete | Query-based tracking across the full camera network, plotted on a live GIS map with timestamps and routes |
| Macro traffic flow & movement analytics | ✅ Complete | Density heatmaps, origin-destination flow mapping, statistical anomaly detection |
| Alert system (blacklist + anomalies) | ✅ Complete | Real-time blacklist alerts and automated route-anomaly flagging |

**Additional capabilities beyond the problem statement:**
- Vehicle re-identification fallback for plates too degraded to read
- Predictive next-camera and ETA estimation from historical traffic patterns
- Live video analysis — the detection pipeline runs on actual uploaded footage, not only on simulated data

---

## ✨ Features

### Core (problem statement requirements)
- **Fuzzy plate search** — edit-distance ranked matching corrects common OCR misreads (0/O, 1/I, 8/B, 5/S) so a degraded read still resolves to the correct vehicle
- **Trajectory reconstruction** — full route history for any plate, animated on an interactive map with per-stop confidence and timestamps
- **Traffic density heatmap** — live, camera-wise detection density
- **Origin-destination flow analysis** — ranked, weighted map of the busiest camera-to-camera corridors city-wide
- **Route anomaly detection** — flags physically impossible speeds, statistically unusual travel times, and rare routes, each against a baseline built from citywide traffic
- **Real-time alerting** — instant notification when a blacklisted plate is detected

### Differentiators
- **Vehicle re-identification fallback** — when OCR fails entirely, the system falls back to color/type matching against known vehicles' last positions, surfaced as scored candidates for human review
- **Predictive next-camera / ETA** — forecasts a vehicle's likely next location and arrival time from historical movement patterns
- **Live footage analysis** — upload real video and run the actual computer-vision pipeline against it, with annotated output

---

## 🏗 Architecture

```
┌─────────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│   Camera Feed     │ ──▶  │  Detection Pipeline   │ ──▶  │  Central Database   │
│ (video / upload)  │      │ (OpenCV + Tesseract)  │      │     (SQLite)        │
└─────────────────┘      └──────────────────────┘      └──────────┬─────────┘
                                                                     │
                      ┌──────────────────────────────────────────────┤
                      ▼                     ▼                     ▼
             ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
             │ Trajectory Engine │  │ Analytics Engine  │  │   Alert System    │
             │ (fuzzy match,     │  │ (heatmap, OD flow, │  │ (blacklist, live   │
             │  re-ID, predict)  │  │  anomaly detection)│  │  WebSocket push)   │
             └─────────────────┘  └──────────────────┘  └──────────────────┘
                      │                     │                     │
                      └──────────────┬──────────────────────────────┘
                                      ▼
                          ┌───────────────────────┐
                          │   GIS Dashboard (UI)    │
                          │  Leaflet + OpenStreetMap │
                          └───────────────────────┘
```

Every camera submits detections to one shared database rather than communicating with other cameras directly. This is what allows "linking data across space and time" — once every camera's output lands in the same table, trajectory reconstruction becomes a query, not a distributed systems problem. This also makes the architecture realistically deployable: it works over a standard network connection to existing camera infrastructure, with no new hardware or inter-camera protocol required.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI (REST + WebSocket) |
| Database | SQLite |
| Computer Vision | OpenCV |
| OCR | Tesseract OCR |
| Frontend | HTML / CSS / JavaScript (no build step) |
| Mapping | Leaflet.js + OpenStreetMap |

No paid APIs or external services are used anywhere in the stack. The entire system runs locally with no vendor dependency or licensing cost.

---

## 📁 Project Structure

```
vahan-netra/
├── README.md
├── backend/
│   ├── app.py                    # FastAPI server — API endpoints, database, live feed
│   ├── anomaly.py                # Route anomaly detection engine
│   ├── reid.py                   # Vehicle re-identification fallback
│   ├── predict.py                # Predictive next-camera / ETA estimation
│   ├── detect_plates.py          # Computer vision + OCR detection pipeline
│   ├── generate_sample_video.py  # Synthetic test video generator
│   ├── ocr_pipeline.py           # Reference implementation notes for a production-grade detector
│   ├── requirements.txt
│   └── sample_traffic.mp4
└── frontend/
    └── index.html                # Dashboard: map, search, analytics, alerts, video upload
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- Tesseract OCR (system package, not a Python library)

**Install Tesseract:**
```bash
# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer: https://github.com/UB-Mannheim/tesseract/wiki
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --port 8000
```

The first run automatically initializes `anpr.db` with seed data modeling realistic commuter traffic patterns. Delete this file at any time to regenerate fresh data.

> **Note:** avoid `uvicorn --reload` during a live demo — it watches the entire project directory including any virtual environment folder, which can trigger unwanted restarts. Use it only during active development, ideally with `--reload-exclude "venv/*"`.

### Frontend
Serve `frontend/index.html` with a local web server (e.g. VS Code's Live Server extension) rather than opening it directly as a file — this avoids browser restrictions on local API/WebSocket calls.

---

## 📖 Usage

1. **Search a plate** — type a plate number; fuzzy matching returns ranked candidates even for imperfect reads
2. **View a trajectory** — select a result to see its full route animate across the map, including any re-identification candidates or next-location predictions
3. **Monitor live analytics** — the sidebar shows real-time detection counts, the traffic heatmap, top routes, and active anomalies
4. **Analyze your own footage** — upload a video file to run real plate detection on it; results integrate immediately into the dashboard
5. **Manage the blacklist** — flagged plates trigger instant alerts when detected

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service health check |
| GET | `/api/cameras` | List all camera nodes |
| GET | `/api/search?q={plate}` | Fuzzy plate search |
| GET | `/api/trajectory/{plate}` | Full trajectory, re-ID candidates, and prediction for a plate |
| GET | `/api/analytics/summary` | City-wide summary statistics |
| GET | `/api/analytics/heatmap` | Per-camera detection density |
| GET | `/api/analytics/anomalies` | Flagged route anomalies |
| GET | `/api/analytics/od-matrix` | Origin-destination traffic flow |
| GET | `/api/alerts` | Recent blacklist alerts |
| GET / POST | `/api/blacklist` | View or add blacklisted plates |
| POST | `/api/upload-video` | Upload and analyze real footage |
| WS | `/ws/live` | Real-time detection stream |

Interactive API documentation is available at `/docs` once the server is running.

---

## ⚠️ Known Limitations

- **OCR accuracy** does not yet meet the problem statement's 90% target under adverse conditions (angled shots, motion blur). The current implementation uses classical computer vision for plate localization; closing this gap requires a trained deep-learning detector (e.g. a fine-tuned YOLOv8 model), which the architecture is designed to accommodate without requiring changes elsewhere in the system.
- **The live detection feed in the default view is simulated** for demonstration purposes. The video upload feature runs genuine computer vision and OCR on real footage.
- **Anomaly detection and predictive estimation require sufficient historical traffic data** on a given route before producing meaningful output — this is expected statistical behavior, not a defect.
- **Access control and audit logging are not yet implemented.** A production deployment handling law-enforcement data would require role-based access control and a defined data retention policy.

---

## 🗺 Roadmap

- [ ] Trained deep-learning plate detector (fine-tuned YOLOv8) to close the OCR accuracy gap
- [ ] Congestion and bottleneck detection using existing density data
- [ ] Role-based access control and audit logging
- [ ] Privacy-preserving edge architecture (plate hashing instead of raw video transmission), aligned with India's Digital Personal Data Protection Act, 2023
- [ ] Extended plate format support (commercial, two-wheeler, temporary plates)

---

## 📚 References

- OpenCV Documentation — https://docs.opencv.org
- Tesseract OCR — https://github.com/tesseract-ocr/tesseract
- Ultralytics YOLOv8 — https://docs.ultralytics.com
- Leaflet.js / OpenStreetMap — https://leafletjs.com
- Digital Personal Data Protection Act, 2023 (India)

---




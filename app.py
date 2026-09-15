"""
City-Wide ANPR Trajectory Tracking & Traffic Analytics — backend API.

This is a working PROTOTYPE backend. It ships with a synthetic camera
network + synthetic detection events so the whole system (search,
trajectory reconstruction, analytics, alerts, live feed) can be
demoed end-to-end without needing real camera hardware.

To go from prototype -> real deployment, only one thing changes:
instead of `generate_synthetic_detections()` populating the `detections`
table, a real pipeline (see ocr_pipeline.py) reading live RTSP camera
feeds would INSERT into the same table. Nothing else in this file
needs to change — that's the point of separating detection from
tracking/analytics.

Run:
    pip install -r requirements.txt --break-system-packages
    uvicorn app:app --reload --port 8000
"""

import asyncio
import json
import random
import shutil
import sqlite3
import string
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from anomaly import detect_anomalies
from reid import find_reid_candidates
import detect_plates

DB_PATH = Path(__file__).parent / "anpr.db"
UPLOADS_DIR = Path(__file__).parent / "uploaded_videos"
RESULTS_DIR = Path(__file__).parent / "processed_videos"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Camera network — real Bengaluru locations so the map looks grounded.
# In production these rows come from your camera inventory / GIS system.
# ---------------------------------------------------------------------------
CAMERAS = [
    {"id": "CAM-01", "name": "Silk Board Junction",     "road": "Hosur Rd",        "lat": 12.9172, "lng": 77.6228},
    {"id": "CAM-02", "name": "Electronic City Toll",    "road": "NH44",            "lat": 12.8452, "lng": 77.6602},
    {"id": "CAM-03", "name": "Marathahalli Bridge",     "road": "Outer Ring Rd",   "lat": 12.9569, "lng": 77.7011},
    {"id": "CAM-04", "name": "Hebbal Flyover",          "road": "Bellary Rd",      "lat": 13.0357, "lng": 77.5970},
    {"id": "CAM-05", "name": "MG Road Metro",           "road": "MG Road",         "lat": 12.9758, "lng": 77.6045},
    {"id": "CAM-06", "name": "Whitefield Main Rd",      "road": "ITPL Rd",         "lat": 12.9698, "lng": 77.7500},
    {"id": "CAM-07", "name": "Yeshwanthpur Junction",   "road": "Tumkur Rd",       "lat": 13.0284, "lng": 77.5544},
    {"id": "CAM-08", "name": "Banashankari Bus Stand",  "road": "Kanakapura Rd",   "lat": 12.9255, "lng": 77.5468},
    {"id": "CAM-09", "name": "Koramangala Sony Signal", "road": "Hosur Rd",        "lat": 12.9352, "lng": 77.6245},
    {"id": "CAM-10", "name": "KR Puram Bridge",         "road": "Old Madras Rd",   "lat": 13.0086, "lng": 77.6960},
]

VEHICLE_COLORS = ["white", "silver", "black", "red", "blue", "grey"]
VEHICLE_TYPES = ["sedan", "hatchback", "SUV", "two-wheeler", "auto-rickshaw", "truck"]
KA_DISTRICT_CODES = ["01", "02", "03", "04", "05", "41", "51"]


def random_plate() -> str:
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=4))
    return f"KA{random.choice(KA_DISTRICT_CODES)}{letters}{digits}"


def noisy_ocr_read(plate: str, confidence: float) -> str:
    """Simulate realistic OCR character confusion on low-confidence reads.

    This is what makes the fuzzy-matching / re-ID story real in the demo
    instead of hand-waved: some detections in the seed data are
    deliberately corrupted the way a real OCR engine misreads plates
    under motion blur, dirt, or bad angles (O<->0, I<->1, B<->8, S<->5).
    """
    if confidence > 0.82:
        return plate
    confusions = {"O": "0", "0": "O", "I": "1", "1": "I", "B": "8", "8": "B", "S": "5", "5": "S"}
    chars = list(plate)
    n_swaps = 1 if confidence > 0.6 else 2
    idxs = random.sample(range(len(chars)), k=min(n_swaps, len(chars)))
    for i in idxs:
        if chars[i] in confusions:
            chars[i] = confusions[chars[i]]
    return "".join(chars)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# DB setup + synthetic seeding
# ---------------------------------------------------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY, name TEXT, road TEXT, lat REAL, lng REAL
        );
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            true_plate TEXT NOT NULL,
            confidence REAL NOT NULL,
            camera_id TEXT NOT NULL,
            vehicle_color TEXT,
            vehicle_type TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (camera_id) REFERENCES cameras(id)
        );
        CREATE TABLE IF NOT EXISTS blacklist (
            plate TEXT PRIMARY KEY, reason TEXT, added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT, camera_id TEXT, timestamp TEXT, reason TEXT
        );
        """
    )
    conn.commit()

    if conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO cameras (id, name, road, lat, lng) VALUES (:id,:name,:road,:lat,:lng)",
            CAMERAS,
        )
        conn.commit()

    if conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0:
        seed_detections(conn)

    if conn.execute("SELECT COUNT(*) FROM blacklist").fetchone()[0] == 0:
        seed_blacklist(conn)

    conn.close()


# A handful of realistic commuter corridors, each with a typical (mean, std)
# travel time in minutes per hop. Most synthetic vehicles follow one of
# these -- this is what gives the anomaly detector an actual baseline of
# "normal" to compare against, instead of every route being unique.
CAM = {c["id"]: c for c in CAMERAS}
COMMON_ROUTES = [
    {"path": ["CAM-02", "CAM-01", "CAM-09", "CAM-05"], "hops": [(22, 4), (9, 2), (7, 2)]},   # EC -> Silk Board -> Koramangala -> MG Road
    {"path": ["CAM-03", "CAM-10", "CAM-04"],           "hops": [(14, 3), (18, 4)]},           # Marathahalli -> KR Puram -> Hebbal
    {"path": ["CAM-07", "CAM-04", "CAM-05"],           "hops": [(16, 3), (20, 4)]},           # Yeshwanthpur -> Hebbal -> MG Road
    {"path": ["CAM-08", "CAM-01", "CAM-09"],           "hops": [(19, 4), (8, 2)]},            # Banashankari -> Silk Board -> Koramangala
]


def _emit_route(rows, plate, cam_ids, hop_seconds, start_time, vehicle_color, vehicle_type):
    t = start_time
    for i, cam_id in enumerate(cam_ids):
        if i > 0:
            t += timedelta(seconds=hop_seconds[i - 1])
        confidence = round(random.uniform(0.55, 0.99), 2)
        observed_plate = noisy_ocr_read(plate, confidence)
        rows.append((
            observed_plate, plate, confidence, cam_id,
            vehicle_color, vehicle_type,
            t.isoformat(timespec="seconds"),
        ))


def seed_detections(conn, n_vehicles: int = 45, common_route_share: float = 0.75):
    """Generate a realistic mix of trajectories: most vehicles follow one of
    a few common corridors (normal baseline traffic), a minority take
    genuinely one-off routes, and a couple of anomalies/re-ID cases are
    deliberately injected so those detectors always have clear, demoable
    cases regardless of what randomness happens to produce."""
    now = datetime.now()
    rows = []
    n_common = int(n_vehicles * common_route_share)

    for _ in range(n_common):
        plate = random_plate()
        route = random.choice(COMMON_ROUTES)
        hop_seconds = [max(60, random.gauss(mean * 60, std * 60)) for mean, std in route["hops"]]
        start = now - timedelta(hours=random.uniform(0.5, 6))
        color, vtype = random.choice(VEHICLE_COLORS), random.choice(VEHICLE_TYPES)
        _emit_route(rows, plate, route["path"], hop_seconds, start, color, vtype)

    for _ in range(n_vehicles - n_common):
        plate = random_plate()
        route_len = random.randint(2, 5)
        cam_ids = [c["id"] for c in random.sample(CAMERAS, k=min(route_len, len(CAMERAS)))]
        hop_seconds = [random.uniform(4 * 60, 40 * 60) for _ in range(len(cam_ids) - 1)]
        start = now - timedelta(hours=random.uniform(0.5, 6))
        color, vtype = random.choice(VEHICLE_COLORS), random.choice(VEHICLE_TYPES)
        _emit_route(rows, plate, cam_ids, hop_seconds, start, color, vtype)

    # --- deliberately injected, guaranteed-demoable anomalies ---
    # 1) Unusual timing: a normal corridor, but one hop takes far longer than usual.
    route = COMMON_ROUTES[0]
    hop_seconds = [max(60, random.gauss(mean * 60, std * 60)) for mean, std in route["hops"]]
    hop_seconds[0] = route["hops"][0][0] * 60 * 4  # 4x the usual time on the first hop
    _emit_route(rows, random_plate(), route["path"], hop_seconds,
                now - timedelta(hours=1), random.choice(VEHICLE_COLORS), random.choice(VEHICLE_TYPES))

    # 2) Impossible speed: two cameras ~20km apart, crossed in 4 minutes (~300 km/h implied).
    _emit_route(rows, random_plate(), ["CAM-02", "CAM-04"], [4 * 60],
                now - timedelta(hours=2), random.choice(VEHICLE_COLORS), random.choice(VEHICLE_TYPES))

    # 3) Re-ID case: a vehicle on a common corridor whose plate is dirty/unreadable
    # at the LAST camera, but the vehicle's colour carries through -- this is
    # exactly what the re-ID fallback should catch and surface as a candidate.
    route = COMMON_ROUTES[1]
    hop_seconds = [max(60, random.gauss(mean * 60, std * 60)) for mean, std in route["hops"]]
    reid_plate = random_plate()
    reid_color = random.choice(VEHICLE_COLORS)
    reid_type = random.choice(VEHICLE_TYPES)
    start = now - timedelta(hours=1.5)
    _emit_route(rows, reid_plate, route["path"][:-1], hop_seconds[:-1], start, reid_color, reid_type)
    # last hop: plate comes back unreadable, but same colour/type, plausible timing
    last_time = start + timedelta(seconds=sum(hop_seconds[:-1]) + hop_seconds[-1])
    rows.append((
        "UNREADABLE", "UNREADABLE", 0.25, route["path"][-1],
        reid_color, reid_type, last_time.isoformat(timespec="seconds"),
    ))

    conn.executemany(
        """INSERT INTO detections
           (plate, true_plate, confidence, camera_id, vehicle_color, vehicle_type, timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return [r[1] for r in rows]  # true plates, for blacklist seeding


def seed_blacklist(conn):
    true_plates = [r[0] for r in conn.execute("SELECT DISTINCT true_plate FROM detections").fetchall()]
    if not true_plates:
        return
    flagged = random.sample(true_plates, k=min(3, len(true_plates)))
    reasons = ["Reported stolen", "Pending e-challan > 90 days", "Flagged in active FIR"]
    for plate, reason in zip(flagged, reasons):
        conn.execute(
            "INSERT INTO blacklist (plate, reason, added_at) VALUES (?,?,?)",
            (plate, reason, datetime.now().isoformat(timespec="seconds")),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# App + live feed simulation
# ---------------------------------------------------------------------------
live_clients: set[WebSocket] = set()


async def live_feed_loop():
    """Simulates cameras continuing to report new detections in real time.
    Every ~3s, generate one new detection (possibly for a blacklisted plate,
    to demo the alert system), broadcast it to connected dashboards."""
    while True:
        await asyncio.sleep(3)
        conn = get_conn()
        blacklisted = [r[0] for r in conn.execute("SELECT plate FROM blacklist").fetchall()]
        use_blacklisted = blacklisted and random.random() < 0.25
        true_plate = random.choice(blacklisted) if use_blacklisted else random_plate()
        cam = random.choice(CAMERAS)
        confidence = round(random.uniform(0.6, 0.99), 2)
        observed_plate = noisy_ocr_read(true_plate, confidence)
        ts = datetime.now().isoformat(timespec="seconds")

        cur = conn.execute(
            """INSERT INTO detections
               (plate, true_plate, confidence, camera_id, vehicle_color, vehicle_type, timestamp)
               VALUES (?,?,?,?,?,?,?)""",
            (observed_plate, true_plate, confidence, cam["id"],
             random.choice(VEHICLE_COLORS), random.choice(VEHICLE_TYPES), ts),
        )
        conn.commit()
        detection_id = cur.lastrowid

        payload = {
            "type": "detection",
            "id": detection_id,
            "plate": observed_plate,
            "confidence": confidence,
            "camera": {"id": cam["id"], "name": cam["name"], "lat": cam["lat"], "lng": cam["lng"]},
            "vehicle_color": random.choice(VEHICLE_COLORS),
            "timestamp": ts,
        }

        if use_blacklisted:
            reason = conn.execute("SELECT reason FROM blacklist WHERE plate=?", (true_plate,)).fetchone()[0]
            conn.execute(
                "INSERT INTO alerts (plate, camera_id, timestamp, reason) VALUES (?,?,?,?)",
                (true_plate, cam["id"], ts, reason),
            )
            conn.commit()
            payload["alert"] = {"plate": true_plate, "reason": reason, "camera": cam["name"]}

        conn.close()

        dead = []
        for ws in live_clients:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            live_clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    task = asyncio.create_task(live_feed_loop())
    yield
    task.cancel()


app = FastAPI(title="City ANPR Trajectory Tracking API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/videos", StaticFiles(directory=RESULTS_DIR), name="videos")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cameras")
def get_cameras():
    conn = get_conn()
    cams = [dict(r) for r in conn.execute("SELECT * FROM cameras").fetchall()]
    conn.close()
    return cams


@app.get("/api/search")
def search_plate(q: str, max_distance: int = 2):
    """Fuzzy plate search. Returns distinct observed plate strings within
    edit-distance `max_distance` of the query, ranked by distance then by
    number of detections (more sightings = more likely to be a real, not a
    one-off OCR glitch)."""
    q = q.strip().upper()
    conn = get_conn()
    rows = conn.execute(
        "SELECT plate, COUNT(*) as hits, AVG(confidence) as avg_conf "
        "FROM detections WHERE plate != 'UNREADABLE' GROUP BY plate"
    ).fetchall()
    conn.close()

    candidates = []
    for r in rows:
        dist = levenshtein(q, r["plate"])
        if dist <= max_distance:
            candidates.append(
                {
                    "plate": r["plate"],
                    "edit_distance": dist,
                    "detections": r["hits"],
                    "avg_confidence": round(r["avg_conf"], 2),
                }
            )
    candidates.sort(key=lambda c: (c["edit_distance"], -c["detections"]))
    return {"query": q, "matches": candidates[:10]}


@app.get("/api/trajectory/{plate}")
def get_trajectory(plate: str):
    """Reconstruct the full route for one plate: every detection, in time
    order, joined with camera location — this is what feeds the map replay."""
    plate = plate.strip().upper()
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT d.id, d.plate, d.confidence, d.vehicle_color, d.vehicle_type, d.timestamp,
               c.id as camera_id, c.name as camera_name, c.road, c.lat, c.lng
        FROM detections d JOIN cameras c ON d.camera_id = c.id
        WHERE d.plate = ?
        ORDER BY d.timestamp ASC
        """,
        (plate,),
    ).fetchall()
    conn.close()
    stops = [dict(r) for r in rows]
    return {
        "plate": plate,
        "stop_count": len(stops),
        "stops": stops,
        "reid_candidates": find_reid_candidates(plate) if stops else [],
    }


@app.get("/api/analytics/summary")
def analytics_summary():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    unique_vehicles = conn.execute("SELECT COUNT(DISTINCT true_plate) FROM detections").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM detections").fetchone()[0] or 0
    busiest = conn.execute(
        """SELECT c.name, COUNT(*) as hits FROM detections d
           JOIN cameras c ON d.camera_id = c.id
           GROUP BY c.id ORDER BY hits DESC LIMIT 1"""
    ).fetchone()
    active_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    conn.close()
    return {
        "total_detections": total,
        "unique_vehicles": unique_vehicles,
        "avg_ocr_confidence": round(avg_conf, 3),
        "busiest_camera": dict(busiest) if busiest else None,
        "active_alerts": active_alerts,
    }


@app.get("/api/analytics/heatmap")
def analytics_heatmap():
    """Per-camera detection density — feeds the heatmap layer on the map."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.id, c.name, c.lat, c.lng, COUNT(d.id) as weight
           FROM cameras c LEFT JOIN detections d ON d.camera_id = c.id
           GROUP BY c.id"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/analytics/anomalies")
def analytics_anomalies(limit: int = 25):
    return detect_anomalies()[:limit]


@app.get("/api/analytics/od-matrix")
def analytics_od_matrix(limit: int = 15):
    """Origin-destination flow: which camera-to-camera hops carry the most
    traffic, and how long they typically take. This is the macro
    "movement pattern" view the problem statement asks for, distinct
    from the per-vehicle trajectory tracking -- it answers "where does
    city traffic actually flow" rather than "where did this one vehicle go".
    """
    conn = get_conn()
    rows = conn.execute(
        """SELECT d.plate, d.timestamp, d.camera_id, c.lat, c.lng, c.name
           FROM detections d JOIN cameras c ON d.camera_id = c.id
           WHERE d.plate != 'UNREADABLE'
           ORDER BY d.plate, d.timestamp ASC"""
    ).fetchall()
    conn.close()

    from collections import defaultdict
    by_plate = defaultdict(list)
    for r in rows:
        by_plate[r["plate"]].append(dict(r))

    edges = defaultdict(lambda: {"count": 0, "total_minutes": 0.0})
    cam_lookup = {}
    for stops in by_plate.values():
        for a, b in zip(stops, stops[1:]):
            key = (a["camera_id"], b["camera_id"])
            ta = datetime.fromisoformat(a["timestamp"])
            tb = datetime.fromisoformat(b["timestamp"])
            minutes = (tb - ta).total_seconds() / 60
            if minutes <= 0:
                continue
            edges[key]["count"] += 1
            edges[key]["total_minutes"] += minutes
            cam_lookup[a["camera_id"]] = {"name": a["name"], "lat": a["lat"], "lng": a["lng"]}
            cam_lookup[b["camera_id"]] = {"name": b["name"], "lat": b["lat"], "lng": b["lng"]}

    results = []
    for (from_id, to_id), stats in edges.items():
        results.append({
            "from_camera_id": from_id,
            "to_camera_id": to_id,
            "from_camera": cam_lookup[from_id]["name"],
            "to_camera": cam_lookup[to_id]["name"],
            "from_lat": cam_lookup[from_id]["lat"], "from_lng": cam_lookup[from_id]["lng"],
            "to_lat": cam_lookup[to_id]["lat"], "to_lng": cam_lookup[to_id]["lng"],
            "trip_count": stats["count"],
            "avg_minutes": round(stats["total_minutes"] / stats["count"], 1),
        })
    results.sort(key=lambda r: -r["trip_count"])
    return results[:limit]


@app.get("/api/alerts")
def get_alerts(limit: int = 20):
    conn = get_conn()
    rows = conn.execute(
        """SELECT a.id, a.plate, a.timestamp, a.reason, c.name as camera_name, c.lat, c.lng
           FROM alerts a JOIN cameras c ON a.camera_id = c.id
           ORDER BY a.timestamp DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/blacklist")
def get_blacklist():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM blacklist").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/blacklist")
def add_blacklist(plate: str, reason: str = "Manually flagged"):
    plate = plate.strip().upper()
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO blacklist (plate, reason, added_at) VALUES (?,?,?)",
        (plate, reason, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return {"status": "added", "plate": plate}


@app.post("/api/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Form("CAM-UPLOAD"),
    camera_name: str = Form("Uploaded footage"),
    lat: float = Form(12.9716),
    lng: float = Form(77.5946),
    store: bool = Form(True),
):
    """Runs the REAL detection+OCR pipeline (detect_plates.py) on a
    user-uploaded video -- no synthetic/seeded data involved. Returns
    every accepted plate read plus a link to an annotated video showing
    the bounding boxes and OCR text overlaid on the original footage."""
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    job_id = uuid.uuid4().hex[:10]
    input_path = UPLOADS_DIR / f"{job_id}{suffix}"
    output_path = RESULTS_DIR / f"{job_id}_annotated.mp4"

    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Make sure this camera exists so results plot on the map / show up
    # in the dashboard's normal camera list, not just as an orphan ID.
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO cameras (id, name, road, lat, lng) VALUES (?,?,?,?,?)",
        (camera_id, camera_name, "User-uploaded feed", lat, lng),
    )
    conn.commit()
    conn.close()

    try:
        results = detect_plates.process_video(
            str(input_path),
            camera_id=camera_id,
            annotate_path=str(output_path),
            store_to_db=store,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        input_path.unlink(missing_ok=True)

    return {
        "camera_id": camera_id,
        "reads_found": len(results),
        "reads": results,
        "annotated_video_url": f"/videos/{output_path.name}",
        "stored_to_dashboard": store,
    }


@app.websocket("/ws/live")
async def live_feed(ws: WebSocket):
    await ws.accept()
    live_clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive / ignore client pings
    except WebSocketDisconnect:
        live_clients.discard(ws)

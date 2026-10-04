"""
Predictive next-camera / ETA estimation.

Turns the system from reactive ("where has this vehicle been") into
proactive ("where is it probably headed, and when will it get there").
This is genuinely useful for intercepting a flagged vehicle rather than
only reviewing its history after the fact -- which is the gap between
a search tool and an operational policing tool.

Method (kept simple and explainable, consistent with the rest of this
project's design philosophy -- a traffic-ops team needs to trust and
audit this, not take a black box's word for it):

  1. Find the vehicle's last confirmed camera.
  2. Look at every OTHER vehicle that has ever been seen at that same
     camera, and where they went next. This gives a frequency
     distribution over "next camera" -- e.g. of everyone seen at
     Electronic City Toll, 80% next appeared at Silk Board Junction.
  3. Convert frequency into probability, and use the average historical
     travel time on that specific hop as the ETA.

This is explicitly a population-level pattern, not a guarantee about
this specific vehicle -- it says where vehicles *like this one's recent
position* usually go, not where this exact vehicle is going. That
distinction matters and should be stated to anyone using it.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from anomaly import get_trajectories_and_edges

DB_PATH = Path(__file__).parent / "anpr.db"

MIN_HISTORICAL_SAMPLES = 2  # below this, a "pattern" is just noise


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _camera_lookup(conn):
    rows = conn.execute("SELECT id, name, road, lat, lng FROM cameras").fetchall()
    return {r["id"]: dict(r) for r in rows}


def _last_confirmed_camera(plate):
    """Where and when was this (observed) plate string last seen."""
    conn = get_conn()
    row = conn.execute(
        """SELECT d.timestamp, c.id as camera_id, c.name as camera_name
           FROM detections d JOIN cameras c ON d.camera_id = c.id
           WHERE d.plate = ? AND d.plate != 'UNREADABLE'
           ORDER BY d.timestamp DESC LIMIT 1""",
        (plate,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def predict_next_camera(plate, top_n=3):
    last = _last_confirmed_camera(plate)
    if not last:
        return {"plate": plate, "last_camera": None, "predictions": []}

    _, edges = get_trajectories_and_edges()
    conn = get_conn()
    cameras = _camera_lookup(conn)
    conn.close()

    # Every historical edge that starts at this vehicle's last known camera.
    outgoing = {
        to_id: times for (from_id, to_id), times in edges.items()
        if from_id == last["camera_id"] and len(times) >= MIN_HISTORICAL_SAMPLES
    }

    if not outgoing:
        return {
            "plate": plate,
            "last_camera": last["camera_name"],
            "last_seen": last["timestamp"],
            "predictions": [],
            "note": "Not enough historical traffic through this camera yet to predict a pattern.",
        }

    total_trips = sum(len(times) for times in outgoing.values())
    last_time = datetime.fromisoformat(last["timestamp"])

    predictions = []
    for to_id, times in outgoing.items():
        count = len(times)
        avg_seconds = sum(times) / count
        probability = round(count / total_trips, 3)
        eta_minutes = round(avg_seconds / 60, 1)
        predicted_arrival = last_time + timedelta(seconds=avg_seconds)
        cam = cameras.get(to_id, {})
        predictions.append({
            "camera_id": to_id,
            "camera_name": cam.get("name", to_id),
            "lat": cam.get("lat"),
            "lng": cam.get("lng"),
            "probability": probability,
            "eta_minutes": eta_minutes,
            "predicted_arrival": predicted_arrival.isoformat(timespec="seconds"),
            "based_on_trips": count,
        })

    predictions.sort(key=lambda p: -p["probability"])

    return {
        "plate": plate,
        "last_camera": last["camera_name"],
        "last_seen": last["timestamp"],
        "predictions": predictions[:top_n],
    }

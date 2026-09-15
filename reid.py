"""
Vehicle re-identification fallback.

The problem statement explicitly calls out plates that are "dirty or
damaged" and conditions like motion blur -- cases where OCR legitimately
can't produce text. Rather than losing the vehicle entirely at that
camera, this module asks: is there a known vehicle whose last confirmed
sighting makes it plausible that THIS unreadable sighting is a
continuation of its route?

This is deliberately NOT a silent auto-merge. Colour-matching is a weak
signal on its own (plenty of white sedans exist) -- these are surfaced
as ranked, scored CANDIDATES for a human operator to confirm, exactly
the way the confidence-scored fuzzy plate matching works. Treating a
probabilistic re-ID match as ground truth would be the wrong design for
a law-enforcement tool.

Matching logic per candidate (kept simple and explainable on purpose):
  1. Colour must match the vehicle's known colour.
  2. The implied travel speed from the vehicle's last confirmed camera
     to the unreadable sighting's camera must be physically plausible
     (not impossibly fast, not absurdly slow for the distance).
  3. Ranked by how "normal" that implied speed is (closer to a typical
     urban travel speed scores higher).
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from anomaly import haversine_km, MAX_PLAUSIBLE_SPEED_KMH

DB_PATH = Path(__file__).parent / "anpr.db"

TYPICAL_SPEED_BAND_KMH = (15, 55)  # confident zone: normal city-road travel speeds
MAX_LOOKAHEAD_HOURS = 3
MIN_PLAUSIBLE_SPEED_KMH = 2  # below this, "travel" is noise (parked, same spot)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _last_known_state(conn, plate):
    """Most recent confirmed (non-UNREADABLE) sighting of this plate."""
    row = conn.execute(
        """SELECT d.timestamp, d.vehicle_color, d.vehicle_type,
                  c.id as camera_id, c.name as camera_name, c.lat, c.lng
           FROM detections d JOIN cameras c ON d.camera_id = c.id
           WHERE d.plate = ? AND d.plate != 'UNREADABLE'
           ORDER BY d.timestamp DESC LIMIT 1""",
        (plate,),
    ).fetchone()
    return dict(row) if row else None


def find_reid_candidates(plate, max_results=5):
    """For a given (resolved) plate, look for UNREADABLE sightings after its
    last confirmed appearance that could plausibly be the same vehicle
    continuing its route."""
    conn = get_conn()
    last = _last_known_state(conn, plate)
    if not last or not last["vehicle_color"]:
        conn.close()
        return []

    last_time = datetime.fromisoformat(last["timestamp"])
    window_end = last_time + timedelta(hours=MAX_LOOKAHEAD_HOURS)

    unreadable = conn.execute(
        """SELECT d.id, d.timestamp, d.vehicle_color, d.vehicle_type,
                  c.id as camera_id, c.name as camera_name, c.lat, c.lng
           FROM detections d JOIN cameras c ON d.camera_id = c.id
           WHERE d.plate = 'UNREADABLE' AND d.timestamp > ? AND d.timestamp <= ?""",
        (last["timestamp"], window_end.isoformat(timespec="seconds")),
    ).fetchall()
    conn.close()

    candidates = []
    for u in unreadable:
        if u["vehicle_color"] != last["vehicle_color"]:
            continue  # colour must match -- the one hard filter we apply

        u_time = datetime.fromisoformat(u["timestamp"])
        seconds = (u_time - last_time).total_seconds()
        if seconds <= 0:
            continue

        distance_km = haversine_km(last["lat"], last["lng"], u["lat"], u["lng"])
        speed_kmh = distance_km / (seconds / 3600) if seconds > 0 else float("inf")

        if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH or speed_kmh < MIN_PLAUSIBLE_SPEED_KMH:
            continue  # physically implausible either way -- not a candidate at all

        # Score: full confidence for speeds within a normal city-road band;
        # tapers off toward the physical limits on either side. A speed
        # deep inside "normal" is a much stronger signal than one that's
        # only barely physically possible.
        low, high = TYPICAL_SPEED_BAND_KMH
        if low <= speed_kmh <= high:
            score = 0.9
        elif speed_kmh < low:
            score = 0.9 * (speed_kmh - MIN_PLAUSIBLE_SPEED_KMH) / (low - MIN_PLAUSIBLE_SPEED_KMH)
        else:
            score = 0.9 * (1 - (speed_kmh - high) / (MAX_PLAUSIBLE_SPEED_KMH - high))
        score = round(max(0.1, min(0.9, score)), 2)

        type_bonus = 0.08 if u["vehicle_type"] == last["vehicle_type"] else 0
        score = min(0.98, round(score + type_bonus, 2))

        candidates.append({
            "detection_id": u["id"],
            "camera": u["camera_name"],
            "timestamp": u["timestamp"],
            "lat": u["lat"],
            "lng": u["lng"],
            "vehicle_color": u["vehicle_color"],
            "vehicle_type": u["vehicle_type"],
            "travel_time_minutes": round(seconds / 60, 1),
            "implied_speed_kmh": round(speed_kmh, 1),
            "confidence": score,
            "basis": f"same colour ({u['vehicle_color']}), "
                     f"{speed_kmh:.0f} km/h implied from last confirmed sighting "
                     f"-- unconfirmed, needs human review",
        })

    candidates.sort(key=lambda c: -c["confidence"])
    return candidates[:max_results]

# Vahan Netra — City-Wide ANPR Trajectory Tracking (prototype)

A working prototype for the SIH problem statement: multi-camera ANPR
trajectory tracking + city traffic analytics. Ships with a realistic
**simulated** camera network and detection log (Bengaluru locations)
so the whole system — search, trajectory reconstruction, live feed,
alerts, analytics — runs end-to-end without needing real camera hardware.

## What's actually implemented

- **Fuzzy plate search** — handles OCR-style misreads (O/0, I/1, B/8, S/5)
  via edit-distance ranking, not just exact match.
- **Trajectory reconstruction** — pick a plate, see every camera it hit,
  in order, with confidence per read, animated on the map.
- **Live feed simulation** — a WebSocket pushes a new "detection" every
  ~3 seconds, as if cameras were reporting in real time.
- **Blacklist alerts** — a few plates are pre-flagged; if the live feed
  happens to "see" one, an alert fires on the dashboard immediately.
- **City analytics** — per-camera density heatmap, busiest node, total
  detections, average OCR confidence.

## What's simulated (and how to be honest about that to judges)

There's no real camera footage or trained OCR model here — that's out of
scope for two people in a hackathon. `backend/app.py` generates a
synthetic but plausible dataset (multiple cameras, realistic timestamps,
some deliberately noisy/low-confidence reads) so every feature above is
demoable. `backend/ocr_pipeline.py` is a clearly separated stub showing
exactly where a real YOLOv8 (plate detection) + PaddleOCR (text
recognition) pipeline would plug in — it would just insert rows into the
same `detections` table, and nothing else in the system changes. That
separation *is* the pitch point: "the tracking and analytics engine is
camera-agnostic; swapping in real RTSP feeds is a data-source change, not
an architecture change."

## Run it

**Backend:**
```bash
cd backend
pip install -r requirements.txt --break-system-packages
uvicorn app:app --reload --port 8000
```
First run auto-creates `anpr.db` with seed data. Delete the file to
reseed with a fresh random dataset.

**Frontend:**
Just open `frontend/index.html` in a browser (it talks to
`http://localhost:8000` directly — no build step needed). If you open it
via `file://` and your browser blocks the WebSocket/fetch calls, serve it
instead:
```bash
cd frontend
python3 -m http.server 5500
# then visit http://localhost:5500
```

## Try it

1. Open the dashboard — camera markers and a heatmap load on the map.
2. Copy a plate string from `backend/anpr.db` (or just wait for the live
   feed panel on the right to show one) and search it. Try changing one
   character — fuzzy search should still surface it.
   the same table
3. Click a match to see its full route animate across the map with a
   timeline of camera hits and confidence scores.
4. Leave it running for a minute — the live feed occasionally "sees" a
   blacklisted plate and an alert appears instantly on the right panel.

## Upload your own footage (in the dashboard now)

The frontend has an "Analyze your own footage" panel in the left
sidebar. Pick a video file, optionally label the camera, and click
"Run detection on this video" — it runs the same real detect+OCR
pipeline as `detect_plates.py`, adds a new camera pin on the map at that
location, stores any accepted reads into the same `detections` table,
and gives you a link to the annotated output video. Everything else in
the dashboard (search, trajectory, analytics, anomalies) then treats
those reads exactly like any other camera's data.

Backend endpoint: `POST /api/upload-video` (multipart form: `file`,
`camera_id`, `camera_name`, `lat`, `lng`, `store`).

**Set expectations before you demo with real footage:** the detector
(`find_plate_candidates` in `detect_plates.py`) was tuned against the
synthetic clip's plain, high-contrast look. Real video — real
backgrounds, lighting, clutter, camera resolution — will very likely
need retuning. The two knobs to adjust first if it's over- or
under-detecting on your clip:
- **aspect ratio bounds** (currently 1.5–6.0) — how wide-vs-tall a
  plate-shaped box needs to be
- **`brightness_delta`** (currently 15) — how much brighter than the
  frame average a region needs to be to be considered plate-like;
  lower it if plates aren't standing out much in your footage, raise it
  if you're getting false positives on other bright things (headlights,
  sky, road markings)

If tuning the classical detector doesn't get you far enough, the
documented next step is swapping in a real trained detector (see
`ocr_pipeline.py`'s notes on YOLOv8) — the rest of the pipeline
(OCR, confidence scoring, DB insert) doesn't need to change either way.

## Real OCR on real video (not simulated)

`backend/generate_sample_video.py` renders a short synthetic traffic clip
(three vehicles: clean, angled, motion-blurred plate — the exact
conditions the problem statement names). `backend/detect_plates.py` then
runs a **real** pipeline on those actual video frames: classical CV
plate localization (Canny edges + contour geometry) and real Tesseract
OCR, no faked text anywhere.

```bash
cd backend
python3 generate_sample_video.py
python3 detect_plates.py sample_traffic.mp4 --camera CAM-05 --annotate annotated_output.mp4 --store
```

- `--annotate` writes a video with bounding boxes + OCR text overlaid, watchable proof for judges.
- `--store` inserts real reads into `anpr.db`, so they immediately show up in the live dashboard.

**Honest result from a test run:** the pipeline correctly localized and
read the clean, front-on plate — but misread `KA05MJ4218` as `KAOSMJ4218`
(0→O, 5→S), a genuine OCR confusion that your fuzzy-search feature is
built to handle. It **completely missed** the angled and motion-blurred
plates — the classical bounding-rectangle filter only recognizes
axis-aligned rectangles, so a rotated or blurred plate doesn't pass the
geometry check. This is real, not a contrived talking point: it's
concrete evidence that (a) a production system needs a rotation-tolerant
or learned detector (fine-tuned YOLOv8, per `ocr_pipeline.py`'s stub) for
the hard cases, and (b) confidence-scored fuzzy matching is doing real
work even on the easy case. Use both findings in your pitch.

## Route anomaly detection

`backend/anomaly.py` builds a baseline of normal camera-to-camera travel
times from all vehicles, then flags any hop that is:
- **IMPOSSIBLE_SPEED** — implied speed between two cameras exceeds a
  physically plausible max (default 120 km/h) — usually a fuzzy-match
  error merging two different vehicles, occasionally a real violation.
- **UNUSUAL_TIMING** — statistically abnormal travel time for that
  specific camera pair (z-score based) versus how long it normally takes.
- **RARE_ROUTE** — a camera-to-camera hop almost never seen city-wide,
  regardless of timing.

It's exposed at `GET /api/analytics/anomalies` and shown in the
dashboard's "Route anomalies" panel — click a card to jump straight to
that vehicle's trajectory on the map.

**Important design note:** the seed data generator (`app.py`) was
rewritten to model realistic commuter corridors (most vehicles follow
one of a few common routes, a minority genuinely don't) rather than
fully random routes for every vehicle. With fully random routes there's
no real "normal" to compare against, so almost everything looks equally
rare and the feature is noise. This matters for your real deployment
too: route anomaly detection needs enough historical traffic on a
corridor before it can say anything meaningful about what's unusual on
it — say this explicitly to judges if asked about cold-start behavior.

Also note: anomaly detection groups hops by **resolved** vehicle
identity, not the raw OCR string — it assumes plate resolution (the
fuzzy-matching search) has already linked noisy reads of the same
vehicle together. This is worth stating in your pitch: anomaly
detection sits *after* plate resolution in the pipeline, not before it.

## Vehicle re-ID fallback (unconfirmed candidates, not auto-merge)

When OCR can't read a plate at all (`detect_plates.py` still localizes a
plate-shaped box but the text fails the format check), the pipeline now
extracts the vehicle's colour from the body above the plate and stores
it as an `UNREADABLE` detection instead of discarding the sighting.
`reid.py` then checks: for any vehicle's last confirmed sighting, is
there a same-coloured `UNREADABLE` sighting at a nearby camera, at a
physically plausible travel speed? If so it's surfaced as a **ranked,
scored candidate** in the trajectory view — never silently merged, since
colour alone is a weak signal (plenty of white sedans exist). Click a
candidate in the dashboard to see a dashed amber line to it on the map.

Exposed via `GET /api/trajectory/{plate}` (now includes a
`reid_candidates` array). Say this explicitly in your pitch: this is a
*human-in-the-loop* fallback, not a claim of certainty — appropriate for
a law-enforcement tool where a wrong auto-merge has real consequences.

## Origin-destination flow (macro traffic patterns)

`GET /api/analytics/od-matrix` aggregates every camera-to-camera hop
across all vehicles into ranked routes (trip count + average travel
time), shown both as a weighted flow-line overlay on the map (thicker
line = more traffic) and a ranked list in the sidebar. This is the
"origin-destination patterns" requirement from the problem statement,
distinct from single-vehicle trajectory tracking — it answers "where
does city traffic actually flow" rather than "where did this one
vehicle go."

## Where to take this next (good "Phase 2" slide material)

- Real YOLOv8 + PaddleOCR pipeline (`ocr_pipeline.py` is the entry point)
- Vehicle re-ID fallback (color/type match) when OCR confidence is too
  low to trust — keeps a trajectory alive across an unreadable plate
- Edge-side processing sending only plate hashes/embeddings, not raw
  video, to the central server (privacy-by-design, DPDP Act 2023 framing)
- Predictive next-camera / ETA estimation from historical route patterns
- Route-anomaly detection for the analytics layer

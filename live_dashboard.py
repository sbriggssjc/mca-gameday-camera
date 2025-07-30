import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

from flask import Flask, jsonify, redirect, render_template, request, url_for

DB_PATH = Path("output/live_play_log.db")
CLIP_DIR = Path("video/highlights")

app = Flask(__name__)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "timestamp TEXT, label TEXT, confidence REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scoreboard ("
        "id INTEGER PRIMARY KEY CHECK(id=1), home TEXT, away TEXT)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO scoreboard(id, home, away) VALUES (1, '0', '0')"
    )
    conn.commit()
    conn.close()


_last_ts: float | None = None


def classify_clip(path: Path) -> tuple[str, float]:
    """Placeholder classifier returning a deterministic pseudo-random label."""
    import random

    labels = ["Jet Sweep", "Dive", "Power R", "Pass", "Screen"]
    random.seed(path.stat().st_mtime)
    label = random.choice(labels)
    confidence = random.uniform(0.75, 0.99)
    return label, confidence


def new_clips() -> Iterable[Path]:
    global _last_ts
    clips = sorted(CLIP_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    for clip in clips:
        mtime = clip.stat().st_mtime
        if _last_ts is None or mtime > _last_ts:
            _last_ts = mtime
            yield clip


def classifier_loop() -> None:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for clip in new_clips():
            label, conf = classify_clip(clip)
            ts = time.strftime("%H:%M:%S")
            conn = get_db()
            conn.execute(
                "INSERT INTO predictions(timestamp, label, confidence) VALUES (?,?,?)",
                (ts, label, conf),
            )
            conn.commit()
            conn.close()
        time.sleep(10)


@app.route("/", methods=["GET", "POST"])
def index():
    conn = get_db()
    if request.method == "POST":
        home = request.form.get("home", "0")
        away = request.form.get("away", "0")
        conn.execute("UPDATE scoreboard SET home=?, away=? WHERE id=1", (home, away))
        conn.commit()
        return redirect(url_for("index"))

    preds = conn.execute(
        "SELECT timestamp, label, confidence FROM predictions ORDER BY id DESC"
    ).fetchall()
    score = conn.execute("SELECT home, away FROM scoreboard WHERE id=1").fetchone()
    conn.close()
    log = [dict(r) for r in preds][::-1]
    latest = log[-1] if log else None
    return render_template("dashboard.html", log=log, latest=latest, score=score)


@app.route("/data.json")
def data_json():
    conn = get_db()
    rows = conn.execute(
        "SELECT label, COUNT(*) as count FROM predictions GROUP BY label"
    ).fetchall()
    conn.close()
    return jsonify({r["label"]: r["count"] for r in rows})


if __name__ == "__main__":
    init_db()
    thread = threading.Thread(target=classifier_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=5000)

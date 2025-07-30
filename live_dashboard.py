from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

LOG_PATH = Path("live_log.json")
SCORE_PATH = Path("live_score.json")

app = Flask(__name__)


def load_plays() -> list[dict]:
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            plays = json.load(f)
            if isinstance(plays, list):
                return plays
    except Exception:
        pass
    return []


def load_score() -> dict:
    try:
        with SCORE_PATH.open("r", encoding="utf-8") as f:
            score = json.load(f)
            if isinstance(score, dict):
                return {"MCA": int(score.get("MCA", 0)), "Opp": int(score.get("Opp", 0))}
    except Exception:
        pass
    return {"MCA": 0, "Opp": 0}


def save_score(score: dict) -> None:
    SCORE_PATH.write_text(json.dumps(score), encoding="utf-8")


@app.route("/")
def index() -> str:
    score = load_score()
    stream_id = os.environ.get("YOUTUBE_STREAM_ID")
    return render_template("dashboard.html", score=score, stream_id=stream_id)


@app.route("/api/plays")
def api_plays() -> tuple[str, int] | tuple[str, int, dict]:
    return jsonify(load_plays())


@app.route("/api/score", methods=["GET", "POST"])
def api_score() -> tuple[str, int] | tuple[str, int, dict]:
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        score = {"MCA": int(data.get("MCA", 0)), "Opp": int(data.get("Opp", 0))}
        save_score(score)
        return jsonify(score)
    return jsonify(load_score())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

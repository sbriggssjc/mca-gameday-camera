from __future__ import annotations

<<<<<<< HEAD
import os
from pathlib import Path

from tools.json_io import load_json_safe, dump_json_safe

=======
import json
import os
from pathlib import Path

>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
from flask import Flask, jsonify, render_template, request

from predict_next_play import predict_play

SCOUTING_PATH = Path("analysis/Victory_Christian_scouting_report.json")

LOG_PATH = Path("live_log.json")
SCORE_PATH = Path("live_score.json")
OPPONENT = os.environ.get("OPPONENT_NAME", "Victory Christian")


def load_scouting() -> list[dict]:
<<<<<<< HEAD
    data = load_json_safe(SCOUTING_PATH, default=[])
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        pats = data.get("patterns", [])
        if isinstance(pats, list):
            return [d for d in pats if isinstance(d, dict)]
=======
    try:
        with SCOUTING_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            if isinstance(data, dict):
                pats = data.get("patterns", [])
                if isinstance(pats, list):
                    return [d for d in pats if isinstance(d, dict)]
    except Exception:
        pass
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
    return []

app = Flask(__name__)

SCOUTING_PATTERNS = load_scouting()


def load_plays() -> list[dict]:
<<<<<<< HEAD
    plays = load_json_safe(LOG_PATH, default=[])
    if isinstance(plays, list):
        return plays
=======
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            plays = json.load(f)
            if isinstance(plays, list):
                return plays
    except Exception:
        pass
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
    return []


def load_score() -> dict:
<<<<<<< HEAD
    score = load_json_safe(SCORE_PATH, default={})
    if isinstance(score, dict):
        return {"MCA": int(score.get("MCA", 0)), "Opp": int(score.get("Opp", 0))}
=======
    try:
        with SCORE_PATH.open("r", encoding="utf-8") as f:
            score = json.load(f)
            if isinstance(score, dict):
                return {"MCA": int(score.get("MCA", 0)), "Opp": int(score.get("Opp", 0))}
    except Exception:
        pass
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
    return {"MCA": 0, "Opp": 0}


def save_score(score: dict) -> None:
<<<<<<< HEAD
    dump_json_safe(SCORE_PATH, score)
=======
    SCORE_PATH.write_text(json.dumps(score), encoding="utf-8")
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa


@app.route("/")
def index() -> str:
    score = load_score()
    stream_id = os.environ.get("YOUTUBE_STREAM_ID")
    return render_template(
        "dashboard.html",
        score=score,
        stream_id=stream_id,
        opponent=OPPONENT,
    )


@app.route("/api/plays")
def api_plays() -> tuple[str, int] | tuple[str, int, dict]:
    return jsonify(load_plays())


@app.route("/api/scouting")
def api_scouting() -> tuple[str, int] | tuple[str, int, dict]:
    return jsonify(SCOUTING_PATTERNS)


@app.route("/api/predict")
def api_predict() -> tuple[str, int] | tuple[str, int, dict]:
    opponent = request.args.get("opponent", "")
    formation = request.args.get("formation", "")
    down = request.args.get("down", type=int)
    distance = request.args.get("distance", type=int)
    quarter = request.args.get("quarter", type=int)
    result = predict_play(opponent, formation, down, distance, quarter)
    return jsonify(result)


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

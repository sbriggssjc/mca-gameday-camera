import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

SELF_LOG_PATH = Path("training/logs/self_learning_log.json")
PLAYER_COUNTS_PATH = Path("output/summaries/player_play_counts.json")
REVIEW_QUEUE_PATH = Path("training/review_queue.json")


def load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="AI Performance Dashboard", layout="wide")

    history: List[Dict[str, Any]] = load_json(SELF_LOG_PATH) or []
    player_counts: Dict[str, int] = load_json(PLAYER_COUNTS_PATH) or {}
    review_queue: List[Dict[str, Any]] = load_json(REVIEW_QUEUE_PATH) or []

    df = pd.DataFrame(history)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    st.sidebar.header("Filters")
    video_options = sorted(df["video"].dropna().unique()) if not df.empty else []
    selected_video = st.sidebar.selectbox("Video", ["All"] + video_options)
    search_text = st.sidebar.text_input("Search by name/date")

    filtered_df = df.copy()
    if selected_video != "All":
        filtered_df = filtered_df[filtered_df["video"] == selected_video]
    if search_text:
        search_text_lower = search_text.lower()
        filtered_df = filtered_df[
            filtered_df["video"].str.lower().str.contains(search_text_lower, na=False)
            | filtered_df["timestamp"].astype(str).str.contains(search_text_lower, na=False)
        ]

    st.title("AI Performance Dashboard")

    st.header("Summary Stats (Most Recent Run)")
    if not filtered_df.empty:
        last = filtered_df.iloc[-1]
    else:
        last = {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Plays Detected", int(last.get("plays_detected", 0)))
    c1.metric("Jersey Numbers Recognized", int(last.get("jersey_numbers_recognized", 0)))
    c2.metric("OCR Failures", int(last.get("jersey_ocr_failures", 0)))
    c2.metric("Unknown Play Types", int(last.get("unknown_play_types", 0)))
    c3.metric("Flagged Frames Saved", int(last.get("frames_saved", 0)))

    st.header("Graphs")
    if not filtered_df.empty:
        # OCR success rate over time
        filtered_df["ocr_success_rate"] = (
            filtered_df["jersey_numbers_recognized"]
            / (filtered_df["jersey_numbers_recognized"] + filtered_df["jersey_ocr_failures"])
        )
        ocr_chart_df = filtered_df.set_index("timestamp")["ocr_success_rate"]
        st.subheader("OCR Success Rate Over Time")
        st.line_chart(ocr_chart_df)

        flagged_df = filtered_df.set_index("video")["frames_saved"]
        st.subheader("Total Flagged Examples Per Game")
        st.bar_chart(flagged_df)
    else:
        st.write("No learning history available.")

    st.subheader("Jersey Number Frequency")
    if player_counts:
        jersey_df = pd.DataFrame(
            list(player_counts.items()), columns=["Jersey", "Plays"]
        ).sort_values("Plays", ascending=False)
        st.bar_chart(jersey_df.set_index("Jersey"))
        st.subheader("Play Count Table")
        st.table(jersey_df)
    else:
        st.write("No play count summary available.")

    st.header("Review Queue Snapshot")
    ocr_entries = [e for e in review_queue if e.get("type") == "uncertain_jersey"]
    play_entries = [e for e in review_queue if e.get("type") == "unknown_play"]
    st.write(f"Unresolved OCR Entries: {len(ocr_entries)}")
    st.write(f"Untagged Plays: {len(play_entries)}")


if __name__ == "__main__":
    main()

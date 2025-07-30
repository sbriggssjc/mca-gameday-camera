"""Streamlit dashboard for visualizing play data."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

try:  # optional dependency for PDF export
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None  # type: ignore

PREDICTIONS_PATH = Path('outputs/predictions.json')
METADATA_PATH = Path('outputs/play_metadata.json')
PDF_PATH = Path('outputs/game_summary.pdf')


def load_json(path: Path) -> Any:
    """Load JSON data if the file exists."""
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def compute_player_counts(metadata: List[Dict[str, Any]]) -> Counter:
    """Return snap count per jersey number from metadata."""
    counts: Counter = Counter()
    for frame in metadata:
        for trk in frame.get('tracks', []):
            jersey = trk.get('jersey')
            if jersey:
                counts[jersey] += 1
    return counts


def generate_pdf(df: pd.DataFrame, player_counts: Counter, output: Path) -> None:
    """Generate a simple summary PDF if FPDF is available."""
    if FPDF is None:
        raise RuntimeError('fpdf package is required for PDF export')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Game Summary', ln=1)
    pdf.set_font('Helvetica', size=12)

    pdf.cell(0, 8, f'Total Plays: {len(df)}', ln=1)
    if not df.empty:
        top_play = df['play_type'].value_counts().idxmax()
        top_count = df['play_type'].value_counts().max()
        pdf.cell(0, 8, f'Most Called: {top_play} ({top_count})', ln=1)
    if 'success' in df.columns:
        successes = int(df['success'].sum())
        pdf.cell(0, 8, f'Successful Plays: {successes} / {len(df)}', ln=1)
    if player_counts:
        leader = max(player_counts, key=player_counts.get)
        pdf.cell(0, 8, f'Leading Player: #{leader} ({player_counts[leader]} snaps)', ln=1)

    pdf.output(str(output))


def main() -> None:
    st.set_page_config(page_title='Play Dashboard', layout='wide')
    st.title('Game Dashboard')

    pred_data = load_json(PREDICTIONS_PATH) or []
    meta_data = load_json(METADATA_PATH) or []

    df = pd.DataFrame(pred_data)
    player_counts = compute_player_counts(meta_data) if meta_data else Counter()

    st.sidebar.header('Filters')
    play_options = ['All'] + sorted(df['play_type'].dropna().unique()) if not df.empty else ['All']
    player_options = ['All'] + sorted(player_counts) if player_counts else ['All']
    selected_play = st.sidebar.selectbox('Play Type', play_options)
    selected_player = st.sidebar.selectbox('Player', player_options)

    filtered_df = df.copy()
    if selected_play != 'All':
        filtered_df = filtered_df[filtered_df['play_type'] == selected_play]
    if selected_player != 'All' and meta_data:
        frames_with_player = {
            frame['frame']
            for frame in meta_data
            for trk in frame.get('tracks', [])
            if trk.get('jersey') == selected_player
        }
        filtered_df = filtered_df[filtered_df.get('start_frame').isin(frames_with_player)]

    st.header('Play Breakdown')
    if not df.empty and 'play_type' in df.columns:
        st.subheader('Play Frequency')
        st.bar_chart(df['play_type'].value_counts())

    if not df.empty and 'success' in df.columns and 'time' in df.columns:
        st.subheader('Play Success Over Time')
        success_df = df.sort_values('time')[['time', 'success']]
        st.line_chart(success_df.set_index('time'))

    if player_counts:
        st.subheader('Snap Counts')
        snap_df = pd.DataFrame(player_counts.items(), columns=['Jersey', 'Snaps']).sort_values('Snaps', ascending=False)
        st.bar_chart(snap_df.set_index('Jersey'))

    st.subheader('All Plays')
    if not filtered_df.empty:
        table_cols = [c for c in ['time', 'play_type', 'confidence'] if c in filtered_df.columns]
        st.table(filtered_df[table_cols])
    else:
        st.write('No plays to display.')

    if st.button('Generate PDF'):
        try:
            generate_pdf(df, player_counts, PDF_PATH)
            st.success(f'Saved summary to {PDF_PATH}')
        except Exception as e:
            st.error(str(e))


if __name__ == '__main__':
    main()

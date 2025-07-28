import json
from pathlib import Path
from datetime import datetime
import streamlit as st


OCR_REVIEW_PATH = Path('/training/labels/ocr_review.json')
CONFIRMED_JERSEYS_PATH = Path('/training/labels/confirmed_jerseys.json')
UNCERTAIN_DIR = Path('/training/uncertain_jerseys')
LABELS_DIR = Path('/training/labels')
FRAMES_DIR = Path('/training/frames')
CONFIRMED_PLAYS_PATH = Path('/training/labels/confirmed_play_types.json')


st.set_page_config(page_title='Coach Review', layout='wide')


def load_json(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def append_json(path: Path, item: dict) -> None:
    data = []
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_ocr_entries():
    entries = load_json(OCR_REVIEW_PATH)
    if isinstance(entries, list):
        entries.sort(key=lambda x: x.get('timestamp', ''))
        return entries
    return []


def load_unknown_play_entries():
    entries = []
    if LABELS_DIR.exists():
        for json_file in sorted(LABELS_DIR.glob('play_*.json')):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('play_type') == 'unknown':
                    ts = data.get('timestamp', '')
                    fname = json_file.stem + '.jpg'
                    entry = {
                        'play_id': data.get('play_id'),
                        'timestamp': ts,
                        'frame': str(FRAMES_DIR / fname),
                    }
                    entries.append(entry)
            except Exception:
                continue
    entries.sort(key=lambda x: x.get('timestamp', ''))
    return entries


def jersey_tab(entries):
    if 'ocr_idx' not in st.session_state:
        st.session_state.ocr_idx = 0
    total = len(entries)
    if total == 0:
        st.write('No uncertain jerseys found.')
        return
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button('Previous', disabled=st.session_state.ocr_idx <= 0):
            st.session_state.ocr_idx = max(0, st.session_state.ocr_idx - 1)
    with col3:
        if st.button('Next', disabled=st.session_state.ocr_idx >= total - 1):
            st.session_state.ocr_idx = min(total - 1, st.session_state.ocr_idx + 1)
    entry = entries[st.session_state.ocr_idx]
    img_path = UNCERTAIN_DIR / entry.get('filename', '')
    st.image(str(img_path), width=400)
    st.write(f"Video: {entry.get('video','')}")
    st.write(f"Frame ID: {entry.get('frame_id','')}")
    st.write(f"Timestamp: {entry.get('timestamp','')}")
    st.write(f"BBox: {entry.get('bbox')}")
    jersey = st.text_input('Correct Jersey Number', key=f'jersey_input_{st.session_state.ocr_idx}')
    if st.button('Submit', key=f'jersey_submit_{st.session_state.ocr_idx}'):
        record = {
            'filename': entry.get('filename'),
            'jersey_number': jersey,
            'bbox': entry.get('bbox'),
            'timestamp': entry.get('timestamp'),
        }
        append_json(CONFIRMED_JERSEYS_PATH, record)
        st.success('Saved')


def play_tab(entries):
    if 'play_idx' not in st.session_state:
        st.session_state.play_idx = 0
    total = len(entries)
    if total == 0:
        st.write('No unknown plays found.')
        return
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button('Previous ', key='play_prev', disabled=st.session_state.play_idx <= 0):
            st.session_state.play_idx = max(0, st.session_state.play_idx - 1)
    with col3:
        if st.button('Next ', key='play_next', disabled=st.session_state.play_idx >= total - 1):
            st.session_state.play_idx = min(total - 1, st.session_state.play_idx + 1)
    entry = entries[st.session_state.play_idx]
    st.image(entry['frame'], width=400)
    st.write(f"Play ID: {entry.get('play_id')}")
    st.write(f"Timestamp: {entry.get('timestamp')}")
    play_type = st.text_input('Play Type', key=f'play_type_{st.session_state.play_idx}')
    ball_carrier = st.text_input('Ball Carrier Jersey Number', key=f'ball_{st.session_state.play_idx}')
    if st.button('Submit', key=f'play_submit_{st.session_state.play_idx}'):
        record = {
            'play_id': entry.get('play_id'),
            'timestamp': entry.get('timestamp'),
            'play_type': play_type,
            'ball_carrier': ball_carrier,
            'filename': Path(entry['frame']).name,
        }
        append_json(CONFIRMED_PLAYS_PATH, record)
        st.success('Saved')


def main():
    ocr_entries = load_ocr_entries()
    play_entries = load_unknown_play_entries()

    st.markdown(f"**Pending Jerseys: {len(ocr_entries)}**")
    st.markdown(f"**Pending Plays: {len(play_entries)}**")

    tab1, tab2 = st.tabs(['Uncertain Jersey Numbers', 'Unknown Play Types'])

    with tab1:
        jersey_tab(ocr_entries)
    with tab2:
        play_tab(play_entries)


if __name__ == '__main__':
    main()

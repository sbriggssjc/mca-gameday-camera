import numpy as np


def estimate_snap_t(audio_rms, fps_audio, start_idx, window=0.8):
    # Look for first sustained RMS increase after pre-play period
    # Return index of snap (samples)
    w = int(window * fps_audio)
    seg = audio_rms[start_idx:start_idx + 5 * w]
    if seg.size == 0:
        return start_idx
    # Simple threshold: mean + 2*std
    thr = seg.mean() + 2.0 * seg.std()
    for i in range(min(len(seg) - w, 5 * w)):
        if seg[i:i + w].mean() > thr:
            return start_idx + i
    return start_idx


def estimate_whistle_t(audio_rms, fps_audio, snap_idx, max_len_s=12.0):
    # Look for sharp HF energy or sustained high RMS near play end; fallback to inactivity
    end_idx = snap_idx + int(max_len_s * fps_audio)
    return end_idx


def merge_short_gaps(plays, min_gap_s=1.0, fps=30):
    # If two segments are separated by tiny gap, merge to avoid mid-play cuts
    merged = []
    for seg in plays:
        if merged and (seg[0] - merged[-1][1]) < int(min_gap_s * fps):
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)
    return merged

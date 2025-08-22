from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
import math
from typing import List, Tuple, Optional


@dataclass
class SegParams:
    fps_video: float
    fps_audio: float | None = None
    pre_s: float = 2.0
    post_s: float = 2.5
    max_play_s: float = 12.0
    min_idle_s: float = 1.5
    min_sustain_s: float = 0.25
    whistle_hf_band: Tuple[int, int] = (3000, 6000)
    motion_band_h: Tuple[float, float] = (0.35, 0.65)
    motion_thr_mult: float = 2.0
    audio_thr_mult: float = 2.0


@dataclass
class PlayWindow:
    idx: int
    start_f: int
    snap_f: int
    whistle_f: int
    end_f: int


class SnapWhistleFinder:
    """
    Single source of truth for play windows.
    - Audio RMS preferred for snap + whistle.
    - Motion fallback when audio missing/low-SNR.
    - Produces non-overlapping, fully refined [start,end] with snap,end markers.
    """

    def __init__(self, params: SegParams):
        self.p = params

    # ===== Public API =====
    def find_plays(self, vid_reader, audio_rms: Optional[np.ndarray]) -> List[PlayWindow]:
        """
        vid_reader: object with properties width, height, frame_count, fps and iter_frames() -> np.ndarray BGR
        audio_rms: 1-D np.ndarray (per-sample or per-chunk RMS). If None -> motion-only mode.
        """
        H, W = vid_reader.height, vid_reader.width
        N = vid_reader.frame_count
        fps = self.p.fps_video

        # 1) Precompute motion energy per frame (fast)
        motion = self._motion_energy(vid_reader, band=self.p.motion_band_h)

        # 2) Downsample/align audio RMS to video fps if present
        audio_v = self._align_audio_to_video(audio_rms, N, fps) if audio_rms is not None else None

        # 3) Detect snaps as peaks after idle lulls
        snap_frames = self._detect_snaps(audio_v, motion, fps)

        # 4) For each snap, detect whistle/end (audio preferred)
        plays: List[PlayWindow] = []
        for i, snap_f in enumerate(snap_frames, start=1):
            start_f = max(0, snap_f - int(self.p.pre_s * fps))
            whistle_f = self._detect_whistle(audio_v, motion, snap_f, fps)
            if whistle_f is None:
                whistle_f = min(N - 1, snap_f + int(self.p.max_play_s * fps))
            end_f = min(N - 1, whistle_f + int(self.p.post_s * fps))
            plays.append(PlayWindow(i, start_f, snap_f, whistle_f, end_f))

        # 5) Deduplicate/clean: drop overlapping by preferring earlier snap; never "merge"
        plays = self._suppress_overlaps(plays)

        return plays

    # ===== Internals =====
    def _motion_energy(self, vid_reader, band: Tuple[float, float]) -> np.ndarray:
        # Light-weight frame-diff energy restricted to horizontal band around LOS
        fps = self.p.fps_video
        band_lo, band_hi = band
        prev = None
        energy = np.zeros(vid_reader.frame_count, dtype=np.float32)

        for fi, frame in enumerate(vid_reader.iter_frames()):
            h, w = frame.shape[:2]
            y0, y1 = int(h * band_lo), int(h * band_hi)
            sub = frame[y0:y1]
            if cv2 is not None:
                roi = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            else:
                roi = sub.mean(axis=2).astype(np.uint8)
            if prev is not None:
                if cv2 is not None:
                    diff = cv2.absdiff(roi, prev)
                else:
                    diff = np.abs(roi.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
                energy[fi] = float(diff.mean())
            prev = roi
        # Normalize
        if energy.std() > 1e-6:
            energy = (energy - energy.mean()) / (energy.std() + 1e-6)
        return energy

    def _align_audio_to_video(self, audio_rms: np.ndarray, N_frames: int, fps_video: float) -> np.ndarray:
        # Assume audio_rms sampled at ~fps_audio (or per sample). Resample to N_frames.
        if audio_rms is None or len(audio_rms) == 0:
            return None
        x = np.linspace(0, 1, len(audio_rms), endpoint=False)
        xv = np.linspace(0, 1, N_frames, endpoint=False)
        return np.interp(xv, x, audio_rms).astype(np.float32)

    def _detect_snaps(self, audio_v: Optional[np.ndarray], motion: np.ndarray, fps: float) -> List[int]:
        snaps: List[int] = []
        idle_win = int(self.p.min_idle_s * fps)
        sustain = int(self.p.min_sustain_s * fps)

        # Build a unified "activity" signal; audio preferred
        if audio_v is not None:
            mu, sd = float(audio_v.mean()), float(audio_v.std() + 1e-6)
            thr = mu + self.p.audio_thr_mult * sd
            activity = audio_v
        else:
            mu, sd = float(motion.mean()), float(motion.std() + 1e-6)
            thr = mu + self.p.motion_thr_mult * sd
            activity = motion

        i = idle_win
        while i < len(activity) - sustain:
            # require idle lull
            if activity[i - idle_win : i].mean() < (mu + 0.2 * sd):
                # snap = first sustained rise
                if activity[i : i + sustain].mean() > thr:
                    snaps.append(i)
                    # skip forward to avoid double-detect
                    i += int(0.8 * fps)
                    continue
            i += 1
        return snaps

    def _detect_whistle(self, audio_v: Optional[np.ndarray], motion: np.ndarray, snap_f: int, fps: float) -> Optional[int]:
        # Prefer audio decay with brief HF spike; else motion decay
        # Simple heuristic: play ends when both audio and motion drop near baseline for ~0.6s
        dur_cap = int(self.p.max_play_s * fps)
        end_search = min(len(motion) - 1, snap_f + dur_cap)
        window_ok = int(0.6 * fps)

        mu_m, sd_m = float(motion.mean()), float(motion.std() + 1e-6)
        low_m = mu_m + 0.4 * sd_m

        if audio_v is not None:
            mu_a, sd_a = float(audio_v.mean()), float(audio_v.std() + 1e-6)
            low_a = mu_a + 0.4 * sd_a
            for j in range(snap_f + int(0.8 * fps), end_search - window_ok):
                if (motion[j : j + window_ok].mean() < low_m) and (
                    audio_v[j : j + window_ok].mean() < low_a
                ):
                    return j
        # Fallback: motion only
        for j in range(snap_f + int(0.8 * fps), end_search - window_ok):
            if motion[j : j + window_ok].mean() < low_m:
                return j
        return None

    def _suppress_overlaps(self, plays: List[PlayWindow]) -> List[PlayWindow]:
        if not plays:
            return plays
        plays.sort(key=lambda p: (p.snap_f, p.start_f))
        cleaned = [plays[0]]
        for pw in plays[1:]:
            last = cleaned[-1]
            if pw.start_f <= last.end_f:
                # Overlap → keep earlier snap (last) and drop this one
                continue
            cleaned.append(pw)
        # Reindex
        for i, p in enumerate(cleaned, 1):
            p.idx = i
        return cleaned

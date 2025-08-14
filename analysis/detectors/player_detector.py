"""
Lightweight player detector used by the pipeline.

Two entrypoints:
  from analysis.detectors import player_detector as det
  boxes = det.player_detector(frame_bgr)        # function-style
  # or
  from analysis.detectors.player_detector import PlayerDetector
  boxes = PlayerDetector().detect(frame_bgr)    # class-style

Returns: list of dicts {x1,y1,x2,y2,score,label}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2, numpy as np

@dataclass
class DetectorConfig:
    history: int = 150
    var_threshold: float = 16.0
    detect_shadows: bool = True
    morph_open: int = 3
    morph_close: int = 5
    min_area: int = 200
    max_area: int = 20000
    min_aspect: float = 0.25
    max_aspect: float = 2.5
    dilate_iters: int = 1

class PlayerDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=self.cfg.detect_shadows,
        )
        self.k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_open,  self.cfg.morph_open))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_close, self.cfg.morph_close))

    def _post(self, mask):
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drop shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        if self.cfg.dilate_iters > 0:
            mask = cv2.dilate(mask, None, iterations=self.cfg.dilate_iters)
        return mask

    def _gate(self, rect: Tuple[int,int,int,int], frame_area: int) -> bool:
        x,y,w,h = rect
        if w<=0 or h<=0: return False
        area = w*h
        if area < self.cfg.min_area or area > self.cfg.max_area: return False
        aspect = w/float(h)
        if not (self.cfg.min_aspect <= aspect <= self.cfg.max_aspect): return False
        if area > 0.25*frame_area: return False
        return True

    def detect(self, frame_bgr) -> List[Dict]:
        if frame_bgr is None or frame_bgr.size == 0: return []
        H,W = frame_bgr.shape[:2]
        fg = self.bg.apply(frame_bgr)
        mask = self._post(fg)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Dict] = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if not self._gate((x,y,w,h), W*H): continue
            area = float(w*h)
            score = float(min(1.0, area/20000.0))
            boxes.append({"x1":int(x), "y1":int(y), "x2":int(x+w), "y2":int(y+h),
                          "score":score, "label":"player"})
        return boxes

# module-level singleton for function-style use
_GLOBAL_DET: Optional[PlayerDetector] = None
def _get_global() -> PlayerDetector:
    global _GLOBAL_DET
    if _GLOBAL_DET is None:
        _GLOBAL_DET = PlayerDetector()
    return _GLOBAL_DET

def player_detector(frame_bgr) -> List[Dict]:
    return _get_global().detect(frame_bgr)

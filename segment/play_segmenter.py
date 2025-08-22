from __future__ import annotations


def _windowize(*args, **kwargs):
    raise NotImplementedError("Legacy windowize disabled. Use SnapWhistleFinder.find_plays().")


def coalesce_segments(*args, **kwargs):
    raise NotImplementedError("Legacy merging disabled. Snap→Whistle produces final windows.")


def primary_detect(*args, **kwargs):
    raise NotImplementedError("Legacy segmenter disabled. Use SnapWhistleFinder.find_plays().")


def segment_video(*args, **kwargs):
    raise NotImplementedError("Legacy segmenter disabled. Use SnapWhistleFinder.find_plays().")

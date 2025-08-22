# Training Dataset

This directory stores training videos, extracted frames, and labels used for self-learning.

```
training/dataset/
├── videos/   # raw input clips
├── frames/   # extracted reference frames
└── labels/   # JSON label files per video
```

Use `ai_trainer.py` to populate these folders and evaluate models.

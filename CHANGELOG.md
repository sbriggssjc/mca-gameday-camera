# Changelog

All notable changes to **mca-gameday-camera** will be documented in this file.

## [v0.3.0] — 2025-08-21

### Added
- **Optional Google Drive uploads**: Drive sync is disabled by default and can be enabled with `GOOGLE_DRIVE_SYNC=1`. When disabled, the pipeline skips Drive imports and continues cleanly.
- **Robust backfill for legacy/new JSONL**:
  - Accepts `formation` as either a dict (`{"name": "...", "confidence": ...}`) or a scalar string (`"Reo"`).
  - Helper `_as_name(x)` normalizes formation names from mixed shapes.
- **Classifier outputs plumbed through**:
  - Pipeline writes `"playcall": {"name": ..., "confidence": ...}` to JSONL.
  - Backfill surfaces `play_family` and `playcall_confidence` in CSV.

### Changed
- **CSV schema unified** (order matters):


play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration

- `formation` is always a scalar name in CSV, `formation_confidence` is numeric (defaults to `0.0`).
- `t0`/`t1` are written as blank fields when missing; JSONL uses `null`.
- **Playbook logging**:
- Logs at start: `[playbook] source=<arg or path>`
- After resolution: `[playbook] OK: loaded playbook from <resolved_path>`
- Echo requested: `[playbook] OK: requested playbook: <original_arg>`

### Fixed
- **Drive query f-string crash** in `tools/gdrive_sync.py`:
- Replaced fragile nested-quote f-string with safe escaping/formatting.
- Removed import-time side effects so importing the module can never crash the run.
- **Storage cleanup** resilience:
- Lazy import of Drive uploader; errors are logged as warnings and the pipeline proceeds.

### Notes
- Audio diagnostics & capture remain unchanged; existing Pulse/ALSA probing and env-driven audio settings still apply.

### Upgrade Guide
1. Deploy the code.
2. (Optional) Enable Drive uploads by setting:
 ```bash
 export GOOGLE_DRIVE_SYNC=1


Ensure credentials are configured before enabling.
3. Validate with the provided smoke test:

```
tools/smoke_test.sh

```

---

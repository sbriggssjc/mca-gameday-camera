#!/usr/bin/env bash
#
# duplicates.sh - locate duplicate files and optionally delete them interactively.
# Runs in reporting (dry-run) mode by default.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: duplicates.sh [options]

Options:
  --root DIR          Directory to scan (default: repository root).
  --min-size SIZE     Only consider files >= SIZE bytes (default: 0).
  --limit N           Limit the number of duplicate sets shown (default: all).
  --resolve           Interactively remove duplicates after confirmation.
  --keep DIR          Always keep files under this directory (can repeat).
  -h, --help          Show this help message.

Examples:
  duplicates.sh --limit 10
  duplicates.sh --resolve --min-size 1048576
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

ROOT="$REPO_ROOT"
MIN_SIZE=0
LIMIT=0
DRY_RUN=1
KEEP_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --min-size)
      MIN_SIZE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --resolve)
      DRY_RUN=0
      shift
      ;;
    --keep)
      KEEP_PATHS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[duplicates] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "[duplicates] root directory not found: $ROOT" >&2
  exit 1
fi

ROOT=$(cd "$ROOT" && pwd)

TMP_JSON=$(mktemp)
trap 'rm -f "$TMP_JSON"' EXIT

python3 - "$ROOT" "$MIN_SIZE" "${KEEP_PATHS[@]}" <<'PY' > "$TMP_JSON"
import hashlib
import json
import os
import sys

if len(sys.argv) < 3:
    raise SystemExit("usage: script ROOT MIN_SIZE [KEEP...] ")
root = sys.argv[1]
min_size = int(sys.argv[2])
keep_prefixes = [os.path.abspath(p) for p in sys.argv[3:]]

def should_keep(path):
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(prefix) for prefix in keep_prefixes)

size_map = {}
for current_root, dirs, files in os.walk(root):
    if '.git' in dirs:
        dirs.remove('.git')
    for name in files:
        full_path = os.path.join(current_root, name)
        try:
            size = os.path.getsize(full_path)
        except OSError:
            continue
        if size < min_size:
            continue
        size_map.setdefault(size, []).append(full_path)

duplicates = []
for size, paths in size_map.items():
    if len(paths) < 2:
        continue
    hash_map = {}
    for path in paths:
        if should_keep(path):
            continue
        try:
            with open(path, 'rb') as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            continue
        hash_map.setdefault(digest, []).append(path)
    for digest, dup_paths in hash_map.items():
        if len(dup_paths) > 1:
            duplicates.append({'size': size, 'paths': sorted(dup_paths)})

duplicates.sort(key=lambda item: (-item['size'], item['paths']))
json.dump(duplicates, sys.stdout)
PY

if [[ ! -s "$TMP_JSON" ]]; then
  echo "[duplicates] no duplicates detected"
  exit 0
fi

if (( DRY_RUN )); then
  echo "[duplicates] reporting mode (dry-run)"
  python3 - "$TMP_JSON" "$LIMIT" <<'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
limit = int(sys.argv[2])
if limit > 0:
    data = data[:limit]
for idx, entry in enumerate(data, 1):
    size_mb = entry['size'] / (1024 * 1024)
    print(f"[{idx}] {size_mb:.2f} MiB")
    for i, path in enumerate(entry['paths'], 1):
        print(f"    ({i}) {path}")
    print()
print('Run with --resolve to interactively remove duplicates.')
PY
  exit 0
fi

python3 - "$TMP_JSON" "$LIMIT" <<'PY'
import json
import os
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)
limit = int(sys.argv[2])
if limit > 0:
    data = data[:limit]

if not data:
    print('[duplicates] no duplicate sets to resolve')
    raise SystemExit(0)

print('[duplicates] interactive resolution mode')
print('For each set choose file numbers to delete (comma separated) or press Enter to skip.')
print('Type q to quit at any time. Files are deleted immediately after confirmation.')

for idx, entry in enumerate(data, 1):
    size_mb = entry['size'] / (1024 * 1024)
    print(f"\nSet {idx}: {size_mb:.2f} MiB duplicates")
    for i, path in enumerate(entry['paths'], 1):
        print(f"  ({i}) {path}")
    while True:
        resp = input('Select files to delete (e.g. 2 3) or press Enter to skip: ').strip()
        if resp.lower() == 'q':
            print('[duplicates] aborting at user request')
            raise SystemExit(0)
        if not resp:
            print('  skipping set')
            break
        try:
            indexes = sorted({int(part) for part in resp.replace(',', ' ').split()}, reverse=True)
        except ValueError:
            print('  invalid selection, try again')
            continue
        invalid = [i for i in indexes if i < 1 or i > len(entry['paths'])]
        if invalid:
            print(f'  invalid indexes: {invalid}')
            continue
        to_delete = [entry['paths'][i - 1] for i in indexes]
        print('  files selected for deletion:')
        for path in to_delete:
            print(f'    - {path}')
        confirm = input('  Confirm delete? (yes/no): ').strip().lower()
        if confirm not in {'y', 'yes'}:
            print('  deletion cancelled')
            continue
        for path in to_delete:
            try:
                os.remove(path)
                print(f'    removed {path}')
            except OSError as exc:
                print(f'    failed to remove {path}: {exc}')
        break
print('\n[duplicates] finished')
PY

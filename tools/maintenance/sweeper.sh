#!/usr/bin/env bash
set -euo pipefail
: "${OUT:?set OUT to a writable staging dir}"
#!/usr/bin/env bash
#
# sweeper.sh - archive large, old artifacts to an rclone remote.
# Defaults to a dry-run so you can review the plan before any bytes move.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sweeper.sh [options]

Archive files that are both older than the age threshold and larger than the
size threshold to the configured rclone remote.

Options:
  --run                 Execute the rclone move (defaults to dry-run).
  --source DIR          Source tree to scan (default: repository root).
  --remote NAME         rclone remote name (default: $SWEEPER_REMOTE or "archive").
  --remote-path PATH    Remote path inside the remote (default: $SWEEPER_REMOTE_PATH or "mca-gameday-camera/archive").
  --min-age DURATION    Minimum age filter passed to rclone (default: $SWEEPER_MIN_AGE or "7d").
  --min-size SIZE       Minimum size filter passed to rclone (default: $SWEEPER_MIN_SIZE or "200M").
  --log FILE            Optional log file for rclone output.
  -h, --help            Show this help message.

Environment:
  SWEEPER_REMOTE        Override the default remote name.
  SWEEPER_REMOTE_PATH   Override the default remote path.
  SWEEPER_MIN_AGE       Override the default minimum age (e.g. "14d").
  SWEEPER_MIN_SIZE      Override the default minimum size (e.g. "500M").
  SWEEPER_SOURCE        Override the default source directory.
USAGE
}

command -v rclone >/dev/null 2>&1 || {
  echo "[sweeper] rclone is required but was not found in PATH" >&2
  exit 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

DRY_RUN=1
SOURCE_DIR=${SWEEPER_SOURCE:-$REPO_ROOT}
REMOTE_NAME=${SWEEPER_REMOTE:-archive}
REMOTE_PATH=${SWEEPER_REMOTE_PATH:-mca-gameday-camera/archive}
MIN_AGE=${SWEEPER_MIN_AGE:-7d}
MIN_SIZE=${SWEEPER_MIN_SIZE:-200M}
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      DRY_RUN=0
      shift
      ;;
    --source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE_NAME="$2"
      shift 2
      ;;
    --remote-path)
      REMOTE_PATH="$2"
      shift 2
      ;;
    --min-age)
      MIN_AGE="$2"
      shift 2
      ;;
    --min-size)
      MIN_SIZE="$2"
      shift 2
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[sweeper] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[sweeper] source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

SOURCE_DIR=$(cd "$SOURCE_DIR" && pwd)
TARGET="${REMOTE_NAME}:${REMOTE_PATH}"

EXCLUDES=(
  "--exclude" ".git/**"
  "--exclude" "*/__pycache__/**"
  "--exclude" "**/.ipynb_checkpoints/**"
  "--exclude" "tools/maintenance/**"
)

ARGS=(
  "move"
  "$SOURCE_DIR"
  "$TARGET"
  "--min-age" "$MIN_AGE"
  "--min-size" "$MIN_SIZE"
  "--transfers" "4"
  "--checkers" "4"
  "--verbose"
  "--create-empty-src-dirs"
)

if (( DRY_RUN )); then
  ARGS+=("--dry-run")
fi

ARGS+=("--progress" "--stats" "5s")

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  ARGS+=("--log-file" "$LOG_FILE" "--log-format" "DATE,TIME,MESSAGE")
fi

printf '[sweeper] Source      : %s\n' "$SOURCE_DIR"
printf '[sweeper] Remote      : %s\n' "$TARGET"
printf '[sweeper] Min age     : %s\n' "$MIN_AGE"
printf '[sweeper] Min size    : %s\n' "$MIN_SIZE"
printf '[sweeper] Mode        : %s\n' "$([[ $DRY_RUN -eq 1 ]] && echo "dry-run" || echo "live")"

if (( DRY_RUN )); then
  echo "[sweeper] This is a dry-run. Use --run to execute the move."
fi

echo "[sweeper] Launching rclone..."
rclone "${ARGS[@]}" "${EXCLUDES[@]}"

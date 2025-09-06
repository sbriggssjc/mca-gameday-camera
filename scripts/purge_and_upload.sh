#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

VERSION="1.0.0"
SCRIPT_NAME="$(basename "$0")"
ORIG_ARGS=("$@")

# Defaults
RECORDINGS_DIR="${RECORDINGS_DIR:-$(pwd)/recordings/raw}"
EXTENSIONS=(mkv ts)
DRY_RUN=false
UPLOAD=false
NO_ARCHIVE=false
DAYS=""
KEEP=""
MAX_SIZE=""
REMOTE="${GDRIVE_REMOTE:-gdrive:}"
DEST="${GDRIVE_FOLDER:-mca-gameday-camera/backups}"
YES=false

usage() {
  cat <<USAGE
Usage: $SCRIPT_NAME [options]
  --dry-run             simulate actions, do not delete files
  --days N              select files older than N days
  --keep N              keep N newest per extension, purge older
  --max-size GB         ensure directory size <= GB
  --extensions "mkv ts" override extensions list
  --dir PATH            recordings directory (default: $RECORDINGS_DIR)
  --upload              upload archive and report via rclone
  --remote REMOTE       rclone remote (default: $REMOTE)
  --dest PATH           remote folder (default: $DEST)
  --no-archive          skip tarball creation
  --yes                 non-interactive; proceed without prompts
  --version             show version
  -h|--help             this help
USAGE
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --days) DAYS="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --max-size) MAX_SIZE="$2"; shift 2 ;;
    --extensions) IFS=' ' read -r -a EXTENSIONS <<< "$2"; shift 2 ;;
    --dir) RECORDINGS_DIR="$2"; shift 2 ;;
    --upload) UPLOAD=true; shift ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --no-archive) NO_ARCHIVE=true; shift ;;
    --yes) YES=true; shift ;;
    --version) echo "$SCRIPT_NAME v$VERSION"; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$RECORDINGS_DIR"
REPORT_DIR="logs/maintenance"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="$REPORT_DIR/report-$TIMESTAMP.txt"

# Colors
if [[ -t 1 ]]; then
  GREEN=$(tput setaf 2)
  BLUE=$(tput setaf 4)
  YELLOW=$(tput setaf 3)
  RED=$(tput setaf 1)
  RESET=$(tput sgr0)
else
  GREEN=""; BLUE=""; YELLOW=""; RED=""; RESET=""
fi

log(){
  local tag="$1"; shift
  local msg="$*"
  local color=""
  case "$tag" in
    purge) color="$GREEN";;
    upload) color="$BLUE";;
    report) color="$YELLOW";;
    warn) color="$RED";;
  esac
  if [[ -t 1 ]]; then
    printf "%b[%s]%b %s\n" "$color" "$tag" "$RESET" "$msg"
  else
    printf "[%s] %s\n" "$tag" "$msg"
  fi | tee -a "$REPORT_FILE"
}

human(){
  numfmt --to=iec --suffix=B --format="%7.2f" "$1"
}

# Build find args for extensions
build_find_args(){
  FIND_ARGS=()
  if (( ${#EXTENSIONS[@]} > 0 )); then
    FIND_ARGS+=( \( )
    for ((i=0;i<${#EXTENSIONS[@]};i++)); do
      FIND_ARGS+=(-name "*.${EXTENSIONS[i]}")
      if (( i < ${#EXTENSIONS[@]}-1 )); then FIND_ARGS+=(-o); fi
    done
    FIND_ARGS+=( \) )
  fi
}

build_find_args

# Header
log report "Report time: $(date)"
log report "Version: $SCRIPT_NAME v$VERSION"
log report "CLI: $SCRIPT_NAME $(printf '%q ' "${ORIG_ARGS[@]}")"

# Candidate collection
declare -a CANDIDATES
declare -A REASONS

add_candidate(){
  local f="$1" r="$2"
  if [[ -v REASONS["$f"] ]]; then
    REASONS["$f"]+=",$r"
  else
    CANDIDATES+=("$f")
    REASONS["$f"]="$r"
  fi
}

select_by_days(){
  [[ -n "$DAYS" ]] || return
  log purge "Selecting files older than $DAYS days"
  find "$RECORDINGS_DIR" -type f "${FIND_ARGS[@]}" -mtime +"$DAYS" -print0 | while IFS= read -r -d '' f; do
    add_candidate "$f" "days>$DAYS"
  done
}

select_by_keep(){
  [[ -n "$KEEP" ]] || return
  log purge "Selecting files exceeding keep $KEEP"
  for ext in "${EXTENSIONS[@]}"; do
    mapfile -d '' -t entries < <(find "$RECORDINGS_DIR" -type f -name "*.${ext}" -printf '%T@ %p\0' | sort -z -n)
    local total=${#entries[@]}
    local cutoff=$(( total - KEEP ))
    if (( cutoff > 0 )); then
      for ((i=0;i<cutoff;i++)); do
        file="${entries[i]}"
        path="${file#* }"
        add_candidate "$path" "keep>$KEEP"
      done
    fi
  done
}

select_by_max_size(){
  [[ -n "$MAX_SIZE" ]] || return
  local cap=$(( MAX_SIZE * 1024 * 1024 * 1024 ))
  local total=$(du -sb "$RECORDINGS_DIR" | awk '{print $1}')
  if (( total <= cap )); then return; fi
  log purge "Directory size $(human $total) exceeds cap $(human $cap)"
  mapfile -d '' -t entries < <(find "$RECORDINGS_DIR" -type f "${FIND_ARGS[@]}" -printf '%T@ %s %p\0' | sort -z -n)
  local current=$total
  for entry in "${entries[@]}"; do
    (( current <= cap )) && break
    ts=${entry%% *}
    rest=${entry#* }
    size=${rest%% *}
    path=${rest#* }
    add_candidate "$path" "max>$MAX_SIZE"
    current=$(( current - size ))
  done
}

select_by_days
select_by_keep
select_by_max_size

# Final candidate list and open file check
command -v lsof >/dev/null 2>&1 || log warn "lsof not found; open file check skipped"
declare -a FINAL
TOTAL_CAND_SIZE=0
declare -A SEEN
for f in "${CANDIDATES[@]}"; do
  [[ -n "${SEEN[$f]:-}" ]] && continue
  SEEN[$f]=1
  if command -v lsof >/dev/null 2>&1 && lsof "$f" >/dev/null 2>&1; then
    log warn "Skipping open file $f"
    continue
  fi
  FINAL+=("$f")
  size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  TOTAL_CAND_SIZE=$((TOTAL_CAND_SIZE + size))
  log purge "candidate $(human $size) $f (reason: ${REASONS[$f]})"

done
log report "Total candidates: ${#FINAL[@]} files, $(human $TOTAL_CAND_SIZE)"

# Analysis report
log report "=== Disk Usage ==="
df -h "$RECORDINGS_DIR" | tee -a "$REPORT_FILE"
df -ih "$RECORDINGS_DIR" | tee -a "$REPORT_FILE"
for ext in "${EXTENSIONS[@]}"; do
  cnt=$(find "$RECORDINGS_DIR" -type f -name "*.${ext}" | wc -l)
  sz=$(find "$RECORDINGS_DIR" -type f -name "*.${ext}" -print0 | \
     du --files0-from=- -ch 2>/dev/null | tail -n1 | awk '{print $1}')
  log report "Ext .$ext count=$cnt size=$sz"
done
log report "Top 20 largest files:"
find "$RECORDINGS_DIR" -type f "${FIND_ARGS[@]}" -printf '%s\t%p\n' | sort -nr | head -n 20 | while IFS=$'\t' read -r s p; do printf "%s\t%s\n" "$(numfmt --to=iec $s)" "$p"; done | tee -a "$REPORT_FILE"
log report "Count and size per day (last 30 days):"
for i in $(seq 0 29); do
  start=$(date -d "$i days ago" +%Y-%m-%d)
  end=$(date -d "$((i-1)) days ago" +%Y-%m-%d)
  c=$(find "$RECORDINGS_DIR" -type f "${FIND_ARGS[@]}" -newermt "$start" ! -newermt "$end" | wc -l)
  s=$(find "$RECORDINGS_DIR" -type f "${FIND_ARGS[@]}" -newermt "$start" ! -newermt "$end" -print0 | du --files0-from=- -ch 2>/dev/null | tail -n1 | awk '{print $1}')
  log report "$start count=$c size=$s"
done

# Archive and upload
TAR_FILE="/tmp/gameday-purge-$TIMESTAMP.tgz"
MANIFEST="/tmp/gameday-manifest-$TIMESTAMP.txt"
UPLOAD_PATH=""
if (( ${#FINAL[@]} > 0 )) && ! $NO_ARCHIVE; then
  if $DRY_RUN; then
    log purge "[dry-run] would create archive $TAR_FILE"
  else
    > "$MANIFEST"
    FILELIST=$(mktemp)
    for f in "${FINAL[@]}"; do
      size=$(stat -c%s "$f")
      sha=$(sha256sum "$f" | awk '{print $1}')
      printf "%s\t%s\t%s\n" "$size" "$sha" "$f" >> "$MANIFEST"
      printf '%s\0' "$f" >> "$FILELIST"
    done
    tar --null -czf "$TAR_FILE" --files-from "$FILELIST"
    rm -f "$FILELIST"
    log purge "Created archive $TAR_FILE"
  fi
fi

if $UPLOAD; then
  if ! command -v rclone >/dev/null 2>&1; then
    log warn "rclone not found. Install with: sudo apt-get update && sudo apt-get install -y rclone"
    exit 1
  fi
  year=$(date +%Y); month=$(date +%m); day=$(date +%d)
  UPLOAD_PATH="${REMOTE}${DEST}/${year}/${month}/${day}/"
  log upload "Uploading to $UPLOAD_PATH"
  if ! $DRY_RUN; then
    rclone copy "$REPORT_FILE" "$UPLOAD_PATH" || { log upload "Report upload failed"; exit 1; }
    if (( ${#FINAL[@]} > 0 )) && ! $NO_ARCHIVE; then
      rclone copy "$TAR_FILE" "$UPLOAD_PATH" || { log upload "Archive upload failed"; exit 1; }
    fi
  else
    log upload "[dry-run] skipping upload"
  fi
fi

# Delete files
PURGED_COUNT=0
FREED_BYTES=0
if (( ${#FINAL[@]} > 0 )); then
  if $DRY_RUN; then
    log purge "[dry-run] would delete ${#FINAL[@]} files"
  else
    for f in "${FINAL[@]}"; do
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      rm -f "$f"
      FREED_BYTES=$((FREED_BYTES + sz))
      ((PURGED_COUNT++))
    done
    log purge "Deleted $PURGED_COUNT files"
  fi
fi

summary="purged $PURGED_COUNT files / freed $(human $FREED_BYTES) / report $REPORT_FILE"
if [[ -n "$UPLOAD_PATH" ]]; then
  summary+=" / uploaded to $UPLOAD_PATH"
else
  summary+=" / no upload"
fi
log report "Summary: $summary"
exit 0

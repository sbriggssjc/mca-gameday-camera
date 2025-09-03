#!/bin/bash
set -euo pipefail

# Ensure required ignore patterns exist
GITIGNORE=".gitignore"
for pattern in recordings/ video/ '*.mkv' '*.mp4'; do
    if ! grep -Fxq "$pattern" "$GITIGNORE"; then
        echo "$pattern" >> "$GITIGNORE"
    fi
done

# Optionally reset filter-repo marker
if [[ "${1:-}" == "--reset-filter-repo" ]]; then
    rm -f .git/filter-repo/already_ran
fi

# Re-add origin remote if missing
if ! git remote | grep -q '^origin$'; then
    if [[ -n "${ORIGIN_URL:-}" ]]; then
        git remote add origin "$ORIGIN_URL"
    elif [[ -f .git/origin_url ]]; then
        git remote add origin "$(cat .git/origin_url)"
    else
        echo "origin remote missing and ORIGIN_URL not set" >&2
    fi
fi

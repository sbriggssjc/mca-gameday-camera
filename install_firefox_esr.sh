#!/usr/bin/env bash
# Download and extract the latest Firefox ESR for Jetson Nano (ARM64)

set -euo pipefail

BASE_URL="https://ftp.mozilla.org/pub/firefox/releases/"

log() {
    echo -e "$1"
}

log "🔍 Fetching Firefox ESR versions..."
if ! html=$(curl -fsSL "$BASE_URL"); then
    log "❌ Failed to fetch release directory."
    exit 1
fi

# Extract version directories that end in 'esr/'
versions=$(echo "$html" | grep -oE '<a href="[0-9]+[^"]*esr/">' | awk -F'"' '{print $2}' | cut -d'/' -f1 | sort -V)
latest=$(echo "$versions" | tail -n1)

if [[ -z "$latest" ]]; then
    log "❌ No ESR versions found."
    exit 1
fi

log "✅ Latest ESR version detected: $latest"
archive_url="${BASE_URL}${latest}/linux-aarch64/en-US/firefox-${latest}.tar.bz2"
archive="firefox-esr.tar.bz2"

log "📦 Downloading Firefox archive..."
if ! wget -O "$archive" "$archive_url" >/dev/null 2>&1; then
    log "❌ Download failed or URL not found."
    exit 1
fi

log "📦 Extracting $archive..."
[ -d firefox ] && rm -rf firefox
if ! tar -xjf "$archive"; then
    log "❌ Extraction failed."
    rm -f "$archive"
    exit 1
fi
rm -f "$archive"

log "🚀 Firefox ESR ready to launch at ./firefox/firefox"

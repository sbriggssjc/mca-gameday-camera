#!/usr/bin/env bash
# Download and extract the latest Firefox ESR for Jetson Nano (ARM64)

set -euo pipefail

BASE_URL="https://ftp.mozilla.org/pub/firefox/releases/"

log() {
    echo -e "$1"
}

log "🔍 Fetching Firefox ESR versions..."
if ! html=$(curl -fsSL "$BASE_URL"); then
    log "❌ Failed to fetch release index."
    exit 1
fi

releases=$(echo "$html" | grep -oE 'href="[0-9]+(\.[0-9]+)*esr/' | sed 's#href="##;s#/##' | sort -V)
latest=$(echo "$releases" | tail -n1)

if [[ -z "$latest" ]]; then
    log "❌ No ESR versions found."
    exit 1
fi

log "✅ Latest ESR version detected: $latest"
archive="firefox-${latest}.tar.bz2"
url="${BASE_URL}${latest}/linux-aarch64/en-US/${archive}"
rm -f "$archive"

log "📦 Downloading $archive..."
if ! wget -q "$url"; then
    log "❌ Download failed."
    exit 1
fi

log "📦 Extracting $archive..."
# Remove old installation if present
[ -d firefox ] && rm -rf firefox
if ! tar -xjf "$archive"; then
    log "❌ Extraction failed."
    rm -f "$archive"
    exit 1
fi
rm -f "$archive"

version_no_suffix=${latest%esr}
log "🚀 Firefox ESR ${version_no_suffix} is ready to run at ./firefox/firefox"

#!/usr/bin/env bash
# Download and extract the latest Firefox ESR for Jetson Nano (ARM64)

set -euo pipefail

BASE_URL="https://ftp.mozilla.org/pub/firefox/releases/"
ARCH="linux-aarch64"
LANG="en-US"

log() {
    echo -e "$1"
}

log "🔍 Fetching Firefox ESR versions..."
if ! RELEASES=$(curl -fsSL "$BASE_URL"); then
    log "❌ Failed to fetch release directory."
    exit 1
fi

log "📄 Raw directory listing:"
echo "$RELEASES" | head -n 20

# Parse ESR version directories
VERSIONS=$(echo "$RELEASES" | grep -oP '(?<=href="/pub/firefox/releases/)[0-9]+\.[0-9]+(\.[0-9]+)?esr/' | sort -V)
log "📋 Parsed ESR versions:"
echo "$VERSIONS"

if [[ -z "$VERSIONS" ]]; then
    log "❌ No ESR versions found. Parsing failed."
    exit 1
fi

LATEST=$(echo "$VERSIONS" | tail -n 1 | tr -d '/')
filename="firefox-${LATEST}.tar.bz2"
download_url="${BASE_URL}${LATEST}/${ARCH}/${LANG}/${filename}"

log "✅ Latest ESR version detected: $LATEST"
log "🌐 Downloading: $download_url"

if ! wget -O "$filename" "$download_url"; then
    log "❌ Download failed."
    exit 1
fi

log "📦 Extracting..."
[ -d firefox ] && rm -rf firefox
if ! tar -xjf "$filename"; then
    log "❌ Extraction failed."
    rm -f "$filename"
    exit 1
fi
rm -f "$filename"

log "🚀 Firefox ESR $LATEST is ready to run at ./firefox/firefox"

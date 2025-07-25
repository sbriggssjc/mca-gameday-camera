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
if ! html=$(curl -fsSL "$BASE_URL"); then
    log "❌ Failed to fetch release directory."
    exit 1
fi

log "📄 Raw directory listing:"
echo "$html" | head -n 20

# Parse ESR version directories
versions=$(echo "$html" | grep -oP '(?<=href=")[0-9]+\.[0-9]+(\.[0-9]+)?esr/')
versions=$(echo "$versions" | sort -V)
log "📋 Parsed ESR versions:"
echo "$versions"

if [[ -z "$versions" ]]; then
    log "❌ No ESR versions found. Parsing failed."
    exit 1
fi

latest=$(echo "$versions" | tail -n 1 | tr -d '/')
filename="firefox-${latest}.tar.bz2"
download_url="${BASE_URL}${latest}/${ARCH}/${LANG}/${filename}"

log "✅ Latest ESR version detected: $latest"
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

log "🚀 Firefox ESR $latest is ready to run at ./firefox/firefox"

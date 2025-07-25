#!/usr/bin/env bash
# Download and extract the latest Firefox ESR for Jetson Nano (ARM64)

set -e

BASE_URL="https://ftp.mozilla.org/pub/firefox/releases/"

log() {
    echo -e "$1"
}

# Fetch ESR versions
log "🔍 Checking Firefox ESR versions..."
releases=$(wget -qO- "$BASE_URL" | grep -oE 'href="[0-9]+(\.[0-9]+)*esr/' | sed 's/href="//;s#/##' | sort -V)

latest=$(echo "$releases" | tail -n1)

if [ -z "$latest" ]; then
    log "❌ No ESR versions found."
    exit 1
fi

log "✅ Latest ESR version detected: $latest"

archive="firefox-${latest}.tar.bz2"
url="${BASE_URL}${latest}/linux-aarch64/en-US/${archive}"

log "📦 Downloading ${archive}..."
if ! wget -q "$url"; then
    log "❌ Download failed."
    exit 1
fi

log "📦 Extracting ${archive}..."
if ! tar -xjf "$archive"; then
    log "❌ Extraction failed."
    rm -f "$archive"
    exit 1
fi

rm -f "$archive"

firefox_path="$(pwd)/firefox/firefox"
log "✅ Firefox ${latest} installed."
log "🚀 Launch with: $firefox_path"

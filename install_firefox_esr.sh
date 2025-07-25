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
echo "$RELEASES" > firefox_listing.html
echo "🧾 Full HTML saved to firefox_listing.html (view it with 'less' or open in text editor)"

# Parse ESR version directories
VERSIONS=$(echo "$RELEASES" | sed -n 's/.*href="\/pub\/firefox\/releases\/\([^"]*esr\)\/".*/\1/p' | sort -V)
log "📋 Parsed ESR versions:"
echo "$VERSIONS"

if [[ -z "$VERSIONS" ]]; then
    log "❌ No ESR versions found. Parsing failed."
    exit 1
fi

LATEST=$(echo "$VERSIONS" | tail -n 1 | tr -d "/")
FILENAME="firefox-${LATEST}.tar.bz2"
DOWNLOAD_URL="https://ftp.mozilla.org/pub/firefox/releases/${LATEST}/linux-aarch64/en-US/${FILENAME}"
BASE_VERSION="${LATEST%esr}"

log "✅ Latest ESR version detected: $LATEST"
log "🌐 Downloading: $DOWNLOAD_URL"

if ! wget -O "$FILENAME" "$DOWNLOAD_URL"; then
    log "❌ Download failed."
    exit 1
fi

log "📦 Extracting..."
[ -d firefox ] && rm -rf firefox
if ! tar -xjf "$FILENAME"; then
    log "❌ Extraction failed."
    rm -f "$FILENAME"
    exit 1
fi
rm -f "$FILENAME"

log "🚀 Firefox ESR $BASE_VERSION is ready to run at ./firefox/firefox"

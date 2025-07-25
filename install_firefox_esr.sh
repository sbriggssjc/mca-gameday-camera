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

log "🧪 Trying ESR versions from newest to oldest..."

for VERSION in $(echo "$VERSIONS" | grep -E '^[0-9]+\.[0-9]+(\.[0-9]+)?esr$' | sort -Vr); do
    FILENAME="firefox-${VERSION}.tar.bz2"
    DOWNLOAD_URL="${BASE_URL}${VERSION}/linux-aarch64/en-US/${FILENAME}"
    log "🔎 Testing: $DOWNLOAD_URL"

    if wget --spider "$DOWNLOAD_URL" 2>/dev/null; then
        log "✅ Found working version: $VERSION"
        log "🌐 Downloading: $DOWNLOAD_URL"
        wget -O "$FILENAME" "$DOWNLOAD_URL"
        LATEST="$VERSION"
        break
    fi
done

if [ ! -f "$FILENAME" ]; then
    log "❌ No compatible ARM64 ESR version found."
    exit 1
fi

BASE_VERSION="${LATEST%esr}"

log "📦 Extracting..."
[ -d firefox ] && rm -rf firefox
if ! tar -xjf "$FILENAME"; then
    log "❌ Extraction failed."
    rm -f "$FILENAME"
    exit 1
fi
rm -f "$FILENAME"

log "🚀 Firefox ESR $BASE_VERSION is ready to run at ./firefox/firefox"

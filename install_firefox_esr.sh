#!/usr/bin/env bash
# Download and extract the latest Firefox ESR using Mozilla's redirect service

set -euo pipefail

URL="https://download.mozilla.org/?product=firefox-esr-latest&os=linux64&lang=en-US"
ARCHIVE="firefox-latest-esr.tar.bz2"
TARGET_DIR="firefox-esr"

log() {
    echo -e "$1"
}

log "🌐 Downloading latest Firefox ESR..."
if ! curl -L "$URL" -o "$ARCHIVE"; then
    log "❌ Download failed."
    exit 1
fi

log "📦 Extracting..."
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
if ! tar -xjf "$ARCHIVE" -C "$TARGET_DIR" --strip-components=1; then
    log "❌ Extraction failed."
    rm -f "$ARCHIVE"
    exit 1
fi
rm -f "$ARCHIVE"

chmod +x "$TARGET_DIR/firefox"

LAUNCHER="$TARGET_DIR/run_firefox.sh"
cat <<'EOF' > "$LAUNCHER"
#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/firefox" "$@"
EOF
chmod +x "$LAUNCHER"

log "✅ Firefox ESR installed in ./$TARGET_DIR"
log "Launch it with ./$LAUNCHER"

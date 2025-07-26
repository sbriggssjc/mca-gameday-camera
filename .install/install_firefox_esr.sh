#!/usr/bin/env bash
set -euo pipefail

VERSION="115.11.0esr"
ARCHIVE=".install/firefox.tar.bz2"
URL="https://ftp.mozilla.org/pub/firefox/releases/115.11.0esr/linux-aarch64/en-US/firefox-115.11.0esr.tar.bz2"
INSTALL_DIR=".install/firefox-esr"

echo "Installing Firefox ESR ${VERSION}..."

echo "Preparing installation directory..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

echo "Downloading from ${URL}..."
if curl -fL -o "$ARCHIVE" "$URL"; then
    echo "Download completed."
else
    echo "❌ Failed to download Firefox." >&2
    exit 1
fi

echo "Extracting archive..."
if tar -xjf "$ARCHIVE" -C "$INSTALL_DIR" --strip-components=1; then
    echo "Extraction complete."
else
    echo "❌ Failed to extract archive." >&2
    rm -f "$ARCHIVE"
    exit 1
fi

rm -f "$ARCHIVE"
chmod +x "$INSTALL_DIR/firefox"

if [ -x "$INSTALL_DIR/firefox" ]; then
    echo "✅ Firefox ESR ${VERSION} installed."
else
    echo "❌ Firefox installation failed." >&2
    exit 1
fi

echo "Launching Firefox..."
"$INSTALL_DIR/firefox" &

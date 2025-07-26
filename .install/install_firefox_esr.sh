#!/usr/bin/env bash
set -euo pipefail

VERSION="115.10.0esr"
ARCHIVE=".install/firefox-${VERSION}.tar.bz2"
URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/firefox-${VERSION}.tar.bz2"
INSTALL_DIR=".install/firefox-esr"

echo "Installing Firefox ESR ${VERSION}..."

echo "Preparing installation directory..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

echo "Downloading from ${URL}..."
if ! curl -fL -o "$ARCHIVE" "$URL"; then
    echo "❌ Failed to download Firefox." >&2
    exit 1
fi

echo "Extracting archive..."
if ! tar -xjf "$ARCHIVE" --strip-components=1 -C "$INSTALL_DIR"; then
    echo "❌ Failed to extract archive." >&2
    rm -f "$ARCHIVE"
    exit 1
fi

rm -f "$ARCHIVE"

chmod +x "$INSTALL_DIR/firefox"

echo "✅ Firefox ESR ${VERSION} installed."
echo "Launching Firefox..."
"$INSTALL_DIR/firefox" &

#!/usr/bin/env bash

set -euo pipefail

# Desired Firefox ESR version
VERSION="115.10.0esr"

# Download URL and archive name
URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/firefox-${VERSION}.tar.bz2"
ARCHIVE="firefox-${VERSION}.tar.bz2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/firefox-esr"

mkdir -p "$INSTALL_DIR"

echo "Downloading Firefox ESR ${VERSION}..."
curl -L -o "$ARCHIVE" "$URL"

echo "Extracting archive..."
tar -xjf "$ARCHIVE" --strip-components=1 -C "$INSTALL_DIR"

rm -f "$ARCHIVE"

echo "✅ Firefox ESR ${VERSION} installed successfully"
echo "📂 Installed at: .install/firefox-esr/"
echo "👉 Launch it using: .install/firefox-esr/firefox"

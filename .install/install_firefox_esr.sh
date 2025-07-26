#!/bin/bash

set -e

VERSION="115.10.0esr"
ARCHIVE="firefox-${VERSION}.tar.gz"
URL="https://ftp.mozilla.org/pub/firefox/releases/115.10.0esr/linux-aarch64/en-US/firefox-115.10.0esr.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/firefox-esr"

mkdir -p "$INSTALL_DIR"

echo "Downloading Firefox ESR ${VERSION}..."
curl -LO "$URL"

echo "Extracting archive..."
tar -xzf "$ARCHIVE"

echo "Moving files to $INSTALL_DIR..."
mv firefox "$INSTALL_DIR/"
rm "$ARCHIVE"

echo "✅ Firefox ESR installed successfully"
echo "📂 Installed at: .install/firefox-esr/"
echo "🧪 Run it using: .install/firefox-esr/firefox"

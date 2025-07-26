#!/usr/bin/env bash
set -euo pipefail

VERSION="115.8.0esr"
URL="https://ftp.mozilla.org/pub/firefox/releases/115.8.0esr/linux-aarch64/en-US/firefox-115.8.0esr.tar.bz2"
ARCHIVE="firefox-${VERSION}.tar.bz2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/firefox-esr"

mkdir -p "$INSTALL_DIR"

echo "Downloading Firefox ESR ${VERSION}..."
if ! curl -L -o "$ARCHIVE" "$URL"; then
    echo "Error: failed to download ${URL}" >&2
    exit 1
fi

echo "Validating archive..."
if ! tar -tjf "$ARCHIVE" > /dev/null 2>&1; then
    echo "Error: downloaded file is not a valid tar archive" >&2
    rm -f "$ARCHIVE"
    exit 1
fi

echo "Extracting archive..."
tar -xjf "$ARCHIVE" --strip-components=1 -C "$INSTALL_DIR"
rm -f "$ARCHIVE"

echo "Firefox ESR installed to .install/firefox-esr/"

#!/usr/bin/env bash
set -euo pipefail

VERSION="115.11.0esr"
URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/firefox-${VERSION}.tar.bz2"
ARCHIVE="$(basename "$URL")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${SCRIPT_DIR}/firefox-esr"

mkdir -p "$INSTALL_DIR"

echo "Downloading Firefox ESR ${VERSION}..."
if ! curl -fL -o "$ARCHIVE" "$URL"; then
    echo "Error: download failed" >&2
    exit 1
fi

if [ ! -s "$ARCHIVE" ] || [ $(stat -c%s "$ARCHIVE") -le 10485760 ]; then
    echo "Error: downloaded file is too small or invalid" >&2
    rm -f "$ARCHIVE"
    exit 1
fi

echo "Validating archive..."
if ! tar -tjf "$ARCHIVE" > /dev/null 2>&1; then
    echo "Error: downloaded file is not a valid tar archive" >&2
    rm -f "$ARCHIVE"
    exit 1
fi

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

echo "Extracting archive..."
tar -xjf "$ARCHIVE" --strip-components=1 -C "$INSTALL_DIR"
rm -f "$ARCHIVE"

echo "Firefox ESR ${VERSION} installed successfully in ${INSTALL_DIR}."
echo "Launch it with: ${INSTALL_DIR}/firefox &"

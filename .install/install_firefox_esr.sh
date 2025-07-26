#!/usr/bin/env bash
set -euo pipefail

# Firefox ESR version and source URL
ARCHIVE_NAME="firefox-115.3.1esr.tar.bz2"
URL="https://ftp.mozilla.org/pub/firefox/releases/115.3.1esr/linux-aarch64/en-US/${ARCHIVE_NAME}"
INSTALL_DIR=".install/firefox-esr"

# Ensure installation directory exists
mkdir -p "$INSTALL_DIR"

# Download the archive to .install/
echo "Downloading ${URL}..."
if ! curl -fL -o ".install/${ARCHIVE_NAME}" "$URL"; then
    echo "Error: failed to download ${URL}" >&2
    exit 1
fi

# Extract the archive into the installation directory
echo "Extracting ${ARCHIVE_NAME}..."
if ! tar -xjf ".install/${ARCHIVE_NAME}" --strip-components=1 -C "$INSTALL_DIR"; then
    echo "Error: extraction failed" >&2
    rm -f ".install/${ARCHIVE_NAME}"
    exit 1
fi

# Remove the archive after successful extraction
rm -f ".install/${ARCHIVE_NAME}"

# Ensure the Firefox binary is executable
chmod +x "$INSTALL_DIR/firefox"

# Success message
echo "✅ Firefox installed. Run it with .install/firefox-esr/firefox &"

#!/usr/bin/env bash
set -euo pipefail

VERSION="115.11.0esr"
URL="https://ftp.mozilla.org/pub/firefox/releases/115.11.0esr/linux-aarch64/en-US/firefox-115.11.0esr.tar.bz2"
ARCHIVE="firefox-${VERSION}.tar.bz2"
TARGET_DIR=".install/firefox-esr"

# Create the installation directory
printf 'Creating %s...\n' "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR"

# Download the Firefox archive
printf 'Downloading Firefox ESR %s...\n' "$VERSION"
if ! curl -LO "$URL"; then
    echo "Error: failed to download archive" >&2
    exit 1
fi

# Validate the archive type
printf 'Validating archive...\n'
if ! file "$ARCHIVE" | grep -q 'bzip2 compressed data'; then
    echo "Error: downloaded file is not a valid bzip2 archive" >&2
    rm -f "$ARCHIVE"
    exit 1
fi

# Extract the archive
printf 'Extracting archive...\n'
tar -xjf "$ARCHIVE"

# Remove the archive
printf 'Cleaning up...\n'
rm -f "$ARCHIVE"

cd - >/dev/null

# Create symlink to the Firefox binary
printf 'Creating symlink firefox-esr -> %s/firefox/firefox\n' "$TARGET_DIR"
ln -sf "$TARGET_DIR/firefox/firefox" firefox-esr

printf '✅ Firefox ESR %s installed successfully.\n' "$VERSION"
printf '👉 Launch with ./firefox-esr\n'

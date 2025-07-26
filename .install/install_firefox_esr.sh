#!/usr/bin/env bash
set -euo pipefail

VERSION="115.11.0esr"
URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/firefox-${VERSION}.tar.bz2"
ARCHIVE="firefox-${VERSION}.tar.bz2"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/firefox-esr"

# Ensure the download exists before attempting to fetch it
printf 'Checking availability of %s...\n' "$URL"
if ! curl -sfI "$URL" > /dev/null; then
    echo "Error: Firefox ESR ${VERSION} not found at ${URL}" >&2
    exit 1
fi

# Download the archive
printf 'Downloading %s...\n' "$ARCHIVE"
if ! curl -fL -o "$ARCHIVE" "$URL"; then
    echo "Error: failed to download ${URL}" >&2
    exit 1
fi

# Verify the archive is valid
printf 'Verifying archive...\n'
if ! tar -tjf "$ARCHIVE" > /dev/null 2>&1; then
    echo "Error: downloaded file is not a valid tar.bz2 archive" >&2
    rm -f "$ARCHIVE"
    exit 1
fi

# Prepare installation directory
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Extract the archive
printf 'Extracting to %s...\n' "$INSTALL_DIR"
if ! tar -xjf "$ARCHIVE" --strip-components=1 -C "$INSTALL_DIR"; then
    echo "Error: extraction failed" >&2
    rm -f "$ARCHIVE"
    exit 1
fi

# Remove the archive
rm -f "$ARCHIVE"

# Ensure the Firefox binary is executable
chmod +x "$INSTALL_DIR/firefox"

printf '✅ Firefox ESR %s installed. Launch with `%s/firefox &`\n' "$VERSION" "$INSTALL_DIR"

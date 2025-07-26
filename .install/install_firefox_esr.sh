#!/bin/bash

set -e

# Hardcoded Firefox ESR version
VERSION="115.10.0esr"
# Archive filename for the version above
TARBALL="firefox-${VERSION}.tar.bz2"
# Exact download URL for the archive
DOWNLOAD_URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/firefox-${VERSION}.tar.bz2"
INSTALL_DIR=".install/firefox-esr"

echo "Downloading Firefox ESR ${VERSION}..."
echo "URL: ${DOWNLOAD_URL}"

# Create install directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Download the archive
cd "$INSTALL_DIR"
curl -L --fail --silent --show-error -o "$TARBALL" "$DOWNLOAD_URL"

# Validate the download (larger than 1MB)
if [ ! -s "$TARBALL" ] || [ $(stat -c%s "$TARBALL") -lt 1048576 ]; then
    echo "Error: Downloaded file looks invalid. Aborting."
    rm -f "$TARBALL"
    exit 1
fi

# Extract and clean up
tar -xjf "$TARBALL"
rm "$TARBALL"

echo "✅ Firefox ESR ${VERSION} installed."
echo "👉 Launch with ${INSTALL_DIR}/firefox/firefox"

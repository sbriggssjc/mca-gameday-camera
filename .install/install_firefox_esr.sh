#!/bin/bash

set -e

# Hardcode the desired Firefox ESR version instead of scraping Mozilla's site
VERSION="115.11.0esr"
TARBALL="firefox-${VERSION}.tar.bz2"
# Construct the download URL directly from the version
DOWNLOAD_URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/${TARBALL}"
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

# Create symlink to the Firefox binary
cd ../..
ln -sf "$INSTALL_DIR/firefox/firefox" firefox-esr

echo "✅ Firefox ESR $VERSION installed successfully."
echo "👉 Run with: ./firefox-esr"

#!/bin/bash

set -e

VERSION="115.3.1esr"
TARBALL="firefox-${VERSION}.tar.bz2"
DOWNLOAD_URL="https://ftp.mozilla.org/pub/firefox/releases/${VERSION}/linux-aarch64/en-US/${TARBALL}"
INSTALL_DIR=".install/firefox-esr"

echo "Downloading Firefox ESR ${VERSION}..."
echo "URL: ${DOWNLOAD_URL}"

# Create install directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Download and extract
cd "$INSTALL_DIR"
curl -LO "$DOWNLOAD_URL"
tar -xjf "$TARBALL"
rm "$TARBALL"

# Optional symlink
cd ../..
ln -sf $INSTALL_DIR/firefox/firefox firefox-esr

echo "✅ Firefox ESR $VERSION installed successfully."
echo "👉 Run with: ./firefox-esr"

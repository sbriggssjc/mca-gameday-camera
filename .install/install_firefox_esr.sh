#!/bin/bash

set -e

echo "🔍 Looking for latest Firefox ESR version compatible with ARM64..."

# Use a known working version — Firefox ESR 115 is the latest known with ARM64 builds
ESR_VERSION="115.11.0esr"
ARCH="linux-aarch64"
LOCALE="en-US"
INSTALL_DIR=".install/firefox-esr"
TARBALL="firefox-${ESR_VERSION}.tar.bz2"
DOWNLOAD_URL="https://ftp.mozilla.org/pub/firefox/releases/${ESR_VERSION}/${ARCH}/${LOCALE}/${TARBALL}"

echo "🌐 Downloading from: $DOWNLOAD_URL"

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

echo "✅ Firefox ESR $ESR_VERSION installed successfully."
echo "👉 Run with: ./firefox-esr"

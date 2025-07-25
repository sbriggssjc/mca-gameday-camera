#!/usr/bin/env python3
"""Download and extract the latest Firefox ESR for ARM64 Linux."""
import os
import re
import sys
import tarfile
import shutil
import subprocess
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: The 'requests' package is required. Install it with 'pip install requests'.")
    sys.exit(1)

BASE_URL = "https://ftp.mozilla.org/pub/firefox/releases/"


def log(message: str) -> None:
    print(message)


def fetch_latest_esr() -> str:
    """Return the latest ESR version string, e.g. '115.13.0esr'."""
    log("Fetching Firefox releases...")
    try:
        resp = requests.get(BASE_URL, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log(f"Error fetching release index: {e}")
        sys.exit(1)

    entries = re.findall(r'href="([^"/]+/)"', resp.text)
    esr_versions = [e.strip('/') for e in entries if e.endswith('esr/') and re.match(r'^\d', e)]
    if not esr_versions:
        log("No ESR versions found.")
        sys.exit(1)

    def version_key(v: str):
        parts = v[:-3].split('.')
        return [int(p) for p in parts]

    esr_versions.sort(key=version_key)
    latest = esr_versions[-1]
    log(f"Detected ESR version: {latest}")
    return latest


def download_esr(version: str) -> Path:
    sub_url = f"{BASE_URL}{version}/linux-aarch64/en-US/"
    log(f"Checking {sub_url} for archive...")
    try:
        resp = requests.get(sub_url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log(f"Error fetching directory listing: {e}")
        sys.exit(1)

    pattern = rf'href="(firefox-{re.escape(version)}\.tar\.bz2)"'
    match = re.search(pattern, resp.text)
    if not match:
        log("Firefox archive not found.")
        sys.exit(1)

    filename = match.group(1)
    url = f"{sub_url}{filename}"
    dest = Path.home() / filename
    log(f"Downloading {url} to {dest} ...")
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        log(f"Error downloading file: {e}")
        sys.exit(1)

    log("Download complete.")
    return dest


def extract_archive(tar_path: Path, version: str) -> Path:
    extract_dir = Path.home() / f"firefox-{version}"
    log(f"Extracting to {extract_dir} ...")
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        with tarfile.open(tar_path, 'r:bz2') as tar:
            tar.extractall(extract_dir)
    except Exception as e:
        log(f"Error extracting archive: {e}")
        sys.exit(1)
    log("Extraction complete.")
    return extract_dir / "firefox"


def main():
    version = fetch_latest_esr()
    tarball = download_esr(version)
    firefox_dir = extract_archive(tarball, version)
    log("Firefox ready to run.")

    choice = input("Launch Firefox now? [y/N] ").strip().lower()
    if choice == 'y':
        firefox_path = firefox_dir / 'firefox'
        log(f"Launching {firefox_path} ...")
        try:
            subprocess.Popen([str(firefox_path)])
        except Exception as e:
            log(f"Failed to launch Firefox: {e}")
    else:
        log("Installation finished.")


if __name__ == "__main__":
    main()

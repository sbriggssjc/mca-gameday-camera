#!/usr/bin/env python3
"""Download and extract the latest Firefox ESR build for ARM64 Linux."""
import os
import sys
import tarfile
import shutil
from pathlib import Path
import json

try:
    import requests
except ImportError:
    print("Error: The 'requests' package is required. Install it with 'pip install requests'.")
    sys.exit(1)

VERSION = ""
ARCHIVE = Path()
URL = ""
TARGET_DIR = Path("firefox-esr")


def log(msg: str) -> None:
    print(msg)


def get_latest_version() -> str:
    """Return the latest Firefox ESR version string."""
    url = "https://product-details.mozilla.org/1.0/firefox_versions.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("FIREFOX_ESR", "").strip()
    except Exception as e:
        log(f"Error fetching version info: {e}")
        sys.exit(1)


def download() -> Path:
    log(f"Downloading Firefox ESR {VERSION}...")
    try:
        with requests.get(URL, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(ARCHIVE, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        log(f"Error downloading archive: {e}")
        sys.exit(1)
    return ARCHIVE


def extract(tar_path: Path) -> None:
    log("Extracting archive...")
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    TARGET_DIR.mkdir(parents=True)
    try:
        with tarfile.open(tar_path, 'r:bz2') as tar:
            members = []
            for m in tar.getmembers():
                parts = m.name.split('/', 1)
                if len(parts) == 2:
                    m.name = parts[1]
                else:
                    m.name = parts[0]
                members.append(m)
            tar.extractall(TARGET_DIR, members)
    except Exception as e:
        log(f"Error extracting archive: {e}")
        sys.exit(1)
    os.chmod(TARGET_DIR / "firefox", 0o755)

    launcher = TARGET_DIR / "run_firefox.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "exec \"$DIR/firefox\" \"$@\"\n"
    )
    launcher.chmod(0o755)


def main() -> None:
    global VERSION, ARCHIVE, URL
    VERSION = get_latest_version()
    if not VERSION:
        log("Could not determine latest ESR version")
        sys.exit(1)
    URL = (
        f"https://ftp.mozilla.org/pub/firefox/releases/{VERSION}/"
        f"linux-aarch64/en-US/firefox-{VERSION}.tar.bz2"
    )
    ARCHIVE = Path(f"firefox-{VERSION}.tar.bz2")

    tarball = download()
    extract(tarball)
    ARCHIVE.unlink(missing_ok=True)
    log(f"Firefox ESR {VERSION} installed in ./{TARGET_DIR}")
    log(f"Launch it with ./{TARGET_DIR}/run_firefox.sh")


if __name__ == "__main__":
    main()

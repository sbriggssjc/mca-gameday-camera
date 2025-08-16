import os
from pathlib import Path


def load_env(dotenv_path: str = ".env") -> None:
    p = Path(dotenv_path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def require(names):
    """Return the first set environment variable among ``names``.

    ``names`` may be a single variable name or an iterable of aliases. The
    first value found in the environment is returned. If none are set, the
    user is prompted for a value which is used for the current process only.
    """

    if isinstance(names, str):
        names = [names]

    for name in names:
        val = os.environ.get(name)
        if val:
            return val

    msg = (
        f"Required environment variable(s) {', '.join(names)} not set. "
        "Check your .env file."
    )
    print(msg)

    prompt = (
        "Enter YouTube RTMP URL: "
        if any(n in ("YT_RTMP_URL", "YOUTUBE_RTMP_URL") for n in names)
        else f"Enter value for {names[0]}: "
    )
    user_val = input(prompt).strip()
    if user_val:
        return user_val

    raise RuntimeError(msg + " Add it in .env or export it.")

import os
from pathlib import Path

def load_env(dotenv_path: str = ".env") -> None:
    p = Path(dotenv_path)
    if p.exists():
        for line in p.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: 
                continue
            k,v = line.split("=",1)
            os.environ.setdefault(k.strip(), v.strip())

def require(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise RuntimeError(f"Required environment variable {name} is not set. Add it in .env or export it.")
    return val

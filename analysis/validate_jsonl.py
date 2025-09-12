import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path("plays.jsonl")
for i, line in enumerate(p.read_text().splitlines(), 1):
    line = line.strip()
    if not line:
        print(f"[warn] blank line {i}")
        continue
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            print(f"[warn] line {i} is {type(obj).__name__}, not object")
    except Exception as e:
        print(f"[error] line {i} invalid JSON: {e}")

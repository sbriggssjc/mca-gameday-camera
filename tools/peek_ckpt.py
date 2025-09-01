import argparse, os, hashlib


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    p = os.path.abspath(args.path)
    print("[file]", p, os.path.getsize(p), "bytes", "sha256", sha(p))

    try:
        import torch
        obj = torch.load(p, map_location="cpu", weights_only=True)
        print("[torch.load weights_only] type:", type(obj))
        if isinstance(obj, dict):
            print("  keys:", sorted(list(obj.keys()))[:10])
    except Exception as e:
        print("[torch.load weights_only] failed:", e)

    try:
        import torch
        obj = torch.load(p, map_location="cpu", weights_only=False)
        print("[torch.load pickle] type:", type(obj))
        if isinstance(obj, dict):
            print("  keys:", sorted(list(obj.keys()))[:10])
    except Exception as e:
        print("[torch.load pickle] failed:", e)

    try:
        import torch
        ts = torch.jit.load(p, map_location="cpu")
        print("[torchscript] loaded OK:", ts.__class__)
    except Exception as e:
        print("[torchscript] failed:", e)

    try:
        from safetensors.torch import load_file as st_load
        sd = st_load(p)
        print("[safetensors] loaded OK; keys:", len(sd))
    except Exception as e:
        print("[safetensors] failed:", e)


if __name__ == "__main__":
    main()

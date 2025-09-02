import sys, torch, collections, os

def is_state_dict(o):
    return isinstance(o, (dict, collections.OrderedDict)) \
        and all(isinstance(k, str) for k in o.keys()) \
        and any(hasattr(v, "shape") for v in o.values())

def extract_state_dict(obj):
    if is_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("state_dict","model","net","ema","model_state","weights","params"):
            v = obj.get(k)
            if v is not None and is_state_dict(v):
                return v
    return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/convert_ckpt.py <input_checkpoint> <output_state_dict_pt>")
        sys.exit(2)
    inp, outp = sys.argv[1], sys.argv[2]
    if not os.path.exists(inp):
        raise FileNotFoundError(inp)
    try:
        obj = torch.load(inp, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch versions don't accept weights_only
        obj = torch.load(inp, map_location="cpu")

    sd = extract_state_dict(obj)
    if sd is None:
        top = list(obj.keys())[:20] if isinstance(obj, dict) else type(obj).__name__
        raise RuntimeError(f"Couldn't find a state_dict. Top-level: {top}")

    # strip DDP "module." if present
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

    torch.save(sd, outp)
    print(f"Saved state_dict with {len(sd)} tensors -> {outp}")

if __name__ == "__main__":
    main()

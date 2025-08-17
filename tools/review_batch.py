import argparse

from analysis.playbook_loader import load_playbook
from analysis.review_draw import draw_topk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="out_dir", required=True)
    ap.add_argument("--playbook", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--auto-draw", action="store_true")
    args = ap.parse_args()

    pb = load_playbook(args.playbook)
    if args.auto_draw:
        draw_topk(args.out_dir, pb, top_k=args.top_k)
    print("[review_batch] done")


if __name__ == "__main__":
    main()


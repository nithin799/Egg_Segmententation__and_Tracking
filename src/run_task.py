"""Run the final classical egg segmentation on the dev or eval split.

Usage (from the repo root):
  python src/run_task2.py --split dev
  python src/run_task2.py --split eval --final      # eval needs --final: run it ONCE
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import egg_seg as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--data-dir", default="egg.v1i.folder", help="folder with the Roboflow images")
    ap.add_argument("--gt-dir", default="outputs/task1/masks")
    ap.add_argument("--split-file", default="outputs/split.json")
    ap.add_argument("--out", default="outputs/task2")
    ap.add_argument("--final", action="store_true", help="required to touch the eval split")
    args = ap.parse_args()

    if args.split == "eval" and not args.final:
        sys.exit("Refusing to run on eval without --final (eval is used once, after the algorithm is locked).")

    dev, ev = E.load_split(args.split_file)
    names = dev if args.split == "dev" else ev
    paths = E.image_paths(args.data_dir)
    missing = [n for n in names if n not in paths]
    if missing:
        sys.exit(f"Images not found under {args.data_dir}: {missing[:5]} ...")

    pred_dir = os.path.join(args.out, f"pred_{args.split}")
    os.makedirs(pred_dir, exist_ok=True)

    rows = []
    for k, n in enumerate(names, 1):
        rgb = E.load_img(paths[n])
        preds = E.segment_eggs(rgb)
        E.save_instance_map(preds, rgb.shape[:2], os.path.join(pred_dir, f"{n}.png"))
        r = E.score_image(preds, E.load_gt(n, args.gt_dir))
        r["image"] = n
        rows.append(r)
        print(f"[{k}/{len(names)}] {n}  IoU={r['iou']:.3f}  eggs {r['n_pred']}/{r['n_gt']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"final_{args.split}_per_image.csv"), index=False)
    summary = E.summarize(df)
    summary["n_images"] = len(df)
    summary["empty_images"] = int((df.n_pred == 0).sum())
    json.dump(summary, open(os.path.join(args.out, f"final_{args.split}_summary.json"), "w"), indent=2)
    print("\n" + args.split.upper(), summary)


if __name__ == "__main__":
    main()
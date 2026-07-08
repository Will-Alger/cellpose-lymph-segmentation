"""
Run several segmentation methods on the same image and compare their outputs.

Keeps the baseline fixed and lets you A/B any change from the methods/ package:
  - prints a table (cells found, AP vs ground truth, runtime)
  - saves a side-by-side overlay figure so you can SEE the difference

Usage:
    python compare.py testdata/data/2D/gray_2D.png \
        --gt testdata/data/2D/gray_2D_cp4_gt_masks.png \
        --methods baseline global_norm
    python compare.py lowq/gray_2D_moderate.tif --gt testdata/data/2D/gray_2D_cp4_gt_masks.png
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.io import imread

from cellpose import models, metrics, utils
import methods as M


def score(gt, pred):
    """AP@0.5/0.75 and TP/FP/FN vs ground truth, or None if no GT."""
    if gt is None:
        return None
    ap, tp, fp, fn = metrics.average_precision(
        gt.astype(np.int32), pred.astype(np.int32), threshold=[0.5, 0.75])
    return dict(ap50=ap[0], ap75=ap[1], tp=int(tp[0]), fp=int(fp[0]), fn=int(fn[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--gt", default=None, help="Ground-truth label mask (enables AP scoring).")
    p.add_argument("--methods", nargs="*", default=list(M.METHODS),
                   help=f"Subset of: {list(M.METHODS)}")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--out", default=None, help="Output figure path.")
    args = p.parse_args()

    img = imread(args.image)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    if img.dtype != np.uint8:  # match segment.py's 8-bit assumption
        lo, hi = float(img.min()), float(img.max())
        img = (((img.astype(np.float64) - lo) / (hi - lo) * 255) if hi > lo
               else np.zeros_like(img, np.float64)).round().astype(np.uint8)

    gt = imread(args.gt).astype(np.int32) if args.gt else None
    n_gt = (len(np.unique(gt)) - 1) if gt is not None else None

    model = models.CellposeModel(gpu=not args.no_gpu)

    results = []
    for name in args.methods:
        if name not in M.METHODS:
            print(f"! unknown method '{name}', skipping"); continue
        t0 = time.time()
        mask = M.METHODS[name](img, model)
        dt = time.time() - t0
        results.append(dict(name=name, mask=mask,
                            n=len(np.unique(mask)) - 1,
                            secs=dt, sc=score(gt, mask)))

    # ---- table ----
    print(f"\nImage: {args.image}" + (f"   |   GT cells: {n_gt}" if n_gt is not None else ""))
    header = f"{'method':14s} {'cells':>6s}"
    if gt is not None:
        header += f" {'AP@0.5':>8s} {'AP@0.75':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s}"
    header += f" {'secs':>7s}"
    print(header); print("-" * len(header))
    for r in results:
        row = f"{r['name']:14s} {r['n']:6d}"
        if gt is not None and r['sc']:
            s = r['sc']
            row += f" {s['ap50']:8.3f} {s['ap75']:8.3f} {s['tp']:5d} {s['fp']:5d} {s['fn']:5d}"
        row += f" {r['secs']:7.1f}"
        print(row)

    # ---- overlays, laid out in a grid ----
    panels = ([("ground truth", gt)] if gt is not None else []) + \
             [(r['name'], r['mask']) for r in results]
    n = len(panels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        for o in utils.outlines_list(mask):
            ax.plot(o[:, 0], o[:, 1], lw=0.5, color="#ff4d4d")
        cells = len(np.unique(mask)) - 1
        sub = ""
        if title != "ground truth":
            r = next(x for x in results if x['name'] == title)
            if r['sc']:
                sub = f"  AP@.5={r['sc']['ap50']:.3f}"
        ax.set_title(f"{title} · {cells} cells{sub}", fontsize=12)
        ax.axis("off")
    for ax in axes[n:]:            # hide any unused grid cells
        ax.axis("off")
    base = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs("output", exist_ok=True)
    out = args.out or os.path.join("output", f"compare_{base}.png")
    fig.suptitle(f"{os.path.basename(args.image)}"
                 + (f"  (GT={n_gt})" if n_gt is not None else ""), fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved comparison figure -> {out}")


if __name__ == "__main__":
    main()

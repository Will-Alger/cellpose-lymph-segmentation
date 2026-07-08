"""
Run the whole improvement ladder and compare it end-to-end.

Runs baseline -> model1 -> ... -> modelN on every quality level of a degradation
suite, scores each against ground truth, and produces:
  - a matrix table (rows = models, cols = quality levels, cells = AP@0.5)
  - a progression line chart (one line per model across quality)
  - a final overlay figure (ground truth | baseline | best model)

Usage:
    python progression.py            # uses lowq/gray_2D_*.tif + gray_2D GT
    python progression.py --gt <gt> --suite "lowq/myimg_*.tif"
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.io import imread

from cellpose import models, metrics, utils
import methods as M

LEVELS = ["clean", "mild", "moderate", "severe", "extreme"]


def load8(path):
    img = imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float64) - lo) / (hi - lo) * 255 if hi > lo
               else np.zeros_like(img, np.float64)).round().astype(np.uint8)
    return img


def ap(gt, pred):
    a, _, _, _ = metrics.average_precision(gt.astype(np.int32), pred.astype(np.int32),
                                           threshold=[0.5])
    return float(a[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="testdata/data/2D/gray_2D_cp4_gt_masks.png")
    p.add_argument("--suite", default="lowq/gray_2D_*.tif")
    p.add_argument("--overlay-level", default="moderate",
                   help="Which quality level to draw the final overlay on.")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    files = {os.path.basename(f).split("_")[-1].split(".")[0]: f
             for f in glob.glob(args.suite)}
    levels = [lv for lv in LEVELS if lv in files]
    gt = imread(args.gt).astype(np.int32)
    n_gt = len(np.unique(gt)) - 1

    model = models.CellposeModel(gpu=not args.no_gpu)
    ladder = M.LADDER + M.EXTRAS

    # run ladder x levels
    imgs = {lv: load8(files[lv]) for lv in levels}
    ap_grid = {name: {} for name in ladder}      # name -> level -> AP
    cell_grid = {name: {} for name in ladder}
    masks_cache = {}                             # (name, level) -> mask
    for name in ladder:
        fn = M.METHODS[name]
        for lv in levels:
            mask = fn(imgs[lv], model)
            ap_grid[name][lv] = ap(gt, mask)
            cell_grid[name][lv] = len(np.unique(mask)) - 1
            masks_cache[(name, lv)] = mask

    # ---- matrix table ----
    print(f"AP@0.5 by model x quality   (GT = {n_gt} cells)\n")
    hdr = f"{'model':10s} " + " ".join(f"{lv:>9s}" for lv in levels)
    print(hdr); print("-" * len(hdr))
    best_per_level = {lv: max(ladder, key=lambda n: ap_grid[n][lv]) for lv in levels}
    for name in ladder:
        row = f"{name:10s} " + " ".join(
            f"{ap_grid[name][lv]:9.3f}" + ("*" if best_per_level[lv] == name else " ")
            for lv in levels)
        print(row)
    print("\n(* = best at that quality level)")

    print(f"\ncells found (GT={n_gt}):\n")
    print(hdr); print("-" * len(hdr))
    for name in ladder:
        print(f"{name:10s} " + " ".join(f"{cell_grid[name][lv]:9d}" for lv in levels))

    # ---- progression line chart ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = range(len(levels))
    cmap = plt.get_cmap("viridis")
    extra_colors = {"whole_image": "#e8663d", "denoise": "#c94fb0"}
    n_rungs = len(M.LADDER)
    for i, name in enumerate(ladder):
        ys = [ap_grid[name][lv] for lv in levels]
        if name == "baseline":
            style = dict(ls="--", color="#888", marker="o")
        elif name in M.EXTRAS:
            style = dict(ls=":", color=extra_colors.get(name, "#e8663d"), marker="s")
        else:
            style = dict(ls="-", color=cmap(0.12 + 0.8 * i / n_rungs), marker="o")
        ax.plot(xs, ys, lw=2.4, label=name, **style)
    ax.set_xticks(list(xs)); ax.set_xticklabels(levels)
    ax.set_ylabel("AP @ 0.5  (vs ground truth)")
    ax.set_xlabel("image quality")
    ax.set_title("Improvement ladder: accuracy across degradation levels")
    ax.grid(True, alpha=0.25); ax.legend(title="model", frameon=False)
    os.makedirs("output", exist_ok=True)
    ap_path = os.path.join("output", "progression_ap.png")
    fig.tight_layout(); fig.savefig(ap_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved progression chart -> {ap_path}")

    # ---- final overlay: GT | baseline | best model on chosen level ----
    lv = args.overlay_level if args.overlay_level in levels else levels[len(levels) // 2]
    mean_ap = {name: np.mean([ap_grid[name][l] for l in levels]) for name in ladder}
    best = max(ladder, key=lambda n: mean_ap[n])
    panels = [("ground truth", gt),
              (f"baseline", masks_cache[("baseline", lv)]),
              (f"best: {best}", masks_cache[(best, lv)])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for axp, (title, mask) in zip(axes, panels):
        axp.imshow(imgs[lv], cmap="gray")
        for o in utils.outlines_list(mask):
            axp.plot(o[:, 0], o[:, 1], lw=0.5, color="#ff4d4d")
        cells = len(np.unique(mask)) - 1
        extra = "" if title == "ground truth" else f"  AP@.5={ap_grid[title.split(': ')[-1] if 'best' in title else 'baseline'][lv]:.3f}"
        axp.set_title(f"{title} · {cells} cells{extra}", fontsize=12); axp.axis("off")
    fig.suptitle(f"{lv} quality  ·  GT={n_gt} cells  ·  best avg model = {best}", fontsize=13)
    ov_path = os.path.join("output", "progression_overlay.png")
    fig.tight_layout(); fig.savefig(ov_path, dpi=130, bbox_inches="tight")
    print(f"Saved final overlay ({lv}) -> {ov_path}")
    print(f"\nBest model by mean AP across levels: {best}  "
          f"(mean AP {mean_ap[best]:.3f} vs baseline {mean_ap['baseline']:.3f})")


if __name__ == "__main__":
    main()

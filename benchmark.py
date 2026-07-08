"""
Benchmark the faithful tiled segmentation across quality levels.

Runs the snippet's `manual_tile_segment` on the clean image and each degraded
variant, then scores every result against the ground-truth masks using
Cellpose's own average-precision metric. Shows where quality collapses.

Usage:
    python benchmark.py --gt testdata/data/2D/gray_2D_cp4_gt_masks.png \
                        --images lowq/gray_2D_clean.tif lowq/gray_2D_mild.tif ...
    # or just run with no args to use the gray_2D suite in ./lowq
"""

import argparse
import glob
import os

import numpy as np
from skimage.io import imread

from methods.original_snippet import manual_tile_segment
from cellpose import models, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="testdata/data/2D/gray_2D_cp4_gt_masks.png")
    p.add_argument("--images", nargs="*", default=None)
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    images = args.images or sorted(
        glob.glob("lowq/gray_2D_*.tif"),
        key=lambda f: ["clean", "mild", "moderate", "severe", "extreme"].index(
            os.path.splitext(os.path.basename(f))[0].split("_")[-1]))

    gt = imread(args.gt).astype(np.int32)
    n_gt = len(np.unique(gt)) - 1

    model = models.CellposeModel(gpu=not args.no_gpu)
    print(f"\nGround truth: {n_gt} cells\n")
    print(f"{'variant':10s} {'pred':>6s} {'AP@0.5':>8s} {'AP@0.75':>8s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print("-" * 56)

    for path in images:
        img = imread(path)
        pred = manual_tile_segment(img, model, tile_size=200, overlap=100,
                                   cellprob_threshold=1, niter=500)
        ap, tp, fp, fn = metrics.average_precision(
            gt, pred.astype(np.int32), threshold=[0.5, 0.75])
        name = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        n_pred = len(np.unique(pred)) - 1
        print(f"{name:10s} {n_pred:6d} {ap[0]:8.3f} {ap[1]:8.3f} "
              f"{int(tp[0]):5d} {int(fp[0]):5d} {int(fn[0]):5d}")


if __name__ == "__main__":
    main()

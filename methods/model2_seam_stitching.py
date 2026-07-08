"""
model2 = model1 + SEAM STITCHING.

The ONE change vs model1: how tiles are written back. model1 (like the original)
pasted only each tile's center and gave every tile's cells fresh offset IDs — so a
nucleus straddling a seam became TWO cells. Here we instead write each tile's cells
and MERGE any that substantially overlap a cell already written, so a split nucleus
keeps ONE id. This is the fix that removes the double-counting.

Diff this file against model1_global_normalization.py to see the change: the
center-crop + offset block is replaced by the `_write_stitched` call.
"""

import numpy as np


def run(img, model):
    h, w = img.shape
    full_mask = np.zeros((h, w), dtype=np.int64)
    next_id = 0
    tile_size, overlap = 200, 100
    step = tile_size - overlap

    # global normalization (from model1)
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    normalize = {"lowhigh": [lo, hi]}

    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tiles.append((y, x, min(y + tile_size, h), min(x + tile_size, w)))

    for (y1, x1, y2, x2) in tiles:
        patch = img[y1:y2, x1:x2]

        # Skip background tiles
        if patch.max() < 100:
            continue

        masks_tile, _, _ = model.eval(patch, niter=500, cellprob_threshold=1,
                                      normalize=normalize)
        masks_tile = masks_tile.astype(np.int64)

        # CHANGE: merge with cells already written (vs center-crop + offset).
        next_id = _write_stitched(full_mask, masks_tile, y1, x1, next_id)

    return full_mask


def _write_stitched(full_mask, tile, y1, x1, next_id, merge_frac=0.20):
    """Write a tile's cells; if a cell overlaps one already present by >= merge_frac
    of its pixels, adopt that existing id (merge across the seam) instead of a new one."""
    th, tw = tile.shape
    region = full_mask[y1:y1 + th, x1:x1 + tw]
    for L in np.unique(tile):
        if L == 0:
            continue
        m = tile == L
        under = region[m]
        under = under[under > 0]
        assigned = None
        if under.size:
            vals, counts = np.unique(under, return_counts=True)
            if counts.max() / m.sum() >= merge_frac:
                assigned = int(vals[counts.argmax()])
        if assigned is None:
            next_id += 1
            assigned = next_id
        region[m & (region == 0)] = assigned      # claim only free pixels
    return next_id

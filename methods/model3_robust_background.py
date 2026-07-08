"""
model3 = model2 + BIT-DEPTH-SAFE BACKGROUND SKIP.

The ONE change vs model2: the "skip empty tiles" test. The original used
`patch.max() < 100`, which only makes sense for 8-bit (0-255) data — on a raw
16-bit image it never skips, on float data it skips everything. Here we derive the
cutoff from the image's own intensity range (15% of the way from the 1st to the
99th percentile), so it works at any bit depth.

Diff this file against model2_seam_stitching.py: the background `if` line changes.
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
    bg_cut = lo + 0.15 * (hi - lo)     # CHANGE: bit-depth-safe background level

    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tiles.append((y, x, min(y + tile_size, h), min(x + tile_size, w)))

    for (y1, x1, y2, x2) in tiles:
        patch = img[y1:y2, x1:x2]

        # CHANGE: skip empty tiles by the image's own range (was `< 100`, 8-bit only).
        if patch.max() < bg_cut:
            continue

        masks_tile, _, _ = model.eval(patch, niter=500, cellprob_threshold=1,
                                      normalize=normalize)
        masks_tile = masks_tile.astype(np.int64)

        next_id = _write_stitched(full_mask, masks_tile, y1, x1, next_id)

    return full_mask


def _write_stitched(full_mask, tile, y1, x1, next_id, merge_frac=0.20):
    """Write a tile's cells; merge any that overlap a cell already present by
    >= merge_frac of its pixels, so a nucleus split across a seam keeps ONE id."""
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
        region[m & (region == 0)] = assigned
    return next_id

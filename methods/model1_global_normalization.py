"""
model1 = original_snippet + GLOBAL NORMALIZATION.

The ONE change vs the original: instead of letting Cellpose re-normalize each
tile to its own 1st/99th percentile (which makes tiles inconsistent and amplifies
noise in near-empty tiles), we compute the 1st/99th percentile of the WHOLE image
once and force every tile to use that same contrast mapping.

Everything else is the original loop. Diff this file against original_snippet.py
to see exactly what changed (search for "CHANGE").
"""

import numpy as np
from tqdm import tqdm


def run(img, model):
    h, w = img.shape
    full_mask = np.zeros((h, w), dtype=np.int64)
    cell_id_offset = 0
    tile_size, overlap = 200, 100
    step = tile_size - overlap

    # CHANGE: one contrast curve for the whole image, reused by every tile.
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    normalize = {"lowhigh": [lo, hi]}

    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tiles.append((y, x, min(y + tile_size, h), min(x + tile_size, w)))

    for (y1, x1, y2, x2) in tqdm(tiles, desc="model1 tiles"):
        patch = img[y1:y2, x1:x2]

        # Skip background tiles
        if patch.max() < 100:
            continue

        # CHANGE: pass the global normalization to eval (was default per-tile).
        masks_tile, _, _ = model.eval(patch, niter=500, cellprob_threshold=1,
                                      normalize=normalize)
        masks_tile = masks_tile.astype(np.int64)

        # Offset cell IDs to avoid collisions
        local_max = int(masks_tile.max())
        masks_tile[masks_tile > 0] += cell_id_offset

        # Write only center region to avoid boundary artifacts
        cy1 = y1 + overlap // 2 if y1 > 0 else y1
        cx1 = x1 + overlap // 2 if x1 > 0 else x1
        cy2 = y2 - overlap // 2 if y2 < h else y2
        cx2 = x2 - overlap // 2 if x2 < w else x2

        full_mask[cy1:cy2, cx1:cx2] = masks_tile[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
        cell_id_offset += local_max

    return full_mask

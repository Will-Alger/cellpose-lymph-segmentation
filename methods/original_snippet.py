"""
original_snippet — the original code, VERBATIM.

`manual_tile_segment` below is exactly as originally written (only trailing
whitespace on blank lines differs from the raw snippet.py). It is never edited;
every improvement lives in a separate model file. This is the baseline the whole
comparison is measured against.
"""

import numpy as np


def manual_tile_segment(img, model, tile_size=200, overlap=100, cellprob_threshold=+1, niter=500):
    h, w = img.shape
    full_mask = np.zeros((h, w), dtype=np.int64)
    cell_id_offset = 0
    step = tile_size - overlap

    # Generate all tile coordinates
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tiles.append((y, x, y2, x2))

    print(f"Total tiles: {len(tiles)}")

    for i, (y1, x1, y2, x2) in enumerate(tiles):
        if i % 50 == 0:
            print(f"Processing tile {i}/{len(tiles)}...")

        patch = img[y1:y2, x1:x2]

        # Skip background tiles
        if patch.max() < 100:
            continue

        masks_tile, _, _ = model.eval(patch, niter=niter,
                                       cellprob_threshold=cellprob_threshold)
        masks_tile = masks_tile.astype(np.int64)  # add this

        # Offset cell IDs to avoid collisions
        local_max = int(masks_tile.max())
        masks_tile[masks_tile > 0] += cell_id_offset

        # Write only center region to avoid boundary artifacts
        cy1 = y1 + overlap//2 if y1 > 0 else y1
        cx1 = x1 + overlap//2 if x1 > 0 else x1
        cy2 = y2 - overlap//2 if y2 < h else y2
        cx2 = x2 - overlap//2 if x2 < w else x2

        full_mask[cy1:cy2, cx1:cx2] = masks_tile[cy1-y1:cy2-y1, cx1-x1:cx2-x1]

        cell_id_offset += local_max

    return full_mask


def run(img, model):
    """Baseline entry point: the original algorithm with its original defaults."""
    return manual_tile_segment(img, model, tile_size=200, overlap=100,
                               cellprob_threshold=1, niter=500)

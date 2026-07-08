"""
whole_image = no tiling at all. Run CellposeSAM once on the full image.

This is the control the whole project hinges on — and, on our benchmark, the
winner at every quality level. Cellpose normalizes the whole image once (global by
construction) and does its OWN internal 256-px tiling for the network, so this
handles large images without any manual tiling. If this beats the model1..4
ladder, the manual tiling isn't buying anything.
"""

import numpy as np


def run(img, model):
    masks, _, _ = model.eval(img, cellprob_threshold=0.0)
    return masks.astype(np.int64)

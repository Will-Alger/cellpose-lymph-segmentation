"""
methods — the segmentation approaches, one self-contained file each.

Each file exposes `run(img, model) -> label_mask` and is fully standalone (its own
tiling loop), so you can read one and diff it against its neighbor to see exactly
one change. This registry just maps friendly names to those `run` functions.

The ladder (each rung = the previous file + ONE documented change):
    original_snippet            -> baseline      (original code, verbatim)
    model1_global_normalization -> model1        + global normalization
    model2_seam_stitching       -> model2        + seam stitching
    model3_robust_background     -> model3        + bit-depth-safe background skip
    model4_recover_dim_cells    -> model4        + recover dim cells

Alternatives (different premise, not rungs of the ladder):
    whole_image_no_tiling       -> whole_image   (no tiling; the benchmark winner)
    denoise_then_segment        -> denoise       (denoise first; did not win)
"""

from . import original_snippet
from . import model1_global_normalization
from . import model2_seam_stitching
from . import model3_robust_background
from . import model4_recover_dim_cells
from . import whole_image_no_tiling
from . import denoise_then_segment

# friendly name -> run function
METHODS = {
    "baseline":    original_snippet.run,
    "model1":      model1_global_normalization.run,
    "model2":      model2_seam_stitching.run,
    "model3":      model3_robust_background.run,
    "model4":      model4_recover_dim_cells.run,
    "whole_image": whole_image_no_tiling.run,
    "denoise":     denoise_then_segment.run,
}

# the cumulative tiling ladder, in order
LADDER = ["baseline", "model1", "model2", "model3", "model4"]
# alternative approaches, plotted distinctly by progression.py
EXTRAS = ["whole_image", "denoise"]

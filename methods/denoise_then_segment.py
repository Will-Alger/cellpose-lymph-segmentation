"""
denoise = denoise the whole image first, then run whole-image segmentation.

An alternative approach we tested (it did NOT win — classical denoising smooths
away faint nuclei and hurt the mid-quality range). Kept for the record so the
comparison is honest.

NOTE: this cellpose build (4.2.x, SAM line) ships no learned DenoiseModel, so we
use a classical non-local-means denoiser (skimage) — suited to the Poisson/Gaussian
noise of fluorescence. A learned restoration model might do better; this doesn't.
"""

import numpy as np


def run(img, model):
    from skimage.restoration import denoise_nl_means, estimate_sigma
    x = img.astype(np.float32) / 255.0
    sig = float(np.mean(estimate_sigma(x)))
    if sig > 1e-4:
        x = denoise_nl_means(x, h=0.8 * sig, sigma=sig, fast_mode=True,
                             patch_size=5, patch_distance=6)
    den = (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    masks, _, _ = model.eval(den, cellprob_threshold=0.0)
    return masks.astype(np.int64)

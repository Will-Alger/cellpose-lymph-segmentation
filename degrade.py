"""
Generate low-quality variants of a clean microscopy image.

The point: your real lymph-node DAPI is low-contrast / noisy, but we don't have
a labelled copy of it. So instead we take a CLEAN image that HAS ground-truth
masks and degrade it in controlled, microscopy-realistic ways. Because the masks
still apply to the degraded image, we keep a known "correct answer" and can
measure exactly where CellposeSAM starts to fail as quality drops.

Degradations modelled (all optional, each with a severity knob):
  - contrast : compress intensity toward mid-gray (weak staining / low dynamic range)
  - poisson  : shot noise from low photon counts (the dominant noise in fluorescence)
  - gauss    : additive read noise (camera/electronic noise)
  - blur     : gaussian blur (out-of-focus / poor optics)
  - downscale: resize down then back up (low spatial resolution)

Presets combine these into mild -> extreme. Output is uint8 so it drops straight
into segment.py (whose background test assumes 8-bit).

Usage:
    python degrade.py testdata/data/2D/gray_2D.png --suite --outdir lowq
    python degrade.py img.png --preset severe --out img_severe.tif
"""

import argparse
import os

import numpy as np
from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.transform import rescale


# severity -> dict of degradation parameters
PRESETS = {
    "clean":    dict(contrast=1.00, poisson=0.0,  gauss=0.0,  blur=0.0, downscale=1.0),
    "mild":     dict(contrast=0.70, poisson=0.15, gauss=3.0,  blur=0.6, downscale=1.0),
    "moderate": dict(contrast=0.45, poisson=0.30, gauss=6.0,  blur=1.0, downscale=1.0),
    "severe":   dict(contrast=0.28, poisson=0.55, gauss=10.0, blur=1.5, downscale=0.6),
    "extreme":  dict(contrast=0.16, poisson=0.85, gauss=16.0, blur=2.2, downscale=0.45),
}


def to_gray_float(img):
    """Coerce to a single-channel float image in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)  # RGB -> gray
    img = img.astype(np.float64)
    mx = img.max()
    return img / mx if mx > 0 else img


def degrade(img, contrast=1.0, poisson=0.0, gauss=0.0, blur=0.0, downscale=1.0,
            seed=0):
    """Apply a controlled quality reduction. Input any image; returns uint8 [0,255].

    Order mirrors a real acquisition chain: blur (optics) -> low res -> reduced
    signal/contrast -> shot noise -> read noise -> quantize to 8-bit.
    """
    rng = np.random.default_rng(seed)
    x = to_gray_float(img)  # [0,1]

    # Optics: out-of-focus blur.
    if blur > 0:
        x = gaussian(x, sigma=blur)

    # Low spatial resolution: shrink then restore size (loses fine detail).
    if downscale < 1.0:
        small = rescale(x, downscale, anti_aliasing=True, preserve_range=True)
        x = rescale(small, 1.0 / downscale, anti_aliasing=True, preserve_range=True)
        # rescale can be off-by-a-pixel; crop/pad back to original shape.
        H, W = to_gray_float(img).shape
        x = _fit(x, H, W)

    # Weak signal / low contrast: pull values toward mid-gray.
    if contrast != 1.0:
        x = 0.5 + (x - 0.5) * contrast

    x = np.clip(x, 0, 1)

    # Shot noise: photon counting. Lower `photons` => noisier. poisson in [0,1].
    if poisson > 0:
        photons = np.interp(poisson, [0, 1], [200.0, 8.0])  # fewer photons = worse
        x = rng.poisson(np.clip(x, 0, 1) * photons) / photons

    # Read noise: additive gaussian, gauss is in 8-bit units.
    if gauss > 0:
        x = x + rng.normal(0, gauss / 255.0, x.shape)

    x = np.clip(x, 0, 1)
    return (x * 255).round().astype(np.uint8)


def _fit(x, H, W):
    """Crop or zero-pad x to exactly (H, W)."""
    x = x[:H, :W]
    ph, pw = H - x.shape[0], W - x.shape[1]
    if ph > 0 or pw > 0:
        x = np.pad(x, ((0, max(ph, 0)), (0, max(pw, 0))))
    return x


def main():
    p = argparse.ArgumentParser(description="Make low-quality variants of an image.")
    p.add_argument("image")
    p.add_argument("--preset", choices=list(PRESETS), default="severe")
    p.add_argument("--suite", action="store_true",
                   help="Generate ALL presets into --outdir instead of a single --preset.")
    p.add_argument("--outdir", default="lowq")
    p.add_argument("--out", default=None, help="Output path for single-preset mode.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    img = imread(args.image)
    base = os.path.splitext(os.path.basename(args.image))[0]

    if args.suite:
        os.makedirs(args.outdir, exist_ok=True)
        for name, params in PRESETS.items():
            out = degrade(img, seed=args.seed, **params)
            path = os.path.join(args.outdir, f"{base}_{name}.tif")
            imsave(path, out)
            print(f"{name:9s} -> {path}  (mean={out.mean():.1f}, "
                  f"std={out.std():.1f}, max={out.max()})")
    else:
        out = degrade(img, seed=args.seed, **PRESETS[args.preset])
        path = args.out or f"{base}_{args.preset}.tif"
        imsave(path, out)
        print(f"Saved {args.preset} -> {path}")


if __name__ == "__main__":
    main()

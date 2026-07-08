"""
segment.py — run ONE segmentation method on ONE image, save mask + overlay.

Loads an image, creates the CellposeSAM model, runs the chosen method from the
`methods/` package (default: the original snippet), and saves a label mask (.npy
+ .tif) and an outline overlay (.png).

Usage:
    python segment.py path/to/dapi.tif                     # baseline (original code)
    python segment.py path/to/dapi.tif --method whole_image
    python segment.py path/to/image.tif --channel 0 --no-gpu
"""

import argparse
import os

import numpy as np

import methods as M


def load_dapi(path, channel=None):
    """Load a (possibly multi-channel / multi-bit-depth) image as a 2D uint8 array.

    The original snippet assumed a variable `dapi` that was already 2D 8-bit (its
    `patch.max() < 100` test only makes sense for 0-255 data). Bit depth of the
    real image is unknown, so we read the file, pull out one 2D channel, and
    rescale to uint8 if needed (a neutral min-max; see CHANGES.md).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import tifffile
            img = tifffile.imread(path)
        except ImportError:
            from skimage.io import imread
            img = imread(path)
    else:
        from skimage.io import imread
        img = imread(path)

    img = np.asarray(img)
    print(f"Loaded {path}: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}")

    if img.ndim == 3:
        ch_axis = int(np.argmin(img.shape))     # channel axis = smallest dim
        n_ch = img.shape[ch_axis]
        if channel is None:
            channel = 0
            print(f"Multi-channel image ({n_ch} channels on axis {ch_axis}); "
                  f"using channel {channel}. Override with --channel.")
        img = np.take(img, channel, axis=ch_axis)
    elif img.ndim != 2:
        raise ValueError(f"Unsupported image ndim={img.ndim}; expected 2 or 3.")

    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        print(f"Rescaling {img.dtype} -> uint8 via min-max ({lo} .. {hi}).")
        img = ((img.astype(np.float64) - lo) / (hi - lo) * 255.0 if hi > lo
               else np.zeros_like(img, np.float64)).round().astype(np.uint8)

    return img


def main():
    p = argparse.ArgumentParser(description="Run one segmentation method on one image.")
    p.add_argument("image", help="Path to the DAPI image (tif/png/...).")
    p.add_argument("--method", default="baseline", choices=list(M.METHODS),
                   help="Which approach from methods/ to run.")
    p.add_argument("--channel", type=int, default=None,
                   help="Channel index for multi-channel images.")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    p.add_argument("--out", default=None, help="Output prefix (default: alongside input).")
    args = p.parse_args()

    dapi = load_dapi(args.image, channel=args.channel)

    from cellpose import models
    model = models.CellposeModel(gpu=not args.no_gpu)
    print(f"Model loaded (gpu={not args.no_gpu}). Running method: {args.method}")

    full_mask = M.METHODS[args.method](dapi, model)
    total_cells = len(np.unique(full_mask)) - 1
    cf_p = total_cells / (dapi.shape[0] * dapi.shape[1])
    print(f"Total cells: {total_cells}")
    print(f"Cf/P: {cf_p:.6f}")

    if args.out:
        out_prefix = args.out
    else:
        os.makedirs("output", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_prefix = os.path.join("output", f"{base}_{args.method}")
    np.save(out_prefix + ".npy", full_mask)
    print(f"Saved label mask -> {out_prefix}.npy")
    try:
        import tifffile
        tifffile.imwrite(out_prefix + ".tif", full_mask.astype(np.int32))
        print(f"Saved label mask -> {out_prefix}.tif")
    except ImportError:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cellpose import utils
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(dapi, cmap="gray")
        for o in utils.outlines_list(full_mask):
            ax.plot(o[:, 0], o[:, 1], linewidth=0.5, color="red")
        ax.set_title(f"{args.method} · {total_cells} cells")
        ax.axis("off")
        fig.savefig(out_prefix + "_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved overlay -> {out_prefix}_overlay.png")
    except ImportError:
        print("(matplotlib not installed; skipping overlay PNG)")


if __name__ == "__main__":
    main()

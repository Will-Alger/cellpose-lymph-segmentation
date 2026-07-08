# Changes log

Tracks how `segment.py` evolves from the original `snippet.py`. We start with a
**faithful** completion (run it as-is), then apply documented improvements.

## Phase 0 — Faithful completion (DONE)

`segment.py` wraps the original `manual_tile_segment` **verbatim**. Only
scaffolding the snippet was missing was added:

- **Imports** (`numpy`, `cellpose.models`).
- **Image loading** (`load_dapi`): reads tif/png, extracts one 2D channel from
  multi-channel images, and rescales non-8-bit data to uint8 via neutral min-max.
  - *Why min-max to uint8:* the snippet's background skip `patch.max() < 100`
    only makes sense for 0–255 data. This is the one scaffolding decision that
    touches pixel values; flagged for revisit below.
- **Model creation**: `CellposeModel(gpu=True)` → loads `cpsam_v2` (CellposeSAM).
- **Output saving**: `.npy` + `.tif` label masks, plus printed `Total cells` / `Cf/P`.

Algorithm unchanged from the snippet in this phase.

## Experiment harness + improvement ladder (DONE)

`methods/` is a **cumulative ladder**, one self-contained file per approach (each
duplicates the tiling loop for readability — diff neighboring files to see the one
change; search for `CHANGE`). `original_snippet.py` is the original code verbatim;
`model1..model4` each add EXACTLY ONE change. `methods/__init__.py` is the registry.
See README.md for the file map.

- `compare.py <image> --gt <gt> --methods baseline model2 …` — A/B any subset on one
  image (table + side-by-side overlay).
- `progression.py` — runs the whole ladder across every quality level, prints the
  model×quality AP matrix, and saves `progression_ap.png` (line chart) +
  `progression_overlay.png` (GT | baseline | best).

The ladder (each = previous + one change):
| model | one change added | flag |
|-------|------------------|------|
| baseline | the original snippet | — |
| model1 | global normalization (one contrast curve for the whole image) | `norm_global` |
| model2 | seam stitching — merge cells across tile overlaps by pixel overlap | `stitch` |
| model3 | bit-depth-safe background skip (percentile vs `max()<100`) | `robust_bg` |
| model4 | recover dim cells (cellprob threshold +1 → 0) | `recover_dim` |

### Results log
- **model1 (global norm)** vs baseline on `gray_2D_moderate`: AP@0.5 0.318 → 0.310 —
  **no meaningful change on this image** (evenly-distributed nuclei; per-tile norm
  doesn't diverge). Expected to matter more on images with large empty regions.
- **model2 (+ seam stitching)** on `gray_2D_moderate`: AP@0.5 0.318 → **0.457**,
  false positives 96 → 53, true positives 102 → 127. **The big win** — confirms
  seam double-counting was the dominant error. Addresses Phase-1 item #3.
- model3/model4 + full model×quality matrix: see `progression.py` output /
  `progression_ap.png`.

### KEY FINDING — no tiling beats the tiling ladder
Added two alternatives (`methods.py`): `whole_image` (run CellposeSAM once, no
manual tiling) and `denoise` (NL-means restoration + whole_image; note: this
cellpose build has no DenoiseModel, so classical denoiser used).

Mean AP@0.5 across quality levels (GT=225):
| approach | clean | mild | moderate | severe | mean |
|----------|-------|------|----------|--------|------|
| baseline (original tiling) | 0.549 | 0.459 | 0.318 | 0.004 | 0.266 |
| model4 (best tiling)  | 0.816 | 0.780 | 0.570 | 0.098 | 0.453 |
| **whole_image (no tiling)** | **0.965** | **0.796** | **0.597** | **0.223** | **0.516** |
| denoise (NL-means)    | 0.965 | 0.611 | 0.441 | 0.147 | 0.433 |

**`whole_image` wins at every level.** The manual tiling was the primary source
of error, not a fix — it introduced per-tile normalization inconsistency and seam
double-counting. Running CellposeSAM once (its default global normalization) is
better everywhere; near-perfect on clean (0.965, 227 cells vs 225 GT) and 2× the
tiling ladder at severe. Classical **denoising did not help** (hurt mid-range by
smoothing away faint nuclei). Answers Phase-1 item #6.

**Recommendation:** drop the manual tiling. Use `whole_image`. Only reintroduce
tiling if the real image is too large for GPU memory — and if so, use model2/3's
IoU seam-stitching, never the original center-crop-offset scheme.

## Phase 1 — Planned improvements (NOT YET APPLIED)

Each will be applied as a separate, documented change once the faithful run works
and we have the real image to compare against.

1. **Per-tile normalization → global normalization.** Cellpose's default
   `normalize=True` rescales each 200×200 tile to its *own* 1st/99th percentile.
   On a low-contrast lymph-node image this gives every tile different contrast and
   amplifies noise in near-empty tiles. Fix: normalize the whole image once, then
   pass `normalize=False` to `model.eval`, or use `tile_norm_blocksize`.

2. **Cellpose's internal tiling vs. our manual tiling.** For 200px patches
   Cellpose pads to its fixed 256 block (one tile), so it's not truly "double
   tiling" — but the per-tile normalization above is the real interaction. Verify
   `bsize`/`tile_overlap` behavior and consider larger tiles.

3. **Seam double-counting.** A nucleus split across the center-write boundary
   becomes two labels → inflated `Total cells`. Fix: stitch labels across tile
   overlaps by IoU (cellpose has `stitch_threshold`/utilities), or use a
   running label map with merge-on-overlap.

4. **Bit-depth / background threshold.** Replace the hard-coded `max() < 100`
   8-bit assumption with a percentile- or Otsu-based background test computed from
   the actual image, so 16-bit DAPI works without the min-max rescale.

5. **Parameter sweep.** `cellprob_threshold=+1` (aggressive, drops dim cells) and
   `niter=500` (high) pull in opposite directions on a faint image. Tune against
   ground-truth-ish counts once we see results.

6. **Reconsider tiling at all.** Compare manual tiling vs. running CellposeSAM
   once on the full image with global normalization — manual tiling may be
   unnecessary and the source of most artifacts.

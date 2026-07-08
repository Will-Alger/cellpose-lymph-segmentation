# Lymph-node CellposeSAM segmentation

Recreating an existing cell-segmentation snippet, then improving it with
documented, measured changes. Everything runs with the project venv.

## Quick start (after cloning)

**Prerequisites:** Python 3.10 or 3.11, and git. An NVIDIA GPU is optional (it's
much faster, but everything runs on CPU / Apple Silicon too).

```bash
git clone <your-repo-url>
cd project
```

**1. Create and activate a virtual environment**

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**2. Install PyTorch** (pick the line for your machine):
```bash
# NVIDIA GPU on Windows/Linux — incl. RTX 50-series (Blackwell):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# macOS (Apple Silicon or Intel) — uses MPS/CPU automatically:
pip install torch torchvision

# No GPU (Windows/Linux CPU-only):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. Install the rest of the dependencies**
```bash
pip install -r requirements.txt
```

**4. Get the sample images** (used by all the examples below):
```bash
python get_sample_data.py
```

**5. Smoke test** — confirm the GPU/model/pipeline all work:
```bash
python segment.py testdata/data/2D/gray_2D.png --method whole_image
```
If that prints a cell count and writes files into `output/`, you're set.

> **Command prefix:** once the venv is *activated* (step 1), plain `python …`
> works. If you'd rather not activate it, call the venv Python directly:
> `.venv\Scripts\python.exe …` on Windows, `.venv/bin/python …` on macOS/Linux.
> The examples below use plain `python` and forward-slash paths; on Windows use
> backslashes.

## The approaches — `methods/`

Each approach is **one self-contained file** exposing `run(img, model) -> mask`.
They deliberately duplicate the tiling loop instead of sharing an engine, so you
can read one file top-to-bottom and diff it against its neighbor to see the single
change (search each model file for `CHANGE`).

| file | name | one change vs the file above it |
|------|------|--------------------------------|
| `original_snippet.py` | `baseline` | the original code, verbatim |
| `model1_global_normalization.py` | `model1` | + global normalization |
| `model2_seam_stitching.py` | `model2` | + seam stitching (merge cells across tiles) |
| `model3_robust_background.py` | `model3` | + bit-depth-safe background skip |
| `model4_recover_dim_cells.py` | `model4` | + lower cellprob threshold (catch faint cells) |
| `whole_image_no_tiling.py` | `whole_image` | *no tiling at all* — the benchmark winner |
| `denoise_then_segment.py` | `denoise` | denoise first (tested; did not win) |

`methods/__init__.py` is the registry — it just maps the names above to each
file's `run`, and defines `LADDER` (the tiling rungs) and `EXTRAS` (alternatives).

## Running it

Run these from the project root with the venv activated (see Quick start). Every
generated file lands in `output/`. On Windows, use backslashes in paths (e.g.
`testdata\data\2D\gray_2D.png`).

> The examples below use the bundled sample image `testdata/data/2D/gray_2D.png`
> (and its ground-truth mask `..._cp4_gt_masks.png`). Swap in your own image path
> anywhere you see them.

### 1. Segment one image with one method → `segment.py`
Runs a single method and saves the label mask (`.npy` + `.tif`) and an outline
overlay (`.png`) to `output/`.

```bash
# the original code, on the sample image
python segment.py testdata/data/2D/gray_2D.png

# the winning approach (no tiling) instead
python segment.py testdata/data/2D/gray_2D.png --method whole_image

# your own multi-channel image, using channel 0, on CPU
python segment.py path/to/your.tif --channel 0 --no-gpu
```
Valid `--method` names: `baseline`, `model1`, `model2`, `model3`, `model4`,
`whole_image`, `denoise`.

### 2. A/B several methods on one image → `compare.py`
Prints a scored table (cells, AP, TP/FP/FN) and saves a side-by-side overlay
figure. Pass `--gt` to score against ground truth.

```bash
python compare.py testdata/data/2D/gray_2D.png --gt testdata/data/2D/gray_2D_cp4_gt_masks.png --methods baseline model2 whole_image
```
→ `output/compare_gray_2D.png`

### 3. Make low-quality test images → `degrade.py`
Generates `clean → mild → moderate → severe → extreme` variants (plus a montage)
into `lowq/`, so you can test the low-quality regime with a known answer.

```bash
# a whole suite of severities
python degrade.py testdata/data/2D/gray_2D.png --suite --outdir lowq

# just one severity
python degrade.py testdata/data/2D/gray_2D.png --preset severe
```

### 4. Compare the whole ladder across quality levels → `progression.py`
Runs every method on every degradation level, prints the model×quality AP matrix,
and saves the progression chart + a final overlay. (Run `degrade.py --suite` first
so `lowq/` exists.)

```bash
# uses lowq/gray_2D_*.tif + the gray_2D ground truth by default
python progression.py

# point it at a different degraded suite + ground truth
python progression.py --suite "lowq/your_*.tif" --gt path/to/your_gt.png
```
→ `output/progression_ap.png`, `output/progression_overlay.png`

### End-to-end example (from scratch on the sample image)
```bash
python degrade.py testdata/data/2D/gray_2D.png --suite --outdir lowq
python progression.py
```

## Data
- `testdata/` — real labelled nuclei images (base: `gray_2D.png`, 225 GT cells).
- `lowq/` — degraded variants generated by `degrade.py`.
- `output/` — all generated masks, overlays, and figures.

## Result so far
See `CHANGES.md`. Headline: **`whole_image` (no manual tiling) beats the entire
tiling ladder at every quality level** — the manual tiling was the main source of
error. Validated on one image family with synthetic degradation; re-run
`progression.py` on the real image when available.

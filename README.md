# PixelLab — Image analysis workspace

Flask image workspace for real image histogram analysis, enhancement and Otsu segmentation. The frontend uses HTML, CSS and vanilla JavaScript; NumPy/OpenCV handle pixels, and Matplotlib exports histogram charts.

## Install and run (Windows PowerShell)

Requires Python 3.13. The Python launcher (`py`) is optional; use `python` if it is unavailable.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`. Upload limit is 20 MB, decoded image limit is 16 megapixels and each dimension is capped at 10,000 pixels.

## Features

- Grayscale intensity, RGB, HSV, LAB and YCrCb histograms, with min/max/mean/median/standard deviation/dynamic range/entropy/contrast.
- Contrast stretching, global histogram equalization, CLAHE and gamma correction on grayscale or the LAB lightness channel for color.
- Otsu Binary/Inverse with threshold, between-class variance and foreground/background ratios.
- Optional independent erosion, dilation, opening or closing with configurable kernel size/shape/iterations.
- Before/after slider, zoom and pan, interactive canvas histogram with channel toggles, hover counts, zoom and statistics markers.
- Comparison previews for all enhancement methods, limited undo/redo history, and PNG/CSV exports.

## Pixel conventions

- Uploaded pixels are decoded from file contents; supported suffixes are PNG, JPG/JPEG, BMP and TIFF.
- 16-bit images are linearly mapped using the observed image minimum and maximum. Constant 16-bit images use the high byte. The uploaded file itself is not rewritten.
- RGBA/BGRA is composited over white for analysis, grayscale, enhancement and Otsu. Enhanced output is opaque; the original file remains unchanged in the browser.
- OpenCV's HSV hue channel has range 0–179; its histogram bins retain those raw hue values. The other uint8 channels use 0–255.
- Contrast is reported as intensity standard deviation. These descriptive statistics do not alone prove perceived quality improvement. IoU/Dice and reference metrics are not shown without suitable references.

## Tests

Run the built-in unittest suite without creating bytecode cache files:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -v
```

Tests cover grayscale/color/alpha/16-bit, histogram counts, constant images, Otsu modes, morphology, invalid/oversized uploads, API responses and exports.

## Structure

- `app.py`, `config.py`: Flask app, request limits and JSON error handling.
- `routes/image_routes.py`: analysis and export endpoints.
- `services/image_service.py`: decoding, limits, alpha and preview/export encoding.
- `services/histogram_service.py`: color-space channel histograms, statistics and Matplotlib rendering.
- `services/enhancement_service.py`: stretch, equalization, CLAHE and gamma.
- `services/segmentation_service.py`: Otsu and separate morphology operations.
- `services/evaluation_service.py`: descriptive and mask metrics.
- `templates/` and `static/`: workspace UI and interactions.
- `tests/`: algorithm and Flask API checks.

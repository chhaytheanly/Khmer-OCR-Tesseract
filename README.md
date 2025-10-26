# Khmer OCR (Tesseract) — README

This repository contains a Khmer OCR pipeline implemented as a Jupyter notebook (`src/khmer-ocr/khmer-ocr.`). The pipeline performs preprocessing (logo removal, rotation correction, denoising, region segmentation), multi-pass Tesseract OCR (region-specific + whole-image), and Khmer-specific post-processing.

## Contents

- `Notebook/khmer-ocr.ipynb` — Notebook with the full OCR pipeline.
- `data/images/` — Example images and input files.
- `output/` — Default output directory where the notebook writes `preprocessed.png`, `extracted_text.txt`, and `region_analysis.txt`.
- `requirements.txt` — Python dependencies for the notebook.

## Quick start

1. Install uv
```
see install guide: https://docs.astral.sh/uv/getting-started/installation/
```

2. Install pakages:

```bash
uv venv
source .venv/bin/activate(mac&Linux)
.venv\Scripts\activate(Window)
uv sync
```

3. Install Tesseract OCR and the Khmer traineddata (platform-specific instructions below).
  This project requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  and the **Khmer language model** (`khm.traineddata`)
  Run this script for download data 

  ```bash
  uv run download_data.py
  ```
  Download tesseract The actual OCR engine
  **See wiki: https://github.com/UB-Mannheim/tesseract/wiki**
  for Mac:
  ```bash
  brew install tesseract
```
  For Linux
```
sudo apt update
sudo apt install tesseract-ocr
```

4. Open the notebook in Jupyter Lab / Notebook and run the cells in order:

```bash
jupyter lab
# open src/Notebook/khmer-ocr.ipynb
```

5. Example usage inside the notebook (already provided in the last test cell):

```python
# Basic usage
python -m src.khmer_ocr.script path/to/image.jpg

# With custom output directory
python -m src.khmer_ocr.script path/to/image.jpg -o custom_output

# Without resolution enhancement
python -m src.khmer_ocr.script path/to/image.jpg --no-enhance

```

## Contributing

Feel free to open an issue or submit a PR with improvements to preprocessing, more robust Khmer post-processing rules, or additional tests and example images.

## Authorize

All rights reserve!

  1. LY Chhaythean
  2. SOEUK Bondol
  3. SOPHON Rachana
  4. NEANG Vanna

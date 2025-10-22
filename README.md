# Khmer OCR (Tesseract) — README

This repository contains a Khmer OCR pipeline implemented as a Jupyter notebook (`src/khmer-ocr/khmer-ocr.ipynb`). The pipeline performs preprocessing (logo removal, rotation correction, denoising, region segmentation), multi-pass Tesseract OCR (region-specific + whole-image), and Khmer-specific post-processing.

## Contents

- `src/khmer-ocr/khmer-ocr.ipynb` — Notebook with the full OCR pipeline.
- `data/images/` — Example images and input files.
- `data/images/output/` — Default output directory where the notebook writes `preprocessed.png`, `extracted_text.txt`, and `region_analysis.txt`.
- `requirements.txt` — Python dependencies for the notebook.

## Quick start

1. Create a virtual environment (recommended) and install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install Tesseract OCR and the Khmer traineddata (platform-specific instructions below).

3. Open the notebook in Jupyter Lab / Notebook and run the cells in order:

```bash
jupyter lab
# open src/khmer-ocr/khmer-ocr.ipynb
```

4. Example usage inside the notebook (already provided in the last test cell):

```python
image_test = process_document(r"/media/chhaythean/Drive D/Ai-Edu/data/images/doc2.jpg",
                              output_dir=r"/media/chhaythean/Drive D/Ai-Edu/data/images/output")
```

## Installing Tesseract and Khmer language data

Note: The notebook uses `pytesseract` (Python wrapper). You must install the Tesseract binary and the Khmer traineddata (`khm.traineddata`) so `tesseract --list-langs` shows `khm`.

### Debian/Ubuntu (apt)

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng
# install Khmer if available as a package (name varies by distro)
sudo apt install -y tesseract-ocr-khm || sudo apt install -y tesseract-ocr-kh || true

# If the Khmer package isn't available, download traineddata manually:
TESSDATA_DIR=$(tesseract --print-tessdata-dir)
sudo curl -L -o "$TESSDATA_DIR/khm.traineddata" \
  https://github.com/tesseract-ocr/tessdata/raw/main/khm.traineddata

# Verify installation
tesseract --version
tesseract --list-langs
```

### macOS (Homebrew)

```bash
brew install tesseract
# install language data via brew if available, otherwise download manually
mkdir -p "$(tesseract --print-tessdata-dir)"
curl -L -o "$(tesseract --print-tessdata-dir)/khm.traineddata" \
  https://github.com/tesseract-ocr/tessdata/raw/main/khm.traineddata

# Verify
tesseract --version
tesseract --list-langs
```

### Windows (choco or manual)

1. Install Tesseract from the official installer: https://github.com/tesseract-ocr/tesseract/releases
2. Add the Tesseract install folder (e.g., `C:\Program Files\Tesseract-OCR`) to your PATH.
3. Download `khm.traineddata` and put it in the `tessdata` folder inside the Tesseract installation directory.

After installation, confirm `tesseract --list-langs` includes `khm`.

## Python dependencies

The project uses the following libraries (listed in `requirements.txt`):

- numpy
- opencv-python
- pillow
- pytesseract
- scipy
- matplotlib

Install with:

```bash
pip install -r requirements.txt
```

If you are running inside a Jupyter notebook, make sure the kernel uses the same Python environment/virtualenv where `pytesseract` and other packages are installed.

## Notebook notes and troubleshooting

- If you see errors like `tesseract is not installed or it's not in your PATH`, make sure the Tesseract binary is installed and `pytesseract.pytesseract.tesseract_cmd` points to the binary path (e.g., `/usr/bin/tesseract`):

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
```

- If the notebook raises `FileNotFoundError` when writing outputs, check that the `output_dir` exists or allow the `process_document` function to create it (the notebook version already creates the directory).

- The pipeline logs colours for readability. If running where ANSI colours aren't supported, you can remove color codes (variables `YELLOW`, `GREEN`, etc.).

## Common error fixes

- ValueError while unpacking Hough lines: updated function `advanced_rotation_correction` ensures safe flattening of Hough output.
- `TypeError: string indices must be integers` when writing `region_analysis.txt`: updated `region_specific_ocr` and `multi_pass_ocr` to always return `region_results` as a list of dicts.
- If OCR returns empty text: verify Khmer traineddata is available and that `tesseract --list-langs` includes `khm`.

## Contributing

Feel free to open an issue or submit a PR with improvements to preprocessing, more robust Khmer post-processing rules, or additional tests and example images.

## Authorize

All rights reserve! -------> Chhaythean LY

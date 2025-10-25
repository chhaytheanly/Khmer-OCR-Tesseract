import os
import platform
import urllib.request
from pathlib import Path


def get_tessdata_dir():
    """Get user-writable tessdata directory in home folder"""
    return Path.home() / "tessdata"


def download_khm_data():
    url = "https://github.com/tesseract-ocr/tessdata/raw/main/khm.traineddata"
    target_dir = get_tessdata_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "khm.traineddata"

    print(f" Downloading Khmer data to: {target_path}")
    urllib.request.urlretrieve(url, target_path)
    print(" Download complete!")
    print(f"\n To use this data with Tesseract, run:")
    print(f" tesseract --tessdata-dir {target_dir} image.png output")
    print(f"\n   Or set environment variable:")
    print(f"   export TESSDATA_PREFIX={target_dir}")


if __name__ == "__main__":
    try:
        download_khm_data()
    except PermissionError:
        print("Permission denied. Try running with sudo or as administrator.")
    except Exception as e:
        print(f"Error: {e}")

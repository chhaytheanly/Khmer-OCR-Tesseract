"""
Configuration and constants for Khmer OCR system
"""
import os
from pathlib import Path

# Console colors for output
YELLOW = '\033[93m'
GREEN = '\033[92m'
ENDC = '\033[0m'
PINK = '\033[95m'
BLUE = '\033[94m'

# Default paths
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_TESSDATA_DIR = Path.home() / "tessdata"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "").strip()

# OCR Configuration presets
OCR_CONFIGS = {
    'dark_text': [
        r'--oem 3 --psm 6 -l khm',
        r'--oem 3 --psm 4 -l khm',
        r'--oem 3 --psm 13 -l khm',
        r'-l khm+eng --oem 3 --psm 6'
    ],
    'light_text': [
        r'--oem 3 --psm 8 -l khm',  # Single word
        r'--oem 3 --psm 7 -l khm',  # Single line
        r'--oem 3 --psm 13 -l khm',  # Raw line
        r'-l khm+eng --oem 3 --psm 6'
    ],
    'whole_image': [
        r'--oem 3 --psm 6 -l khm+eng',  # Standard
        r'--oem 3 --psm 4 -l khm+eng',  # Multiple blocks
        r'--oem 3 --psm 11 -l khm+eng',  # Sparse text
        r'-l khm+eng --oem 3 --psm 6'
    ],
    'bounding_box': [
        r'--oem 3 --psm 7 -l khm',
        r'--oem 3 --psm 7 -l khm+eng'
    ],
    'line_primary': [
        r'--oem 3 --psm 7 -l khm+eng'
    ],
    'line_fallback': [
        r'--oem 3 --psm 6 -l khm+eng'
    ]
}

# Logo detection parameters
LOGO_DETECTION_PARAMS = {
    'min_size': 5000,
    'max_size': 50000,
    'aspect_ratio_min': 0.3,
    'aspect_ratio_max': 3.0,
    'solidity_threshold': 0.3
}

# Image processing parameters
IMAGE_PROCESSING = {
    'resolution_scale_factor': 2,
    'bilateral_filter_d': 9,
    'bilateral_filter_sigma_color': 75,
    'bilateral_filter_sigma_space': 75,
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'line_padding': 4,
    'line_confidence_threshold': 35,
    'binarization_block_size': 31,
    'binarization_c': 10,
    'line_merge_kernel': (17, 3)
}

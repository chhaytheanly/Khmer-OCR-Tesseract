"""
Configuration and constants for Khmer OCR system
"""
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
    'clahe_tile_grid_size': (8, 8)
}

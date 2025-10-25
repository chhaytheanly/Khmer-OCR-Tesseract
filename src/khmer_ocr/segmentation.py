"""
Text region segmentation for Khmer OCR
Identifies different types of text regions (dark, light, medium)
"""

import cv2
import numpy as np

from .config import PINK, ENDC, IMAGE_PROCESSING


def ensure_grayscale(image):
    """Convert BGR to grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def segment_text_regions(image):
    """Segment text regions using multiple thresholding strategies"""
    print(PINK + "Segmenting text regions..." + ENDC)
    gray = ensure_grayscale(image)
    masks = []

    # Dark text on light background
    _, dark_text_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    masks.append(("dark_text", dark_text_mask))

    # Light text on dark background
    _, light_text_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    masks.append(("light_text", light_text_mask))

    # Medium contrast text
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, medium_text_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    masks.append(("medium_text", medium_text_mask))

    # Clean masks with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    cleaned_masks = []
    for name, mask in masks:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned_masks.append((name, cleaned))

    return cleaned_masks, gray


def handle_handwritten_text(image):
    """Process handwritten text regions with enhanced preprocessing"""
    print(PINK + "Processing handwritten text regions..." + ENDC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.bilateralFilter(
        gray,
        IMAGE_PROCESSING["bilateral_filter_d"],
        IMAGE_PROCESSING["bilateral_filter_sigma_color"],
        IMAGE_PROCESSING["bilateral_filter_sigma_space"],
    )

    # Enhance contrast
    clahe = cv2.createCLAHE(
        clipLimit=IMAGE_PROCESSING["clahe_clip_limit"],
        tileGridSize=IMAGE_PROCESSING["clahe_tile_grid_size"],
    )
    enhanced = clahe.apply(denoised)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )

    return binary

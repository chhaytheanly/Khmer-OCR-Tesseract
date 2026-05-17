"""
Text region segmentation for Khmer OCR
Identifies different types of text regions (dark, light, medium)
and detects word/line bounding boxes with reading order sorting.
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


def _sort_boxes_reading_order(boxes, y_threshold=12):
    """Sort bounding boxes top-to-bottom, left-to-right"""
    if not boxes:
        return boxes

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines = []
    current_line = [boxes[0]]
    current_y = boxes[0][1]

    for box in boxes[1:]:
        x, y, w, h = box
        if abs(y - current_y) <= y_threshold:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
            current_y = y

    lines.append(sorted(current_line, key=lambda b: b[0]))
    return [b for line in lines for b in line]


def _adaptive_binarize(gray):
    """Adaptive binarization for scanned documents."""
    block_size = int(IMAGE_PROCESSING["binarization_block_size"])
    if block_size % 2 == 0:
        block_size += 1
    c = int(IMAGE_PROCESSING["binarization_c"])
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )


def _column_split(boxes, image_width, gap_ratio=0.08):
    """Split boxes into columns using vertical whitespace gaps."""
    if not boxes:
        return [boxes]

    x_centers = sorted([x + w // 2 for x, y, w, h in boxes])
    gaps = []
    for i in range(1, len(x_centers)):
        gaps.append((x_centers[i] - x_centers[i - 1], x_centers[i - 1], x_centers[i]))

    max_gap = max([g[0] for g in gaps], default=0)
    if max_gap < int(image_width * gap_ratio):
        return [boxes]

    split_point = max(gaps, key=lambda g: g[0])[1]
    left = [b for b in boxes if (b[0] + b[2] // 2) <= split_point]
    right = [b for b in boxes if (b[0] + b[2] // 2) > split_point]
    return [left, right]


def detect_text_lines(image):
    """Detect line-level text boxes with padding and column grouping."""
    print(PINK + "Detecting text lines..." + ENDC)

    gray = ensure_grayscale(image)
    binary = _adaptive_binarize(gray)

    kernel_w, kernel_h = IMAGE_PROCESSING["line_merge_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    height, width = gray.shape[:2]
    min_area = max(250, (height * width) // 6000)
    padding = int(IMAGE_PROCESSING["line_padding"])

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 10:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.3 or aspect_ratio > 30:
            continue

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + w + padding)
        y1 = min(height, y + h + padding)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    boxes = _sort_boxes_reading_order(boxes)
    columns = _column_split(boxes, width)

    ordered = []
    for column_boxes in columns:
        ordered.extend(_sort_boxes_reading_order(column_boxes))

    print(PINK + f"Detected {len(ordered)} text lines" + ENDC)
    return ordered


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

"""
Line detection backends for Khmer OCR
Supports OpenCV (default) and Kraken (optional)
"""
from __future__ import annotations

from typing import List, Tuple

import cv2

from .config import PINK, ENDC, LINE_DETECTION_BACKEND
from .segmentation import detect_text_lines
from .layout_detection import detect_layout_regions


def _kraken_available() -> bool:
    try:
        import kraken  # noqa: F401
        return True
    except Exception:
        return False


def _kraken_line_boxes(gray_image) -> List[Tuple[int, int, int, int]]:
    """Detect line boxes using Kraken if available."""
    try:
        from kraken import binarization
        from kraken.pageseg import segment
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Kraken import failed: {exc}")

    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    pil_image = Image.fromarray(gray_image)
    binarized = binarization.nlbin(pil_image)
    segments = segment(binarized)

    boxes = []
    for line in segments.get("lines", []):
        box = line.get("bbox")
        if not box:
            continue
        x0, y0, x1, y1 = box
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))

    return boxes


def detect_lines(gray_image):
    """Detect line boxes using configured backend and optional layout regions."""
    backend = LINE_DETECTION_BACKEND
    if backend == "auto":
        backend = "kraken" if _kraken_available() else "opencv"

    regions = detect_layout_regions(gray_image)
    if not regions:
        if backend == "kraken":
            print(PINK + "Line detection backend: kraken" + ENDC)
            return _kraken_line_boxes(gray_image)

        print(PINK + "Line detection backend: opencv" + ENDC)
        return detect_text_lines(gray_image)

    print(PINK + f"Line detection backend: {backend} with layout regions" + ENDC)
    all_lines = []
    for x, y, w, h in regions:
        roi = gray_image[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        if backend == "kraken":
            lines = _kraken_line_boxes(roi)
        else:
            lines = detect_text_lines(roi)

        for lx, ly, lw, lh in lines:
            all_lines.append((x + lx, y + ly, lw, lh))

    return all_lines

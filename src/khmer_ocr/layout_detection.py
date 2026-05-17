"""
Layout detection using LayoutParser (optional)
Detects document regions for better line segmentation order
"""
from __future__ import annotations

from typing import List, Tuple

import cv2

from .config import PINK, ENDC, USE_LAYOUTPARSER, LAYOUTPARSER_SCORE_THRESHOLD, LAYOUTPARSER_MODEL


def _layoutparser_available() -> bool:
    try:
        import layoutparser  # noqa: F401
        return True
    except Exception:
        return False


def _select_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model():
    import layoutparser as lp

    device = _select_device()
    model = lp.Detectron2LayoutModel(
        LAYOUTPARSER_MODEL,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", LAYOUTPARSER_SCORE_THRESHOLD],
        label_map=None,
        device=device
    )
    return model


def detect_layout_regions(image) -> List[Tuple[int, int, int, int]]:
    """Detect layout regions (x, y, w, h) using LayoutParser when available."""
    if USE_LAYOUTPARSER == "false":
        return []

    if USE_LAYOUTPARSER == "auto" and not _layoutparser_available():
        return []

    print(PINK + "Detecting layout regions with LayoutParser..." + ENDC)

    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    model = _load_model()
    layout = model.detect(bgr)

    regions = []
    for block in layout:
        x1, y1, x2, y2 = block.coordinates
        regions.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    regions = sorted(regions, key=lambda r: (r[1], r[0]))
    return regions

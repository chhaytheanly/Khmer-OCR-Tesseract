"""
Image preprocessing functions for Khmer OCR
Handles loading, logo removal, rotation correction, and noise removal
"""
import cv2
import numpy as np
from scipy import ndimage
from pathlib import Path

from .config import PINK, GREEN, ENDC, LOGO_DETECTION_PARAMS, IMAGE_PROCESSING


def load_image(image_path):
    """Load image from file path"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def detect_and_remove_logos(image, min_logo_size=None, max_logo_size=None):
    """Detect and remove logos/watermarks from image"""
    if min_logo_size is None:
        min_logo_size = LOGO_DETECTION_PARAMS['min_size']
    if max_logo_size is None:
        max_logo_size = LOGO_DETECTION_PARAMS['max_size']
    
    print(PINK + "Detecting and removing logos..." + ENDC)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logo_mask = np.zeros_like(gray)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_logo_size < area < max_logo_size:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if LOGO_DETECTION_PARAMS['aspect_ratio_min'] < aspect_ratio < LOGO_DETECTION_PARAMS['aspect_ratio_max']:
                solidity = area / (w * h)
                if solidity > LOGO_DETECTION_PARAMS['solidity_threshold']:
                    cv2.drawContours(logo_mask, [contour], -1, 255, -1)
    
    if np.sum(logo_mask) > 0:
        image = cv2.inpaint(image, logo_mask, 3, cv2.INPAINT_TELEA)
        print(GREEN + f"Removed {np.sum(logo_mask > 0)} logo pixels" + ENDC)
    else:
        print(GREEN + "No logos detected" + ENDC)
    
    return image


def ensure_3_channel(image):
    """Convert grayscale to 3-channel BGR"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def ensure_grayscale(image):
    """Convert BGR to grayscale"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def advanced_rotation_correction(image):
    """Correct image rotation using Hough line detection"""
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    angles = []
    if lines is not None:
        for i in range(min(20, len(lines))):
            line = lines[i]
            vals = np.asarray(line).ravel()
            if vals.size < 2:
                continue
            rho, theta = vals[0], vals[1]
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)
    
    if angles:
        median_angle = np.median(angles)
        print(GREEN + f"Correcting rotation by {median_angle:.2f} degrees" + ENDC)
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    return image


def resolution(image, scale_factor=None):
    """Upscale image resolution"""
    if scale_factor is None:
        scale_factor = IMAGE_PROCESSING['resolution_scale_factor']
    
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def remove_noise(image):
    """Remove noise using bilateral filter and morphological operations"""
    denoised = cv2.bilateralFilter(
        image, 
        IMAGE_PROCESSING['bilateral_filter_d'],
        IMAGE_PROCESSING['bilateral_filter_sigma_color'],
        IMAGE_PROCESSING['bilateral_filter_sigma_space']
    )
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def preprocess_pipeline(image_path, enhance=True):
    """Complete preprocessing pipeline"""
    print(PINK + "Loading image..." + ENDC)
    image = load_image(image_path)
    print(GREEN + "Image loaded successfully." + ENDC)
    
    if image is None:
        return None, None

    # Logo removal
    print(PINK + "Logo detection and removal..." + ENDC)
    image = detect_and_remove_logos(image)
    print(GREEN + "Logo processing completed." + ENDC)

    # Rotation correction
    print(PINK + "Advanced rotation correction..." + ENDC)
    image = advanced_rotation_correction(image)
    print(GREEN + "Rotation correction completed." + ENDC)

    # Text region segmentation (imported from segmentation module)
    from .segmentation import segment_text_regions
    print(PINK + "Text region segmentation..." + ENDC)
    text_masks, gray_image = segment_text_regions(image)
    print(GREEN + f"Found {len(text_masks)} text region types" + ENDC)

    # Resolution enhancement
    if enhance:
        print(PINK + "Resolution..." + ENDC)
        image = resolution(image)
        print(GREEN + "Image resolution successfully." + ENDC)

    # Noise removal
    print(PINK + "Removing noise..." + ENDC)
    image = remove_noise(image)
    print(GREEN + "Noise removed." + ENDC)

    return image, text_masks

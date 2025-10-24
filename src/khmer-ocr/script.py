import cv2
import numpy as np
import pytesseract
import os
import unicodedata
import re
import json
from datetime import datetime

# --- Configuration ---
# Load settings from a config file if it exists, otherwise use defaults
CONFIG_FILE = "ocr_config.json"
DEFAULT_CONFIG = {
    "logo_removal": {
        "min_logo_size": 5000,
        "max_logo_size": 50000,
        "aspect_ratio_min": 0.3,
        "aspect_ratio_max": 3.0,
        "solidity_threshold": 0.3
    },
    "rotation_correction": {
        "canny_low": 50,
        "canny_high": 150,
        "hough_threshold": 100,
        "hough_lines_to_check": 20,
        "min_angle": -45,
        "max_angle": 45,
        "min_lines_for_correction": 5
    },
    "preprocessing": {
        "enhance_resolution": True,
        "scale_factor": 2.0,
        "bilateral_filter_d": 9,
        "bilateral_filter_sigma_color": 75,
        "bilateral_filter_sigma_space": 75,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "adaptive_thresh_block_size": 35,
        "adaptive_thresh_c": 15
    },
    "ocr": {
        "psm_modes": {
            "dark_text": [6, 4, 8, 7, 13],
            "light_text": [6, 4, 8, 7, 13],
            "medium_text": [6, 4, 8, 7, 13]
        },
        "confidence_threshold": 40.0,
        "language": "khm+eng"
    },
    "output": {
        "base_dir": "/media/chhaythean/Drive D/Ai-Edu/data/images/output",
        "save_intermediate_images": True
    }
}

def load_config(config_file):
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        # Merge with defaults to ensure all keys exist
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    else:
        print(f"Config file {config_file} not found, using defaults.")
        return DEFAULT_CONFIG

CONFIG = load_config(CONFIG_FILE)

# Color codes for printing (if needed)
PINK = '\033[95m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

# --- Helper Functions ---
def ensure_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def safe_bitwise(image, mask):
    """Safely apply a mask to an image."""
    if len(image.shape) == 3:
        image_3ch = image
    else:
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if mask.shape[:2] != image_3ch.shape[:2]:
        mask = cv2.resize(mask, (image_3ch.shape[1], image_3ch.shape[0]))

    mask = mask.astype(np.uint8)
    return cv2.bitwise_and(image_3ch, image_3ch, mask=mask)

# --- Image Loading and Normal Preprocessing ---
def load_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        else:
            return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def detect_and_remove_logos(image):
    print(PINK + "Detecting and removing logos..." + ENDC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use a slightly different thresholding method for potentially better logo separation
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logo_mask = np.zeros_like(gray)

    params = CONFIG["logo_removal"]
    for contour in contours:
        area = cv2.contourArea(contour)
        if params["min_logo_size"] < area < params["max_logo_size"]:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if params["aspect_ratio_min"] < aspect_ratio < params["aspect_ratio_max"]:
                solidity = area / (w * h)
                if solidity > params["solidity_threshold"]:
                    cv2.drawContours(logo_mask, [contour], -1, 255, -1)

    if np.sum(logo_mask) > 0:
        image = cv2.inpaint(image, logo_mask, 3, cv2.INPAINT_TELEA)
        print(GREEN + f"Removed {np.sum(logo_mask > 0)} logo pixels" + ENDC)
    else:
        print(GREEN + "No logos detected" + ENDC)

    return image

def ensure_3_channel(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

# --- Text Region Recognition and Segmentation ---
def segment_text_regions(image):
    print(PINK + "Segmenting text regions..." + ENDC)
    gray = ensure_grayscale(image)
    masks = []

    # Otsu thresholds for dark and light text
    _, dark_text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(('dark_text', dark_text_mask))

    _, light_text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(('light_text', light_text_mask))

    # Blurred version for potentially medium contrast
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, medium_text_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(('medium_text', medium_text_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_masks = []
    for name, mask in masks:
        # Apply morphological operations to clean up the mask
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned_masks.append((name, cleaned))

    return cleaned_masks, gray

# --- Handwritten Text Handling ---
def handle_handwritten_text(image):
    print(PINK + "Processing handwritten text regions..." + ENDC)

    gray = ensure_grayscale(image)
    params = CONFIG["preprocessing"]

    # Denoise using bilateral filter
    denoised = cv2.bilateralFilter(gray, params["bilateral_filter_d"], params["bilateral_filter_sigma_color"], params["bilateral_filter_sigma_space"])
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"], tileGridSize=params["clahe_tile_grid_size"])
    enhanced = clahe.apply(denoised)
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, params["adaptive_thresh_block_size"], params["adaptive_thresh_c"])

    return binary

# --- Advanced Rotation Correction ---
def advanced_rotation_correction(image):
    print(PINK + "Performing advanced rotation correction..." + ENDC)
    gray = ensure_grayscale(image)
    params = CONFIG["rotation_correction"]

    edges = cv2.Canny(gray, params["canny_low"], params["canny_high"], apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=params["hough_threshold"])

    angles = []
    if lines is not None and len(lines) >= params["min_lines_for_correction"]:
        for i in range(min(params["hough_lines_to_check"], len(lines))):
            if lines[i] is not None:
                rho, theta = lines[i][0]
                angle = np.degrees(theta) - 90
                if params["min_angle"] <= angle <= params["max_angle"]:
                    angles.append(angle)

    if angles:
        median_angle = np.median(angles)
        print(GREEN + f"Correcting rotation by {median_angle:.2f} degrees" + ENDC)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    else:
        print(GREEN + "Insufficient lines detected for reliable rotation correction. Skipping." + ENDC)

    return image

def resolution(image, scale_factor=None):
    if scale_factor is None:
        scale_factor = CONFIG["preprocessing"]["scale_factor"]
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def remove_noise(image):
    params = CONFIG["preprocessing"]
    denoised = cv2.bilateralFilter(image, params["bilateral_filter_d"], params["bilateral_filter_sigma_color"], params["bilateral_filter_sigma_space"])
    # A slightly larger kernel might be better for noise removal
    kernel = np.ones((2, 2), np.uint8) # Changed from 1x1
    cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

# --- OCR Functions ---
def region_specific_ocr(image, text_masks):
    print(PINK + "Performing region-specific OCR..." + ENDC)
    all_results = []

    for region_name, mask in text_masks:
        print(BLUE + f"Processing {region_name}..." + ENDC)

        try:
            masked_image = safe_bitwise(image, mask)
            if len(masked_image.shape) == 3:
                masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            else:
                masked_gray = masked_image
        except Exception as e:
            print(f"Error processing {region_name}: {e}")
            continue

        # Use PSM modes specific to region type if configured, otherwise default
        psm_modes = CONFIG["ocr"]["psm_modes"].get(region_name, CONFIG["ocr"]["psm_modes"]["dark_text"])
        configs = [f'--oem 3 --psm {psm} -l {CONFIG["ocr"]["language"]}' for psm in psm_modes]
        # Also add a default config
        configs.append(f'-l {CONFIG["ocr"]["language"]} --oem 3 --psm 6')

        region_results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(masked_gray, config=config)
                confidence_data = pytesseract.image_to_data(masked_gray, config=config, output_type=pytesseract.Output.DICT)

                confidences = [int(c) for c in confidence_data['conf'] if int(c) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0

                if text.strip():
                    region_results.append({
                        'config': config,
                        'text': text.strip(),
                        'confidence': avg_confidence
                    })
            except Exception as e:
                print(f"OCR error with config {config}: {e}")
                continue

        if region_results:
            best_region = max(region_results, key=lambda x: x['confidence'])
            all_results.append({
                'region': region_name,
                'text': best_region['text'],
                'confidence': best_region['confidence']
            })
            print(f"  {region_name}: {len(best_region['text'])} chars, confidence: {best_region['confidence']:.2f}")

    combined_text = ' '.join([r['text'] for r in all_results if r['text']])
    overall_confidence = np.mean([r['confidence'] for r in all_results]) if all_results else 0

    print(GREEN + f"Combined {len(all_results)} regions, confidence: {overall_confidence:.2f}" + ENDC)

    return {
        'combined_text': combined_text,
        'region_results': all_results,
        'confidence': overall_confidence
    }

def multi_pass_ocr(image, text_masks):
    print(YELLOW + "Starting multi-pass OCR..." + ENDC)

    # Pass 1: Region-specific OCR
    region_results = region_specific_ocr(image, text_masks)
    region_combined_text = region_results.get('combined_text', '')
    region_confidence = region_results.get('confidence', 0)

    # Pass 2: Whole image OCR
    whole_image_configs = [
        f'--oem 3 --psm 6 -l {CONFIG["ocr"]["language"]}',
        f'--oem 3 --psm 4 -l {CONFIG["ocr"]["language"]}',
        f'--oem 3 --psm 11 -l {CONFIG["ocr"]["language"]}',
        f'-l {CONFIG["ocr"]["language"]} --oem 3 --psm 6'
    ]

    whole_image_results = []
    for config in whole_image_configs:
        try:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            text = pytesseract.image_to_string(gray_image, config=config)
            confidence_data = pytesseract.image_to_data(gray_image, config=config, output_type=pytesseract.Output.DICT)

            confs = [int(c) for c in confidence_data['conf'] if int(c) > 0]
            avg_confidence = np.mean(confs) if confs else 0

            if text.strip():
                whole_image_results.append({
                    'config': config,
                    'text': text.strip(),
                    'confidence': avg_confidence
                })
        except Exception as e:
            print(f"Whole image OCR error: {e}")
            continue

    best_whole = max(whole_image_results, key=lambda x: x['confidence']) if whole_image_results else {'text': '', 'confidence': 0}

    # Combine and select best result based on confidence
    all_texts = [region_combined_text, best_whole['text']]
    all_confidences = [region_confidence, best_whole['confidence']]

    if not any(all_confidences):
        final_text = ''
        final_confidence = 0
        result_type = "none"
    else:
        best_idx = int(np.argmax(all_confidences))
        final_text = all_texts[best_idx]
        final_confidence = float(all_confidences[best_idx])
        result_type = "region_based" if best_idx == 0 else "whole_image"

    print(GREEN + f"Selected {result_type} result with confidence: {final_confidence:.2f}" + ENDC)

    return {
        'text': final_text,
        'confidence': final_confidence,
        'region_results': region_results.get('region_results', []),
        'whole_image_results': whole_image_results
    }

# --- Post-Processing Functions ---
def normalize_khmer_unicode(text):
    normalized = unicodedata.normalize('NFC', text)
    # Expanded corrections based on common OCR errors or inconsistencies
    error_corrections = {
        'ាឹ': 'ា',
        'ិះ': 'ិ',
        'ុះ': 'ុ',
        '៉ា': 'ា',
        # Add more specific corrections here if known
    }
    for error, correction in error_corrections.items():
        normalized = normalized.replace(error, correction)
    return normalized

def correct_khmer_spacing(text):
    khmer_pattern = r'([\u1780-\u17FF]+)\s+([\u1780-\u17FF]+)'
    corrected = re.sub(khmer_pattern, r'\1\2', text)
    return corrected.strip()

def numbers_to_khmer(text):
    arabic_digits = '0123456789'
    khmer_digits = '០១២៣៤៥៦៧៨៩'
    translation_table = str.maketrans(arabic_digits, khmer_digits)
    return text.translate(translation_table)

def expand_khmer_abbreviations(text):
    # Expanded dictionary - add more common abbreviations
    abbreviations = {
        'គ.ស.': 'គ្រិស្តសករាជ',
        'ម.រ.': 'មុនគ្រិស្តសករាជ',
        'រ.ដ.': 'រដ្ឋាភិបាល',
        'ឯ.អ.': 'ឯកអគ្គ',
        'អ.ដ.': 'អនុដ្ឋាន',
        'ស.រ.': 'សាធារណរដ្ឋ',
        'ស.វ.': 'សាកលវិទ្យាល័យ',
        'ក.ប.': 'ក្រសួងបរិស្ថាន',
        'ម.ស.': 'មត្តសម្បទា',
        'ថ្ងៃទី': 'ថ្ងៃទី',
        'ម៉ោង': 'ម៉ោង',
        # Add more as needed
    }
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    return text

def spell_check_khmer(text, custom_dict=None):
    # Expanded dictionary of common errors - add more based on analysis
    common_errors = {
        'មហាុ': 'មហា',
        'ប្រទេសជ': 'ប្រទេស',
        'កមពុជ': 'កម្ពុជា',
        'បរជាតិយ': 'ប្រជាជាតិ',
        'សកលវិទ្យាលយ': 'សកលវិទ្យាល័យ',
        'អង្គរវត្ត្': 'អង្គរវត្ត',
        'បាំបាំ': 'ប៉ាប៉ា',
        'សាលារៀន': 'សាលារៀន',
        'ប្រធានបទ': 'ប្រធាឨបទ', # Example of a typo correction
        # Add more specific errors found in OCR output
    }
    for error, correction in common_errors.items():
        text = text.replace(error, correction)
    return text

def postprocess_pipeline(text):
    text = normalize_khmer_unicode(text)
    text = correct_khmer_spacing(text)
    text = numbers_to_khmer(text)
    text = expand_khmer_abbreviations(text)
    text = spell_check_khmer(text)

    # Additional cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()

    return text

# --- Main Preprocessing Pipeline ---
def preprocess_pipeline(image_path, enhance=None):
    if enhance is None:
        enhance = CONFIG["preprocessing"]["enhance_resolution"]

    print(PINK + "Loading image..." + ENDC)
    image = load_image(image_path)
    if image is None:
        return None, None
    print(GREEN + "Image loaded successfully." + ENDC)

    print(PINK + "Logo detection and removal..." + ENDC)
    image = detect_and_remove_logos(image)
    print(GREEN + "Logo processing completed." + ENDC)

    print(PINK + "Advanced rotation correction..." + ENDC)
    image = advanced_rotation_correction(image)
    print(GREEN + "Rotation correction completed." + ENDC)

    print(PINK + "Text region segmentation..." + ENDC)
    text_masks, gray_image = segment_text_regions(image)
    print(GREEN + f"Found {len(text_masks)} text region types" + ENDC)

    print(PINK + "Resolution enhancement..." + ENDC)
    if enhance:
        image = resolution(image)
    print(GREEN + "Image resolution adjusted." + ENDC)

    print(PINK + "Removing noise..." + ENDC)
    image = remove_noise(image)
    print(GREEN + "Noise removed." + ENDC)

    return image, text_masks

# --- Main Processing Workflow ---
def process_document(image_path, output_dir=None):
    if output_dir is None:
        output_dir = CONFIG["output"]["base_dir"]

    print(YELLOW + "    Starting Enhanced Document Processing    " + ENDC)
    os.makedirs(output_dir, exist_ok=True)

    # Generate unique output directory name based on timestamp and image name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    run_output_dir = os.path.join(output_dir, f"{image_name}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    try:
        # Preprocessing
        preprocessed_image, text_masks = preprocess_pipeline(image_path)
        if preprocessed_image is None:
            return None

        # Save preprocessed image if configured
        if CONFIG["output"]["save_intermediate_images"]:
            preprocessed_path = os.path.join(run_output_dir, "preprocessed.png")
            cv2.imwrite(preprocessed_path, preprocessed_image)

        # OCR
        print(YELLOW + "Performing multi-pass OCR..." + ENDC)
        ocr_result = multi_pass_ocr(preprocessed_image, text_masks)
        confidence = ocr_result['confidence']
        print(GREEN + f"OCR completed with confidence: {confidence:.2f}" + ENDC)

        # Confidence check
        if confidence < CONFIG["ocr"]["confidence_threshold"]:
             print(YELLOW + f"Warning: OCR confidence ({confidence:.2f}) is below threshold ({CONFIG['ocr']['confidence_threshold']}). Result might be unreliable." + ENDC)

        # Post-processing
        print(YELLOW + "Post-processing..." + ENDC)
        cleaned_text = postprocess_pipeline(ocr_result['text'])

        # Compile results
        results = {
            'original_text': ocr_result['text'],
            'cleaned_text': cleaned_text,
            'preprocessed_image': os.path.join(run_output_dir, "preprocessed.png") if CONFIG["output"]["save_intermediate_images"] else None,
            'confidence': confidence,
            'region_results': ocr_result.get('region_results', []),
            'whole_image_results': ocr_result.get('whole_image_results', [])
        }

        # Save outputs
        with open(os.path.join(run_output_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        with open(os.path.join(run_output_dir, "ocr_details.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(GREEN + "             Processing Completed           " + ENDC)
        print(GREEN + f"Extracted {len(cleaned_text)} characters with confidence {confidence:.2f}" + ENDC)
        print(GREEN + f"Results saved in: {run_output_dir}" + ENDC)

        return results

    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage:
result = process_document("/media/chhaythean/Drive D/Ai-Edu/data/images/itc.png")
print(result['cleaned_text'])
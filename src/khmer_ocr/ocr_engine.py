"""
OCR engine for Khmer text extraction
Handles multi-pass OCR with region-specific and whole-image strategies
"""
import cv2
import numpy as np
import pytesseract

from .config import PINK, GREEN, BLUE, YELLOW, ENDC, OCR_CONFIGS, TESSERACT_CMD, IMAGE_PROCESSING
from .segmentation import detect_text_lines

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def safe_bitwise(image, mask):
    """Safely apply bitwise AND with mask"""
    # Ensure image is 3-channel
    if len(image.shape) == 3:
        image_3ch = image
    else:
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Ensure mask is the same size as image
    if mask.shape[:2] != image_3ch.shape[:2]:
        mask = cv2.resize(mask, (image_3ch.shape[1], image_3ch.shape[0]))
    
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    return cv2.bitwise_and(image_3ch, image_3ch, mask=mask)


def region_specific_ocr(image, text_masks):
    """Perform OCR on specific text regions"""
    print(PINK + "Performing region-specific OCR..." + ENDC)
    
    all_results = []
    
    for region_name, mask in text_masks:
        print(BLUE + f"Processing {region_name}..." + ENDC)
        
        # Apply mask to original image safely
        try:
            masked_image = safe_bitwise(image, mask)

            if len(masked_image.shape) == 3:
                masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            else:
                masked_gray = masked_image
        
        except Exception as e:
            print(f"Error processing {region_name}: {e}")
            continue
        
        # Select OCR configs based on region type
        if 'dark_text' in region_name:
            configs = OCR_CONFIGS['dark_text']
        else:
            configs = OCR_CONFIGS['light_text']
        
        region_results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(masked_gray, config=config)
                confidence_data = pytesseract.image_to_data(
                    masked_gray, config=config, output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [int(c) for c in confidence_data['conf'] if int(c) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                if text.strip():  # Only consider non-empty results
                    region_results.append({
                        'config': config,
                        'text': text.strip(),
                        'confidence': avg_confidence
                    })
            except Exception as e:
                print(f"OCR error with config {config}: {e}")
                continue
        
        # Select best result for this region
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


def line_ocr(image):
    """Perform OCR on detected line boxes with confidence fallback."""
    print(PINK + "Performing line OCR..." + ENDC)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lines = detect_text_lines(gray)
    primary_configs = OCR_CONFIGS.get('line_primary', [])
    fallback_configs = OCR_CONFIGS.get('line_fallback', [])
    confidence_threshold = IMAGE_PROCESSING.get('line_confidence_threshold', 35)

    all_results = []
    for line in lines:
        x, y, w, h = line
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        line_results = []
        for config in primary_configs:
            try:
                text = pytesseract.image_to_string(roi, config=config)
                confidence_data = pytesseract.image_to_data(
                    roi, config=config, output_type=pytesseract.Output.DICT
                )

                confs = []
                for c in confidence_data.get('conf', []):
                    try:
                        cv = int(float(c))
                        if cv > 0:
                            confs.append(cv)
                    except Exception:
                        continue

                avg_confidence = float(np.mean(confs)) if confs else 0

                if text.strip():
                    line_results.append({
                        'config': config,
                        'text': text.strip(),
                        'confidence': avg_confidence
                    })
            except Exception as e:
                print(f"Line OCR error with config {config}: {e}")
                continue

        best_line = max(line_results, key=lambda x: x['confidence']) if line_results else None

        if best_line and best_line['confidence'] < confidence_threshold:
            for config in fallback_configs:
                try:
                    text = pytesseract.image_to_string(roi, config=config)
                    confidence_data = pytesseract.image_to_data(
                        roi, config=config, output_type=pytesseract.Output.DICT
                    )

                    confs = []
                    for c in confidence_data.get('conf', []):
                        try:
                            cv = int(float(c))
                            if cv > 0:
                                confs.append(cv)
                        except Exception:
                            continue

                    avg_confidence = float(np.mean(confs)) if confs else 0

                    if text.strip():
                        line_results.append({
                            'config': config,
                            'text': text.strip(),
                            'confidence': avg_confidence
                        })
                except Exception as e:
                    print(f"Line OCR fallback error with config {config}: {e}")
                    continue

            best_line = max(line_results, key=lambda x: x['confidence']) if line_results else best_line

        if best_line:
            all_results.append({
                'line': line,
                'text': best_line['text'],
                'confidence': best_line['confidence']
            })

    combined_text = '\n'.join([r['text'] for r in all_results if r['text']])
    overall_confidence = np.mean([r['confidence'] for r in all_results]) if all_results else 0

    print(GREEN + f"Combined {len(all_results)} lines, confidence: {overall_confidence:.2f}" + ENDC)

    return {
        'combined_text': combined_text,
        'line_results': all_results,
        'confidence': overall_confidence
    }


def multi_pass_ocr(image, text_masks):
    """Accuracy-first OCR: line OCR by default with whole-image fallback."""
    print(YELLOW + "Starting accuracy-first OCR..." + ENDC)

    # Pass 1: Line OCR (primary)
    line_results = line_ocr(image)
    line_list = line_results.get('line_results', []) if isinstance(line_results, dict) else line_results
    line_combined_text = line_results.get('combined_text', '') if isinstance(line_results, dict) else '\n'.join([r.get('text', '') for r in line_list])
    line_confidence = line_results.get('confidence', 0) if isinstance(line_results, dict) else (np.mean([r.get('confidence', 0) for r in line_list]) if line_list else 0)

    # Pass 2: Whole image OCR (fallback)
    whole_image_configs = OCR_CONFIGS['whole_image']

    whole_image_results = []
    for config in whole_image_configs:
        try:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            text = pytesseract.image_to_string(gray_image, config=config)
            confidence_data = pytesseract.image_to_data(
                gray_image, config=config, output_type=pytesseract.Output.DICT
            )

            confs = []
            for c in confidence_data.get('conf', []):
                try:
                    cv = int(float(c))
                    if cv > 0:
                        confs.append(cv)
                except Exception:
                    continue

            avg_confidence = float(np.mean(confs)) if confs else 0

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

    all_texts = [line_combined_text, best_whole['text']]
    all_confidences = [line_confidence, best_whole['confidence']]

    if not any(all_confidences):
        final_text = ''
        final_confidence = 0
        result_type = "none"
    else:
        best_idx = int(np.argmax(all_confidences))
        final_text = all_texts[best_idx]
        final_confidence = float(all_confidences[best_idx])
        result_type = "line" if best_idx == 0 else "whole_image"

    print(GREEN + f"Selected {result_type} result with confidence: {final_confidence:.2f}" + ENDC)

    return {
        'text': final_text,
        'confidence': final_confidence,
        'result_type': result_type,
        'line_results': line_list,
        'whole_image_results': whole_image_results
    }

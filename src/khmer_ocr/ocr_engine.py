"""
OCR engine for Khmer text extraction
Handles multi-pass OCR with region-specific and whole-image strategies
"""
import cv2
import numpy as np
import pytesseract

from .config import PINK, GREEN, BLUE, YELLOW, ENDC, OCR_CONFIGS


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


def multi_pass_ocr(image, text_masks):
    """Multi-pass OCR strategy combining region-based and whole-image OCR"""
    print(YELLOW + "Starting multi-pass OCR..." + ENDC)

    # Pass 1: Region-specific OCR
    region_results = region_specific_ocr(image, text_masks)
    region_list = region_results.get('region_results', []) if isinstance(region_results, dict) else region_results
    region_combined_text = region_results.get('combined_text', '') if isinstance(region_results, dict) else ' '.join([r.get('text', '') for r in region_list])
    region_confidence = region_results.get('confidence', 0) if isinstance(region_results, dict) else (np.mean([r.get('confidence', 0) for r in region_list]) if region_list else 0)

    # Pass 2: Whole image OCR
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

    # Select best whole image result
    best_whole = max(whole_image_results, key=lambda x: x['confidence']) if whole_image_results else {'text': '', 'confidence': 0}

    # Combine results - use highest confidence
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
        'region_results': region_list,
        'whole_image_results': whole_image_results
    }

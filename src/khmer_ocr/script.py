"""
Main script for Khmer OCR document processing
Orchestrates the entire pipeline from preprocessing to output
"""
import os
import cv2
from pathlib import Path

from .config import YELLOW, GREEN, ENDC, DEFAULT_OUTPUT_DIR
from .preprocessing import preprocess_pipeline
from .ocr_engine import multi_pass_ocr
from .postprocess import postprocess_pipeline


def process_document(image_path, output_dir=None):
    """
    Process a document image through the complete Khmer OCR pipeline
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs (default: ./output)
    
    Returns:
        Dictionary containing results and metadata
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    output_dir = Path(output_dir)
    print(YELLOW + " Starting Enhanced Document Processing " + ENDC)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Preprocessing
        preprocessed_image, text_masks = preprocess_pipeline(image_path)
        if preprocessed_image is None:
            return None

        # Save preprocessed image
        preprocessed_path = output_dir / "preprocessed.png"
        cv2.imwrite(str(preprocessed_path), preprocessed_image)

        # OCR with region processing
        print(YELLOW + "Performing multi-pass OCR..." + ENDC)
        ocr_result = multi_pass_ocr(preprocessed_image, text_masks)
        print(GREEN + f"OCR completed with confidence: {ocr_result['confidence']:.2f}" + ENDC)

        # Post-processing
        print(YELLOW + "Post-processing..." + ENDC)
        preserve_lines = ocr_result.get('result_type') == 'line'
        cleaned_text = postprocess_pipeline(ocr_result['text'], preserve_lines=preserve_lines)

        # Save detailed results
        results = {
            'original_text': ocr_result['text'],
            'cleaned_text': cleaned_text,
            'preprocessed_image': str(preprocessed_path),
            'confidence': ocr_result['confidence'],
            'result_type': ocr_result.get('result_type', 'unknown'),
            'line_results': ocr_result.get('line_results', []),
            'whole_image_results': ocr_result.get('whole_image_results', [])
        }

        # Save output
        with open(output_dir / "extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Save region analysis
        with open(output_dir / "region_analysis.txt", "w", encoding="utf-8") as f:
            f.write(f"Result Type: {results.get('result_type', 'unknown')}\n")
            f.write("-" * 50 + "\n")

            if results.get('line_results'):
                f.write("Line Results:\n")
                for line_result in results.get('line_results', []):
                    x, y, w, h = line_result['line']
                    f.write(f"Line: x={x}, y={y}, w={w}, h={h}\n")
                    f.write(f"Confidence: {line_result['confidence']:.2f}\n")
                    f.write(f"Text: {line_result['text']}\n")
                    f.write("-" * 50 + "\n")

        print(GREEN + " Processing Completed " + ENDC)
        print(GREEN + f"Extracted {len(cleaned_text)} characters with confidence {results['confidence']:.2f}" + ENDC)

        return results

    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Khmer OCR Document Processing")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("--no-enhance", action="store_true", help="Disable resolution enhancement")
    
    args = parser.parse_args()
    
    result = process_document(args.image_path, args.output)
    
    if result:
        print(f"\n Success! Extracted text saved to: {Path(args.output or DEFAULT_OUTPUT_DIR) / 'extracted_text.txt'}")
    else:
        print("\n Processing failed")
        exit(1)


if __name__ == "__main__":
    main()

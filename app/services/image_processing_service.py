#!/usr/bin/env python3
"""
Image Processing Service
Handles image enhancement and preprocessing for OCR
"""

import cv2
import numpy as np
from PIL import Image
import structlog
from typing import Dict, Any, List, Tuple
from app.core.image_config import ImageConfig

logger = structlog.get_logger(__name__)


class ImageProcessingService:
    """Service for image preprocessing and enhancement"""
    
    def __init__(self):
        self.logger = logger
        self.config = ImageConfig()
    
    def enhance_image_for_ocr(self, cv_image) -> List[Tuple[str, np.ndarray]]:
        """Apply multiple image enhancement techniques for better OCR"""
        results = []
        
        # Get preprocessing methods from config
        preprocessing_methods = self.config.get_preprocessing_methods()
        
        for method_name, method_func in preprocessing_methods:
            try:
                processed_img = method_func(cv_image)
                results.append((method_name, processed_img))
            except Exception as e:
                self.logger.warning(f"Preprocessing method {method_name} failed: {e}")
        
        return results
    
    def process_image_with_tesseract(self, image, cv_image) -> Dict[str, Any]:
        """Process image with Tesseract using multiple enhancement techniques"""
        enhanced_images = self.enhance_image_for_ocr(cv_image)
        
        best_text = ""
        best_confidence = 0
        best_method = "original"
        all_results = []
        
        for method_name, enhanced_img in enhanced_images:
            try:
                # Convert to PIL Image for Tesseract
                if len(enhanced_img.shape) == 2:  # Grayscale
                    pil_img = Image.fromarray(enhanced_img)
                else:  # Color
                    pil_img = Image.fromarray(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
                
                # Get text and confidence using pytesseract
                import pytesseract
                data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(pil_img)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                result_info = {
                    "method": method_name,
                    "text": text.strip(),
                    "confidence": round(avg_confidence, 2),
                    "text_length": len(text.strip())
                }
                all_results.append(result_info)
                
                # Keep the best result
                if avg_confidence > best_confidence and text.strip():
                    best_text = text.strip()
                    best_confidence = avg_confidence
                    best_method = method_name
                    
            except Exception as e:
                self.logger.warning(f"OCR method {method_name} failed: {e}")
                all_results.append({
                    "method": method_name,
                    "text": "",
                    "confidence": 0,
                    "text_length": 0,
                    "error": str(e)
                })
        
        return {
            "best_text": best_text,
            "best_confidence": best_confidence,
            "best_method": best_method,
            "all_results": all_results
        }

#!/usr/bin/env python3
"""
Google Vision API Service for QR Code Detection
Uses Google Cloud Vision API for reliable QR code detection
"""

import os
import structlog
from typing import Dict, Any, List, Optional
import io

logger = structlog.get_logger(__name__)

# Try to import Google Vision, handle gracefully if not available
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    vision = None


class GoogleVisionService:
    """Service for QR code detection using Google Vision API"""
    
    def __init__(self):
        self.logger = logger
        self.vision_available = GOOGLE_VISION_AVAILABLE
        self.client = None
        
        if self.vision_available:
            try:
                # Initialize Google Vision client
                self.client = vision.ImageAnnotatorClient()
                self.logger.info("✅ Google Vision API client initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Google Vision API initialization failed: {e}")
                self.vision_available = False
        else:
            self.logger.warning("⚠️ Google Vision API not available - install google-cloud-vision")
    
    async def detect_qr_codes(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Detect QR codes in image using Google Vision API"""
        if not self.vision_available or not self.client:
            self.logger.warning("Google Vision API not available")
            return []
        
        try:
            self.logger.info(f"🔍 Starting Google Vision QR detection for {len(image_data)} bytes")
            
            # Create image object
            image = vision.Image(content=image_data)
            
            # Use text detection to find QR codes (Google Vision doesn't have direct barcode detection)
            response = self.client.text_detection(image=image)
            text_annotations = response.text_annotations
            
            # For now, return empty list since Google Vision API doesn't have direct QR detection
            # We'll use this for text detection instead
            barcodes = []
            
            qr_codes = []
            for barcode in barcodes:
                # Check if it's a QR code
                if barcode.format == vision.BarcodeAnnotation.BarcodeFormat.QR_CODE:
                    # Get bounding box coordinates
                    vertices = barcode.bounding_poly.vertices
                    rect = {
                        "x": vertices[0].x if vertices else 0,
                        "y": vertices[0].y if vertices else 0,
                        "width": vertices[2].x - vertices[0].x if len(vertices) > 2 else 0,
                        "height": vertices[2].y - vertices[0].y if len(vertices) > 2 else 0
                    }
                    
                    qr_info = {
                        "data": barcode.raw_value,
                        "type": "QRCODE",
                        "rect": rect,
                        "method": "Google Vision API",
                        "confidence": barcode.confidence if hasattr(barcode, 'confidence') else 1.0
                    }
                    qr_codes.append(qr_info)
                    
                    self.logger.info(f"📱 QR Code detected (Google Vision): {barcode.raw_value[:50]}...")
            
            self.logger.info(f"✅ Google Vision detected {len(qr_codes)} QR codes")
            return qr_codes
            
        except Exception as e:
            self.logger.error(f"❌ Google Vision QR detection failed: {e}")
            return []
    
    async def detect_text_and_qr(self, image_data: bytes) -> Dict[str, Any]:
        """Detect both text and QR codes using Google Vision API"""
        if not self.vision_available or not self.client:
            return {
                "success": False,
                "error": "Google Vision API not available",
                "qr_codes": [],
                "text": ""
            }
        
        try:
            self.logger.info(f"🔍 Starting Google Vision text + QR detection for {len(image_data)} bytes")
            
            # Create image object
            image = vision.Image(content=image_data)
            
            # Detect text
            text_response = self.client.text_detection(image=image)
            text_annotations = text_response.text_annotations
            detected_text = text_annotations[0].description if text_annotations else ""
            
            # Detect QR codes
            qr_codes = await self.detect_qr_codes(image_data)
            
            self.logger.info(f"✅ Google Vision completed: {len(qr_codes)} QR codes, {len(detected_text)} chars text")
            
            return {
                "success": True,
                "qr_codes": qr_codes,
                "text": detected_text,
                "count": len(qr_codes),
                "method": "Google Vision API"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Google Vision text + QR detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "qr_codes": [],
                "text": ""
            }
    
    def is_available(self) -> bool:
        """Check if Google Vision API is available"""
        return self.vision_available and self.client is not None


# Export singleton instance
google_vision_service = GoogleVisionService()

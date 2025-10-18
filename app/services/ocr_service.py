"""
Enhanced OCR service with multiple engines and vision analysis
"""

import structlog
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from PIL import Image
import io
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from urllib.parse import urlparse

# OCR imports
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# QR Code imports
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

# OpenCV QR Code detector as fallback
try:
    qr_detector = cv2.QRCodeDetector()
    OPENCV_QR_AVAILABLE = True
except Exception:
    OPENCV_QR_AVAILABLE = False

# Enhanced QR Service
try:
    from .enhanced_qr_service import EnhancedQRService
    ENHANCED_QR_AVAILABLE = True
except ImportError:
    ENHANCED_QR_AVAILABLE = False

logger = structlog.get_logger(__name__)


class OCRService:
    """Enhanced OCR service with multiple engines and vision analysis"""
    
    def __init__(self):
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Initialize OCR engines
        self._init_engines()
        
        # Initialize vision service for business card analysis
        try:
            from .vision_service import VisionService
            self.vision_service = VisionService()
            self.vision_available = True
            self.logger.info("✅ Vision service initialized")
        except Exception as e:
            self.vision_service = None
            self.vision_available = False
            self.logger.warning("⚠️ Vision service not available", error=str(e))
        
        # Initialize Enhanced QR Service
        try:
            if ENHANCED_QR_AVAILABLE:
                self.enhanced_qr_service = EnhancedQRService()
                self.enhanced_qr_available = True
                self.logger.info("✅ Enhanced QR service initialized")
            else:
                self.enhanced_qr_service = None
                self.enhanced_qr_available = False
                self.logger.warning("⚠️ Enhanced QR service not available")
        except Exception as e:
            self.enhanced_qr_service = None
            self.enhanced_qr_available = False
            self.logger.warning("⚠️ Enhanced QR service initialization failed", error=str(e))
    
    def _init_engines(self):
        """Initialize available OCR engines"""
        self.engines = {}
        
        # Tesseract
        if TESSERACT_AVAILABLE:
            try:
                self.engines['tesseract'] = True
                self.logger.info("Tesseract OCR initialized")
            except Exception as e:
                self.logger.warning("Tesseract not available", error=str(e))
        
        # EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                self.engines['easyocr'] = True
                self.logger.info("EasyOCR initialized")
            except Exception as e:
                self.logger.warning("EasyOCR not available", error=str(e))
        
        # PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
                self.engines['paddleocr'] = True
                self.logger.info("PaddleOCR initialized")
            except Exception as e:
                self.logger.warning("PaddleOCR not available", error=str(e))
        
        # QR Code scanning
        if PYZBAR_AVAILABLE:
            try:
                self.engines['qr_scanner'] = True
                self.logger.info("QR Code scanner (pyzbar) initialized")
            except Exception as e:
                self.logger.warning("QR Code scanner (pyzbar) not available", error=str(e))
        elif OPENCV_QR_AVAILABLE:
            try:
                self.engines['qr_scanner'] = True
                self.logger.info("QR Code scanner (OpenCV) initialized")
            except Exception as e:
                self.logger.warning("QR Code scanner (OpenCV) not available", error=str(e))
    
    async def process_image(self, image_data: bytes, engine: str = 'auto') -> Dict[str, Any]:
        """Process image with specified OCR engine"""
        self.logger.info("🚀 Starting OCR processing", 
                        engine=engine, 
                        image_size=len(image_data),
                        available_engines=list(self.engines.keys()))
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            self.logger.info("📸 Image loaded", 
                           format=image.format, 
                           mode=image.mode, 
                           size=f"{image.width}x{image.height}")
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Optimize image size for faster processing
            height, width = cv_image.shape[:2]
            if width > 1200 or height > 1200:
                # Scale down large images
                scale = min(1200/width, 1200/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.logger.info(f"📏 Image resized for speed: {width}x{height} → {new_width}x{new_height}")
            
            self.logger.info("🔄 Converted to OpenCV format", 
                           shape=cv_image.shape)
            
            if engine == 'auto':
                self.logger.info("🎯 Using auto mode - FAST MODE enabled")
                # Use fast mode by default for speed
                results = await self._try_fast_engines(cv_image)
                best_result = self._select_best_result(results)
                self.logger.info("🏆 Best result selected", 
                               engine=best_result.get('engine'),
                               confidence=best_result.get('confidence'),
                               text_length=len(best_result.get('text', '')))
                return best_result
            else:
                self.logger.info(f"🔧 Using specific engine: {engine}")
                # Use specific engine
                result = await self._process_with_engine(cv_image, engine)
                self.logger.info("✅ Engine processing complete", 
                               success=result.get('success'),
                               confidence=result.get('confidence'),
                               text_length=len(result.get('text', '')))
                return result
                
        except Exception as e:
            self.logger.error("❌ OCR processing failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0
            }
    
    async def _try_fast_engines(self, cv_image) -> List[Dict[str, Any]]:
        """Try OCR engines in FAST mode - EasyOCR first for accuracy"""
        self.logger.info("⚡ Fast OCR mode - EasyOCR first for accuracy")
        results = []
        
        # Try EasyOCR first (best for business cards)
        if 'easyocr' in self.engines and self.engines['easyocr']:
            self.logger.info("🔧 Using EasyOCR (best for business cards)")
            try:
                result = await self._process_with_engine(cv_image, 'easyocr')
                results.append(result)
                
                # If EasyOCR gives good results, use it
                if result.get('success') and result.get('confidence', 0) > 0.3:
                    self.logger.info(f"✅ EasyOCR success - stopping here", 
                                   confidence=result.get('confidence', 0),
                                   text_length=len(result.get('text', '')))
                    return results
                
                self.logger.info(f"✅ EasyOCR completed", 
                               success=result.get('success'),
                               confidence=result.get('confidence', 0),
                               text_length=len(result.get('text', '')))
            except Exception as e:
                self.logger.warning(f"❌ EasyOCR failed", error=str(e))
        
        # Fallback to PaddleOCR if EasyOCR fails or gives poor results
        if 'paddleocr' in self.engines and self.engines['paddleocr']:
            self.logger.info("🔄 Trying PaddleOCR fallback")
            try:
                result = await self._process_with_engine(cv_image, 'paddleocr')
                results.append(result)
                self.logger.info(f"✅ PaddleOCR completed", 
                               success=result.get('success'),
                               confidence=result.get('confidence', 0),
                               text_length=len(result.get('text', '')))
            except Exception as e:
                self.logger.warning(f"❌ PaddleOCR failed", error=str(e))
        
        # Final fallback to Tesseract if both fail
        if 'tesseract' in self.engines and self.engines['tesseract']:
            self.logger.info("🔄 Final fallback to Tesseract")
            try:
                result = await self._process_with_engine(cv_image, 'tesseract')
                results.append(result)
            except Exception as e:
                self.logger.warning(f"❌ Tesseract fallback failed", error=str(e))
        
        self.logger.info(f"📊 Fast processing complete: {len(results)} results")
        return results

    async def _try_all_engines(self, cv_image) -> List[Dict[str, Any]]:
        """Try all available OCR engines for maximum accuracy"""
        self.logger.info("🔄 Trying all available engines (full mode)")
        results = []
        
        # Try engines in order of preference: EasyOCR -> PaddleOCR -> Tesseract
        preferred_order = ['easyocr', 'paddleocr', 'tesseract']
        
        for engine_name in preferred_order:
            if engine_name in self.engines and self.engines[engine_name]:
                self.logger.info(f"🔧 Trying engine: {engine_name}")
                try:
                    result = await self._process_with_engine(cv_image, engine_name)
                    results.append(result)
                    self.logger.info(f"✅ Engine {engine_name} completed", 
                                   success=result.get('success'),
                                   confidence=result.get('confidence', 0),
                                   text_length=len(result.get('text', '')))
                except Exception as e:
                    self.logger.warning(f"❌ Engine {engine_name} failed", error=str(e))
            else:
                self.logger.info(f"⏭️ Skipping engine {engine_name} (not available)")
        
        self.logger.info(f"📊 All engine testing complete: {len(results)} results")
        return results
    
    async def _process_with_engine(self, cv_image, engine: str) -> Dict[str, Any]:
        """Process image with specific engine"""
        if engine == 'tesseract' and self.engines.get('tesseract'):
            return await self._tesseract_ocr(cv_image)
        elif engine == 'easyocr' and self.engines.get('easyocr'):
            return await self._easyocr_process(cv_image)
        elif engine == 'paddleocr' and self.engines.get('paddleocr'):
            return await self._paddleocr_process(cv_image)
        else:
            raise ValueError(f"Engine {engine} not available")
    
    async def _tesseract_ocr(self, cv_image) -> Dict[str, Any]:
        """Process with Tesseract"""
        self.logger.info("🔧 Processing with Tesseract OCR")
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            self.logger.info(f"📸 Tesseract input image: {pil_image.size}")
            
            # Get text with confidence - OPTIMIZED for speed
            self.logger.info("🔍 Running Tesseract OCR (fast mode)...")
            # Use faster PSM mode and simpler config
            config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-+() '
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=config)
            text = pytesseract.image_to_string(pil_image, config=config)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            self.logger.info("✅ Tesseract OCR complete", 
                           text_length=len(text.strip()),
                           confidence=avg_confidence,
                           word_count=len(data['text']))
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": avg_confidence / 100.0,
                "engine": "tesseract",
                "raw_data": data
            }
        except Exception as e:
            self.logger.error("❌ Tesseract OCR failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "engine": "tesseract"
            }
    
    async def _easyocr_process(self, cv_image) -> Dict[str, Any]:
        """Process with EasyOCR - OPTIMIZED for speed"""
        self.logger.info("🔧 Processing with EasyOCR (optimized)")
        try:
            self.logger.info("🔍 Running EasyOCR with speed optimizations...")
            loop = asyncio.get_event_loop()
            # Use optimized parameters for speed
            results = await loop.run_in_executor(
                self.executor, 
                self._easyocr_readtext_optimized, 
                cv_image
            )
            
            self.logger.info(f"📊 EasyOCR found {len(results)} text regions")
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            self.logger.info("✅ EasyOCR complete", 
                           text_length=len(full_text),
                           confidence=avg_confidence,
                           regions=len(results))
            
            return {
                "success": True,
                "text": full_text,
                "confidence": avg_confidence,
                "engine": "easyocr",
                "raw_data": results
            }
        except Exception as e:
            self.logger.error("❌ EasyOCR failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "engine": "easyocr"
            }
    
    async def _paddleocr_process(self, cv_image) -> Dict[str, Any]:
        """Process with PaddleOCR"""
        self.logger.info("🔧 Processing with PaddleOCR")
        try:
            self.logger.info("🔍 Running PaddleOCR...")
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self.paddle_ocr.ocr,
                cv_image
            )
            
            self.logger.info(f"📊 PaddleOCR found {len(results) if results else 0} result groups")
            
            # Extract text and confidence with better error handling
            text_parts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) > 1 and len(line[1]) > 1:
                        try:
                            text_parts.append(line[1][0])
                            confidences.append(line[1][1])
                        except (IndexError, TypeError) as e:
                            self.logger.warning("⚠️ Skipping malformed PaddleOCR result", error=str(e))
                            continue
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            self.logger.info("✅ PaddleOCR complete", 
                           text_length=len(full_text),
                           confidence=avg_confidence,
                           lines=len(text_parts))
            
            return {
                "success": True,
                "text": full_text,
                "confidence": avg_confidence,
                "engine": "paddleocr",
                "raw_data": results
            }
        except Exception as e:
            self.logger.error("❌ PaddleOCR failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "engine": "paddleocr"
            }
    
    async def scan_qr_codes(self, image_data: bytes) -> Dict[str, Any]:
        """Scan for QR codes in the image using enhanced detection"""
        self.logger.info("🔍 Starting enhanced QR code scanning")
        
        # Try enhanced QR detection first
        if self.enhanced_qr_available and self.enhanced_qr_service:
            try:
                self.logger.info("🚀 Using enhanced QR detection")
                result = await self.enhanced_qr_service.detect_qr_codes_enhanced(image_data)
                if result['success'] and result['count'] > 0:
                    self.logger.info(f"✅ Enhanced QR detection found {result['count']} QR codes")
                    return result
                else:
                    self.logger.info("⚠️ Enhanced QR detection found no codes, falling back to standard methods")
            except Exception as e:
                self.logger.warning("⚠️ Enhanced QR detection failed, falling back", error=str(e))
        
        # Fallback to standard methods
        if not PYZBAR_AVAILABLE and not OPENCV_QR_AVAILABLE:
            return {
                "success": False,
                "error": "QR code scanning not available (no QR libraries installed)",
                "qr_codes": [],
                "parsed_data": {}
            }
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            self.logger.info("📸 Image loaded for QR scanning", 
                           format=image.format, 
                           size=f"{image.width}x{image.height}")
            
            # Convert to OpenCV format for better QR detection
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhance image for better QR detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            all_qr_codes = []
            
            if PYZBAR_AVAILABLE:
                # Use pyzbar for QR detection
                self.logger.info("🔍 Using pyzbar for QR detection")
                
                # Apply some preprocessing to improve QR detection
                # Increase contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Try multiple preprocessing approaches
                processed_images = [
                    gray,  # Original grayscale
                    enhanced,  # Enhanced contrast
                    cv2.GaussianBlur(gray, (3, 3), 0),  # Slight blur
                    cv2.medianBlur(gray, 3)  # Median blur
                ]
                
                for i, processed_img in enumerate(processed_images):
                    try:
                        # Convert back to PIL for pyzbar
                        pil_image = Image.fromarray(processed_img)
                        
                        # Scan for QR codes
                        qr_codes = pyzbar.decode(pil_image)
                        
                        if qr_codes:
                            self.logger.info(f"📱 Found {len(qr_codes)} QR codes with method {i}")
                            all_qr_codes.extend(qr_codes)
                            
                    except Exception as e:
                        self.logger.warning(f"QR scanning method {i} failed", error=str(e))
                        continue
            else:
                # Use OpenCV QR detector as fallback
                self.logger.info("🔍 Using OpenCV QR detector")
                
                try:
                    # Try with different preprocessing
                    processed_images = [
                        gray,  # Original grayscale
                        cv2.GaussianBlur(gray, (3, 3), 0),  # Slight blur
                        cv2.medianBlur(gray, 3)  # Median blur
                    ]
                    
                    for i, processed_img in enumerate(processed_images):
                        try:
                            # Detect QR codes using OpenCV
                            retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(processed_img)
                            
                            if retval and decoded_info:
                                self.logger.info(f"📱 Found {len(decoded_info)} QR codes with OpenCV method {i}")
                                
                                # Convert OpenCV results to pyzbar-like format
                                for j, (data, points_array) in enumerate(zip(decoded_info, points)):
                                    if data:  # Only add non-empty results
                                        # Create a simple object that mimics pyzbar result
                                        class QRResult:
                                            def __init__(self, data, type_name):
                                                self.data = data.encode('utf-8')
                                                self.type = type_name
                                        
                                        qr_result = QRResult(data, 'QRCODE')
                                        all_qr_codes.append(qr_result)
                                        
                        except Exception as e:
                            self.logger.warning(f"OpenCV QR scanning method {i} failed", error=str(e))
                            continue
                            
                except Exception as e:
                    self.logger.warning("OpenCV QR detection failed", error=str(e))
            
            # Remove duplicates based on data
            unique_qr_codes = []
            seen_data = set()
            
            for qr in all_qr_codes:
                data = qr.data.decode('utf-8')
                if data not in seen_data:
                    unique_qr_codes.append(qr)
                    seen_data.add(data)
            
            self.logger.info(f"📊 Found {len(unique_qr_codes)} unique QR codes")
            
            # Parse QR code data
            parsed_data = {}
            qr_results = []
            
            for qr in unique_qr_codes:
                try:
                    data = qr.data.decode('utf-8')
                    qr_type = qr.type
                    
                    self.logger.info(f"📱 QR Code found", type=qr_type, data_length=len(data))
                    
                    # Parse based on QR code type
                    parsed_qr = self._parse_qr_data(data, qr_type)
                    
                    qr_results.append({
                        "data": data,
                        "type": qr_type,
                        "parsed": parsed_qr,
                        "raw": qr
                    })
                    
                    # Merge parsed data
                    if parsed_qr:
                        parsed_data.update(parsed_qr)
                        
                except Exception as e:
                    self.logger.warning("Failed to parse QR code", error=str(e))
                    qr_results.append({
                        "data": qr.data.decode('utf-8', errors='ignore'),
                        "type": qr.type,
                        "parsed": {},
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "qr_codes": qr_results,
                "parsed_data": parsed_data,
                "count": len(unique_qr_codes)
            }
            
        except Exception as e:
            self.logger.error("❌ QR code scanning failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "qr_codes": [],
                "parsed_data": {}
            }
    
    def _select_best_result(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best OCR result with preference for EasyOCR/PaddleOCR over Tesseract"""
        self.logger.info(f"🏆 Selecting best result from {len(results)} results")
        
        if not results:
            self.logger.warning("❌ No results to select from")
            return {
                "success": False,
                "error": "No OCR engines available",
                "text": "",
                "confidence": 0.0
            }
        
        # Filter successful results
        successful_results = [r for r in results if r.get("success", False)]
        self.logger.info(f"📊 {len(successful_results)} successful results out of {len(results)} total")
        
        if not successful_results:
            self.logger.warning("❌ No successful results, returning first result")
            return results[0]  # Return first result (likely has error info)
        
        # Engine preference weights (higher = better)
        engine_weights = {
            'easyocr': 3.0,
            'paddleocr': 2.5,
            'tesseract': 1.0
        }
        
        # Select result with highest weighted score
        def calculate_score(result):
            engine = result.get("engine", "unknown")
            confidence = result.get("confidence", 0)
            text_length = len(result.get("text", "").strip())
            weight = engine_weights.get(engine, 1.0)
            
            # Score = (confidence * text_length * engine_weight)
            score = confidence * text_length * weight
            self.logger.info(f"📊 {engine}: confidence={confidence:.3f}, text_length={text_length}, weight={weight}, score={score:.2f}")
            return score
        
        best_result = max(successful_results, key=calculate_score)
        
        self.logger.info(f"🏆 Selected best result from {best_result.get('engine', 'unknown')} engine", 
                        confidence=best_result.get('confidence', 0),
                        text_length=len(best_result.get('text', '')))
        return best_result
    
    async def extract_business_card_data(self, image_data: bytes, use_vision: bool = True) -> Dict[str, Any]:
        """Extract structured data from business card using OCR and optionally vision analysis"""
        try:
            # Process image with OCR
            ocr_result = await self.process_image(image_data, engine='auto')
            
            # Scan for QR codes
            qr_result = await self.scan_qr_codes(image_data)
            
            # Parse business card data from OCR
            parsed_data = self._parse_business_card(ocr_result["text"]) if ocr_result["success"] else {}
            
            # Merge QR code data if found
            if qr_result["success"] and qr_result.get("parsed_data"):
                # Merge QR data with OCR data (QR data takes precedence)
                for key, value in qr_result["parsed_data"].items():
                    if value:  # Only use non-empty values
                        parsed_data[key] = value
                
                self.logger.info("📱 QR code data merged with OCR data", 
                               qr_fields=list(qr_result["parsed_data"].keys()))
                
                # If QR code contains URLs, fetch additional details
                urls_to_fetch = []
                for qr in qr_result.get("qr_codes", []):
                    if qr.get("parsed", {}).get("type") == "url":
                        urls_to_fetch.append(qr["data"])
                
                if urls_to_fetch:
                    try:
                        from .qr_fetch_service import fetch_multiple_qr_urls
                        url_details = await fetch_multiple_qr_urls(urls_to_fetch)
                        
                        # Merge fetched URL details
                        for url, details in url_details.items():
                            if details.get("success"):
                                # Add website title and description
                                if details.get("title"):
                                    parsed_data["website_title"] = details["title"]
                                if details.get("description"):
                                    parsed_data["website_description"] = details["description"]
                                
                                # Add contact info from website
                                contact_info = details.get("contact_info", {})
                                if contact_info.get("emails") and not parsed_data.get("email"):
                                    parsed_data["email"] = contact_info["emails"][0]
                                if contact_info.get("phones") and not parsed_data.get("phone"):
                                    parsed_data["phone"] = contact_info["phones"][0]
                                
                                # Add social links
                                social_links = details.get("social_links", {})
                                if social_links:
                                    parsed_data["social_links"] = social_links
                                
                                self.logger.info("🌐 Enhanced business card with URL details", 
                                               url=url, 
                                               has_contact=bool(contact_info),
                                               has_social=bool(social_links))
                    
                    except Exception as e:
                        self.logger.warning("Failed to fetch URL details", error=str(e))
            
            # Use vision analysis if available and requested
            vision_result = None
            self.logger.info(f"🔍 Vision check: use_vision={use_vision}, vision_available={self.vision_available}")
            if use_vision and self.vision_available:
                try:
                    self.logger.info("🔍 Running vision analysis for business card")
                    vision_result = await self.vision_service.analyze_business_card(image_data)
                    
                    self.logger.info("🔍 Vision result received", 
                                   success=vision_result.get("success"),
                                   has_structured_info=bool(vision_result.get("structured_info")),
                                   confidence=vision_result.get("confidence", 0))
                    
                    if vision_result.get("success"):
                        # Merge vision data with OCR data (vision data takes precedence)
                        vision_contact = vision_result.get("structured_info", {}).get("contact_info", {})
                        self.logger.info("🔍 Vision contact data", contact_fields=list(vision_contact.keys()))
                        
                        for key, value in vision_contact.items():
                            if value:  # Only use non-empty values
                                parsed_data[key] = value
                        
                        self.logger.info("✅ Vision analysis completed", 
                                       confidence=vision_result.get("confidence", 0),
                                       fields_extracted=len([k for k, v in vision_contact.items() if v]))
                    else:
                        self.logger.warning("⚠️ Vision analysis failed", error=vision_result.get("error"))
                        
                except Exception as e:
                    self.logger.error("❌ Vision analysis error", error=str(e))
                    self.logger.error("❌ Vision analysis error details", error_type=type(e).__name__, error_message=str(e))
            
            return {
                "success": True,
                "data": parsed_data,
                "confidence": ocr_result.get("confidence", 0.0),
                "raw_text": ocr_result.get("text", ""),
                "engine_used": ocr_result.get("engine", "unknown"),
                "qr_codes": qr_result.get("qr_codes", []),
                "qr_count": qr_result.get("count", 0),
                "qr_parsed_data": qr_result.get("parsed_data", {}),
                "vision_analysis": vision_result,
                "vision_available": self.vision_available and vision_result is not None
            }
            
        except Exception as e:
            self.logger.error("Business card extraction failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    async def analyze_business_card_vision(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze business card using vision service only"""
        if not self.vision_available:
            return {
                "success": False,
                "error": "Vision service not available",
                "data": {},
                "confidence": 0.0
            }
        
        try:
            self.logger.info("🔍 Running vision-only business card analysis")
            vision_result = await self.vision_service.analyze_business_card(image_data)
            
            if vision_result.get("success"):
                structured_info = vision_result.get("structured_info", {})
                contact_info = structured_info.get("contact_info", {})
                
                self.logger.info("✅ Vision analysis completed", 
                               confidence=vision_result.get("confidence", 0),
                               fields_extracted=len([k for k, v in contact_info.items() if v]))
                
                return {
                    "success": True,
                    "data": contact_info,
                    "confidence": vision_result.get("confidence", 0.0),
                    "analysis_notes": structured_info.get("analysis_notes", ""),
                    "quality_assessment": structured_info.get("quality_assessment", {}),
                    "raw_analysis": vision_result.get("raw_analysis", ""),
                    "method": "vision_only"
                }
            else:
                return {
                    "success": False,
                    "error": vision_result.get("error", "Vision analysis failed"),
                    "data": {},
                    "confidence": 0.0
                }
                
        except Exception as e:
            self.logger.error("❌ Vision analysis failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "data": {},
                "confidence": 0.0
            }
    
    def _parse_business_card(self, text: str) -> Dict[str, str]:
        """Parse business card text into structured data"""
        data = {
            "name": "",
            "email": "",
            "phone": "",
            "company": "",
            "title": "",
            "address": "",
            "website": ""
        }
        
        if not text:
            return data
        
        # Clean and normalize text
        text = text.strip()
        
        # Email regex - more comprehensive
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            data["email"] = emails[0]
        
        # Website detection
        website_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        ]
        
        for pattern in website_patterns:
            websites = re.findall(pattern, text)
            if websites:
                # Clean up the website URL
                website = websites[0]
                if not website.startswith(('http://', 'https://')):
                    website = 'http://' + website
                data["website"] = website
                break
        
        # Phone regex - handle international formats
        phone_patterns = [
            r'(\+?91[-.\s]?)?\(?([0-9]{2,3})\)?[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{4})',  # Indian format
            r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',     # US format
            r'Tel[:\s]*(\+?[\d\s\-\(\)]{10,})',                                      # Tel: prefix
            r'Phone[:\s]*(\+?[\d\s\-\(\)]{10,})',                                     # Phone: prefix
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text, re.IGNORECASE)
            if phones:
                # Clean up the phone number
                phone = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
                phone = re.sub(r'[^\d+]', '', phone)  # Keep only digits and +
                if len(phone) >= 10:  # Valid phone number length
                    data["phone"] = phone
                    break
        
        # Split text into lines for analysis
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Name detection - look for person names (2+ words, capitalized)
        # Try to find the first proper name in the text
        words = text.split()
        for i in range(len(words) - 1):
            # Look for two consecutive capitalized words
            if (words[i][0].isupper() and words[i+1][0].isupper() and 
                len(words[i]) > 1 and len(words[i+1]) > 1 and
                not any(char in words[i] for char in ['@', '+', '(', ')', '-', '.', '/']) and
                not any(char in words[i+1] for char in ['@', '+', '(', ')', '-', '.', '/'])):
                potential_name = f"{words[i]} {words[i+1]}"
                # Check if it's not a company name, job title, or common business words
                business_words = ['Inc', 'LLC', 'Corp', 'Ltd', 'Company', 'Co', 'Group', 'Services', 'Systems', 'BIS', 'EA', 'What', 'Business', 'Demands', 'Computer', 'Ltac', 'Secunderabad', 'Sebastian', 'Road', 'Ohni', 'Tovers', 'New', 'Exinz', 'Fax', 'Telz', 'Gasatyam', 'Web', 'sitez', 'Email']
                if (not any(suffix in potential_name for suffix in business_words) and
                    not any(title in potential_name for title in ['Analyst', 'Manager', 'Director', 'CEO', 'CTO', 'Engineer', 'Developer', 'Consultant', 'Specialist', 'Executive', 'President', 'VP'])):
                    data["name"] = potential_name
                    break
        
        # Title detection - look for job titles
        title_keywords = ['Analyst', 'Manager', 'Director', 'CEO', 'CTO', 'Engineer', 'Developer', 
                         'Consultant', 'Specialist', 'Executive', 'President', 'VP', 'Vice President']
        for keyword in title_keywords:
            if keyword in text:
                # Find the context around the keyword
                words = text.split()
                for i, word in enumerate(words):
                    if keyword in word:
                        # Get a few words around the keyword
                        start = max(0, i-1)
                        end = min(len(words), i+2)
                        title_context = ' '.join(words[start:end])
                        data["title"] = title_context
                        break
                if data["title"]:
                    break
        
        # Company detection - look for company names with business suffixes
        company_suffixes = ['Inc', 'LLC', 'Corp', 'Ltd', 'Company', 'Co', 'Group', 'Services', 'Systems']
        for suffix in company_suffixes:
            if suffix in text:
                # Find the context around the suffix
                words = text.split()
                for i, word in enumerate(words):
                    if suffix in word:
                        # Get a few words before the suffix
                        start = max(0, i-2)
                        end = min(len(words), i+1)
                        company_context = ' '.join(words[start:end])
                        data["company"] = company_context
                        break
                if data["company"]:
                    break
        
        # Address detection - look for street patterns
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
            r'[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
            r'\d+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'
        ]
        
        for pattern in address_patterns:
            addresses = re.findall(pattern, text, re.IGNORECASE)
            if addresses:
                data["address"] = addresses[0].strip()
                break
        
        # If no structured data found, try to extract from the raw text
        if not any(data.values()):
            # Fallback: try to extract any meaningful information
            words = text.split()
            if len(words) >= 2:
                # First two words might be name
                potential_name = ' '.join(words[:2])
                if all(word[0].isupper() for word in words[:2]):
                    data["name"] = potential_name
                
                # Look for company in the text
                for i, word in enumerate(words):
                    if word.lower() in ['ltd', 'inc', 'corp', 'llc', 'company']:
                        # Take a few words before this as company name
                        start = max(0, i-3)
                        end = min(len(words), i+1)
                        data["company"] = ' '.join(words[start:end])
                        break
        
        return data
    
    def _parse_qr_data(self, data: str, qr_type: str) -> Dict[str, Any]:
        """Parse QR code data based on its type"""
        parsed = {}
        
        try:
            # Handle different QR code types
            if qr_type == 'QRCODE':
                # Check for common business card formats
                
                # vCard format (BEGIN:VCARD)
                if data.startswith('BEGIN:VCARD'):
                    parsed = self._parse_vcard(data)
                
                # URL format
                elif data.startswith(('http://', 'https://', 'www.')):
                    parsed = self._parse_url(data)
                
                # Email format
                elif '@' in data and '.' in data.split('@')[-1]:
                    parsed = self._parse_email(data)
                
                # Phone format
                elif re.match(r'^[\+]?[0-9\s\-\(\)]+$', data.strip()):
                    parsed = self._parse_phone(data)
                
                # JSON format
                elif data.strip().startswith('{') and data.strip().endswith('}'):
                    parsed = self._parse_json(data)
                
                # Plain text (try to extract contact info)
                else:
                    parsed = self._parse_plain_text(data)
            
            
            self.logger.info(f"📱 Parsed QR data", type=qr_type, fields=list(parsed.keys()))
            return parsed
            
        except Exception as e:
            self.logger.warning("Failed to parse QR data", error=str(e))
            return {"raw_data": data, "type": qr_type, "parse_error": str(e)}
    
    def _parse_vcard(self, data: str) -> Dict[str, Any]:
        """Parse vCard format"""
        parsed = {}
        lines = data.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.upper()
                
                if key == 'FN':
                    parsed['name'] = value
                elif key == 'ORG':
                    parsed['company'] = value
                elif key == 'TITLE':
                    parsed['title'] = value
                elif key == 'EMAIL':
                    parsed['email'] = value
                elif key == 'TEL':
                    parsed['phone'] = value
                elif key == 'ADR':
                    parsed['address'] = value
                elif key == 'URL':
                    parsed['website'] = value
        
        return parsed
    
    def _parse_url(self, data: str) -> Dict[str, Any]:
        """Parse URL data"""
        try:
            parsed_url = urlparse(data)
            return {
                "url": data,
                "domain": parsed_url.netloc,
                "website": data
            }
        except Exception:
            return {"url": data}
    
    def _parse_email(self, data: str) -> Dict[str, Any]:
        """Parse email data"""
        return {"email": data}
    
    def _parse_phone(self, data: str) -> Dict[str, Any]:
        """Parse phone data"""
        # Clean phone number
        clean_phone = re.sub(r'[^\d\+]', '', data)
        return {"phone": clean_phone}
    
    def _parse_json(self, data: str) -> Dict[str, Any]:
        """Parse JSON data"""
        try:
            json_data = json.loads(data)
            return json_data
        except json.JSONDecodeError:
            return {"raw_data": data}
    
    def _parse_plain_text(self, data: str) -> Dict[str, Any]:
        """Parse plain text for contact information"""
        parsed = {}
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', data)
        if email_match:
            parsed['email'] = email_match.group()
        
        # Extract phone
        phone_match = re.search(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', data)
        if phone_match:
            parsed['phone'] = ''.join(phone_match.groups())
        
        # Extract URL
        url_match = re.search(r'https?://[^\s]+', data)
        if url_match:
            parsed['website'] = url_match.group()
        
        # If no structured data found, store as raw text
        if not parsed:
            parsed['raw_text'] = data
        
        return parsed
    
    def _easyocr_readtext_optimized(self, cv_image):
        """Optimized EasyOCR processing for speed"""
        try:
            # Use optimized parameters for faster processing
            results = self.easyocr_reader.readtext(
                cv_image,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-+() ',  # Limit character set
                width_ths=0.7,  # Reduce width threshold for faster processing
                height_ths=0.7,  # Reduce height threshold for faster processing
                paragraph=False,  # Disable paragraph detection for speed
                batch_size=1  # Process one image at a time
            )
            return results
        except Exception as e:
            self.logger.warning(f"Optimized EasyOCR failed, falling back to standard: {e}")
            # Fallback to standard EasyOCR
            return self.easyocr_reader.readtext(cv_image)

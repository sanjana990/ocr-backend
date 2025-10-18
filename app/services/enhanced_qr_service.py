#!/usr/bin/env python3
"""
Enhanced QR Code Detection Service
Uses multiple detection methods for maximum reliability
"""

import cv2
import numpy as np
from PIL import Image
import io
import structlog
from typing import Dict, Any, List, Optional, Tuple
import re
import json

logger = structlog.get_logger(__name__)

class EnhancedQRService:
    """Enhanced QR code detection using multiple methods"""
    
    def __init__(self):
        self.logger = logger
        self.opencv_detector = None
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize available QR detectors"""
        try:
            self.opencv_detector = cv2.QRCodeDetector()
            self.logger.info("✅ OpenCV QR detector initialized")
        except Exception as e:
            self.logger.warning("⚠️ OpenCV QR detector not available", error=str(e))
    
    async def detect_qr_codes_enhanced(self, image_data: bytes) -> Dict[str, Any]:
        """Enhanced QR detection using multiple methods"""
        self.logger.info("🔍 Starting enhanced QR code detection")
        
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            self.logger.info("📸 Image loaded for enhanced QR detection", 
                           format=image.format, 
                           size=f"{image.width}x{image.height}")
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            all_qr_codes = []
            detection_methods = []
            
            # Method 1: OpenCV with multiple preprocessing
            opencv_results = self._detect_with_opencv_enhanced(cv_image)
            if opencv_results['qr_codes']:
                all_qr_codes.extend(opencv_results['qr_codes'])
                detection_methods.append("OpenCV Enhanced")
            
            # Method 2: Image preprocessing for better detection
            preprocessed_results = self._detect_with_preprocessing(cv_image)
            if preprocessed_results['qr_codes']:
                all_qr_codes.extend(preprocessed_results['qr_codes'])
                detection_methods.append("Preprocessing")
            
            # Method 3: Multi-scale detection
            multiscale_results = self._detect_multiscale(cv_image)
            if multiscale_results['qr_codes']:
                all_qr_codes.extend(multiscale_results['qr_codes'])
                detection_methods.append("Multi-scale")
            
            # Remove duplicates
            unique_qr_codes = self._remove_duplicates(all_qr_codes)
            
            self.logger.info(f"📊 Found {len(unique_qr_codes)} unique QR codes using {detection_methods}")
            
            # Parse QR codes
            parsed_data = {}
            qr_results = []
            
            for qr in unique_qr_codes:
                try:
                    data = qr['data']
                    qr_type = qr.get('type', 'QRCODE')
                    
                    self.logger.info(f"📱 QR Code found", type=qr_type, data_length=len(data))
                    
                    parsed_qr = self._parse_qr_data(data, qr_type)
                    
                    qr_results.append({
                        "data": data,
                        "type": qr_type,
                        "parsed": parsed_qr,
                        "method": qr.get('method', 'unknown')
                    })
                    
                    if parsed_qr:
                        parsed_data.update(parsed_qr)
                        
                except Exception as e:
                    self.logger.warning("Failed to parse QR code", error=str(e))
                    qr_results.append({
                        "data": data,
                        "type": qr_type,
                        "parsed": {},
                        "error": str(e),
                        "method": qr.get('method', 'unknown')
                    })
            
            return {
                "success": True,
                "qr_codes": qr_results,
                "parsed_data": parsed_data,
                "count": len(unique_qr_codes),
                "methods_used": detection_methods
            }
            
        except Exception as e:
            self.logger.error("Enhanced QR detection failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "qr_codes": [],
                "parsed_data": {}
            }
    
    def _detect_with_opencv_enhanced(self, cv_image) -> Dict[str, Any]:
        """Enhanced OpenCV detection with multiple approaches"""
        qr_codes = []
        
        if not self.opencv_detector:
            return {"qr_codes": qr_codes}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Try different preprocessing approaches
            preprocessing_methods = [
                ("original", gray),
                ("gaussian_blur", cv2.GaussianBlur(gray, (3, 3), 0)),
                ("median_blur", cv2.medianBlur(gray, 3)),
                ("bilateral", cv2.bilateralFilter(gray, 9, 75, 75)),
                ("histogram_eq", cv2.equalizeHist(gray)),
                ("adaptive_thresh", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
            ]
            
            for method_name, processed_img in preprocessing_methods:
                try:
                    # Try single QR detection
                    result = self.opencv_detector.detectAndDecode(processed_img)
                    if len(result) == 4:
                        retval, decoded_info, points, straight_qrcode = result
                    elif len(result) == 3:
                        retval, decoded_info, points = result
                        straight_qrcode = None
                    else:
                        retval, decoded_info = result
                        points = None
                        straight_qrcode = None
                    
                    if retval and decoded_info:
                        qr_codes.append({
                            'data': decoded_info,
                            'type': 'QRCODE',
                            'method': f"OpenCV_{method_name}",
                            'points': points
                        })
                        self.logger.info(f"✅ Found QR with {method_name}: {decoded_info}")
                    
                    # Try multi QR detection only if single detection didn't find anything
                    if not (retval and decoded_info):
                        result = self.opencv_detector.detectAndDecodeMulti(processed_img)
                        if len(result) == 4:
                            retval, decoded_info, points, straight_qrcode = result
                        elif len(result) == 3:
                            retval, decoded_info, points = result
                            straight_qrcode = None
                        else:
                            retval, decoded_info = result
                            points = None
                            straight_qrcode = None
                        
                        if retval and decoded_info:
                            for i, data in enumerate(decoded_info):
                                if data:  # Only add non-empty data
                                    qr_codes.append({
                                        'data': data,
                                        'type': 'QRCODE',
                                        'method': f"OpenCV_Multi_{method_name}",
                                        'points': points[i] if points and i < len(points) else None
                                    })
                                    self.logger.info(f"✅ Found Multi QR with {method_name}: {data}")
                
                except Exception as e:
                    self.logger.warning(f"OpenCV detection failed for {method_name}", error=str(e))
        
        except Exception as e:
            self.logger.warning("OpenCV enhanced detection failed", error=str(e))
        
        return {"qr_codes": qr_codes}
    
    def _detect_with_preprocessing(self, cv_image) -> Dict[str, Any]:
        """QR detection with advanced preprocessing"""
        qr_codes = []
        
        if not self.opencv_detector:
            return {"qr_codes": qr_codes}
        
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Advanced preprocessing techniques
            preprocessing_techniques = [
                # Morphological operations
                ("morph_open", cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))),
                ("morph_close", cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))),
                
                # Edge enhancement
                ("laplacian", cv2.Laplacian(gray, cv2.CV_64F)),
                ("sobel_x", cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)),
                ("sobel_y", cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)),
                
                # Noise reduction
                ("gaussian_strong", cv2.GaussianBlur(gray, (5, 5), 0)),
                ("median_strong", cv2.medianBlur(gray, 5)),
            ]
            
            for technique_name, processed_img in preprocessing_techniques:
                try:
                    # Convert to uint8 if needed
                    if processed_img.dtype != np.uint8:
                        processed_img = np.uint8(np.absolute(processed_img))
                    
                    result = self.opencv_detector.detectAndDecode(processed_img)
                    if len(result) >= 2:
                        retval, decoded_info = result[:2]
                        if retval and decoded_info:
                            qr_codes.append({
                                'data': decoded_info,
                                'type': 'QRCODE',
                                'method': f"Preprocessing_{technique_name}"
                            })
                            self.logger.info(f"✅ Found QR with {technique_name}: {decoded_info}")
                
                except Exception as e:
                    self.logger.warning(f"Preprocessing detection failed for {technique_name}", error=str(e))
        
        except Exception as e:
            self.logger.warning("Preprocessing detection failed", error=str(e))
        
        return {"qr_codes": qr_codes}
    
    def _detect_multiscale(self, cv_image) -> Dict[str, Any]:
        """Multi-scale QR detection"""
        qr_codes = []
        
        if not self.opencv_detector:
            return {"qr_codes": qr_codes}
        
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Try different scales
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            
            for scale in scales:
                try:
                    # Resize image
                    scaled_img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    
                    result = self.opencv_detector.detectAndDecode(scaled_img)
                    if len(result) >= 2:
                        retval, decoded_info = result[:2]
                        if retval and decoded_info:
                            qr_codes.append({
                                'data': decoded_info,
                                'type': 'QRCODE',
                                'method': f"MultiScale_{scale}x"
                            })
                            self.logger.info(f"✅ Found QR at scale {scale}: {decoded_info}")
                
                except Exception as e:
                    self.logger.warning(f"Multi-scale detection failed for scale {scale}", error=str(e))
        
        except Exception as e:
            self.logger.warning("Multi-scale detection failed", error=str(e))
        
        return {"qr_codes": qr_codes}
    
    def _remove_duplicates(self, qr_codes: List[Dict]) -> List[Dict]:
        """Remove duplicate QR codes based on data content"""
        seen_data = set()
        unique_codes = []
        
        for qr in qr_codes:
            if qr['data'] not in seen_data:
                seen_data.add(qr['data'])
                unique_codes.append(qr)
        
        return unique_codes
    
    def _parse_qr_data(self, data: str, qr_type: str) -> Dict[str, Any]:
        """Parse QR code data based on its type"""
        parsed = {}
        try:
            if qr_type == 'QRCODE':
                if data.startswith('BEGIN:VCARD'):
                    parsed = self._parse_vcard(data)
                elif data.startswith(('http://', 'https://', 'www.')):
                    parsed = self._parse_url(data)
                elif '@' in data and '.' in data.split('@')[-1]:
                    parsed = self._parse_email(data)
                elif re.match(r'^[\+]?[0-9\s\-\(\)]+$', data.strip()):
                    parsed = self._parse_phone(data)
                elif data.strip().startswith('{') and data.strip().endswith('}'):
                    parsed = self._parse_json(data)
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
        """Parse URL data with enhanced business card detection"""
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(data)
            domain = parsed_url.netloc.lower()
            
            # Detect business-related domains
            business_indicators = [
                'linkedin.com', 'company', 'corp', 'inc', 'llc', 'ltd',
                'business', 'enterprise', 'professional', 'contact'
            ]
            
            is_business_site = any(indicator in domain for indicator in business_indicators)
            
            return {
                "url": data, 
                "type": "url",
                "domain": domain,
                "is_business_site": is_business_site,
                "website": data
            }
        except Exception as e:
            return {"url": data, "type": "url", "parse_error": str(e)}
    
    def _parse_email(self, data: str) -> Dict[str, Any]:
        """Parse email data"""
        return {"email": data, "type": "email"}
    
    def _parse_phone(self, data: str) -> Dict[str, Any]:
        """Parse phone data"""
        return {"phone": data, "type": "phone"}
    
    def _parse_json(self, data: str) -> Dict[str, Any]:
        """Parse JSON data"""
        try:
            return {"json_data": json.loads(data), "type": "json"}
        except:
            return {"raw_data": data, "type": "json_invalid"}
    
    def _parse_plain_text(self, data: str) -> Dict[str, Any]:
        """Parse plain text data"""
        return {"text": data, "type": "text"}



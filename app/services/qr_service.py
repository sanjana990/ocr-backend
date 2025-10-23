#!/usr/bin/env python3
"""
QR Code Detection and Parsing Service
Handles QR code detection, parsing, and content analysis
"""

import cv2
import numpy as np
from PIL import Image
import io
import re
import structlog
from typing import Dict, Any, List, Optional

logger = structlog.get_logger(__name__)

# QR code detection will use OpenCV only


class QRService:
    """Service for QR code detection and parsing"""
    
    def __init__(self):
        self.logger = logger
        self.pyzbar_available = False  # pyzbar removed
        
    def detect_qr_codes(self, image_data: bytes) -> list:
        """Detect QR codes in the image using multiple methods"""
        results = []
        
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.logger.warning("Failed to decode image")
                return []
            
            self.logger.info(f"🔍 Processing image: {cv_image.shape}")
            
            # Method 1: OpenCV QR detector with multiple preprocessing
            try:
                qr_detector = cv2.QRCodeDetector()
                # Try multi decode first on original image
                try:
                    retval, decoded_list, points, _ = qr_detector.detectAndDecodeMulti(cv_image)
                except Exception:
                    retval, decoded_list, points = False, [], None
                if retval and decoded_list:
                    for s in decoded_list:
                        if s:
                            results.append({
                                "data": s,
                                "type": "QRCODE",
                                "rect": {"x": 0, "y": 0, "width": 0, "height": 0},
                                "method": "opencv_multi_original"
                            })
                    self.logger.info(f"📱 QR Codes detected (multi original): {len(results)}")
                # Also try single decode on original
                data, pts, _ = qr_detector.detectAndDecode(cv_image)
                if data:
                    results.append({
                        "data": data,
                        "type": "QRCODE",
                        "rect": {"x": 0, "y": 0, "width": 0, "height": 0},
                        "method": "opencv_single_original"
                    })
                    self.logger.info(f"📱 QR Code detected (single original): {data[:50]}...")

                # If nothing found, try preprocessing variants
                if not results:
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    preprocessing_methods = [
                        ("grayscale", gray),
                        ("gaussian", cv2.GaussianBlur(gray, (3, 3), 0)),
                        ("median", cv2.medianBlur(gray, 3)),
                        ("threshold", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                        ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
                    ]
                    for method_name, processed_img in preprocessing_methods:
                        # Multi
                        found_any = False
                        try:
                            retval, decoded_list, points, _ = qr_detector.detectAndDecodeMulti(processed_img)
                        except Exception:
                            retval, decoded_list = False, []
                        if retval and decoded_list:
                            for s in decoded_list:
                                if s:
                                    results.append({
                                        "data": s,
                                        "type": "QRCODE",
                                        "rect": {"x": 0, "y": 0, "width": 0, "height": 0},
                                        "method": f"opencv_multi_{method_name}"
                                    })
                                    found_any = True
                            if found_any:
                                self.logger.info(f"📱 QR Codes detected (multi {method_name}): {len(results)}")
                                break
                        # Single
                        data, pts, _ = qr_detector.detectAndDecode(processed_img)
                        if data:
                            results.append({
                                "data": data,
                                "type": "QRCODE",
                                "rect": {"x": 0, "y": 0, "width": 0, "height": 0},
                                "method": f"opencv_single_{method_name}"
                            })
                            self.logger.info(f"📱 QR Code detected (single {method_name}): {data[:50]}...")
                            break
            except Exception as e:
                self.logger.warning("OpenCV QR detection failed", error=str(e))
            
            # Method 3: Enhanced detection with preprocessing
            if not results:
                try:
                    # Convert to grayscale
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply different preprocessing techniques
                    preprocessing_methods = [
                        ("original", gray),
                        ("gaussian", cv2.GaussianBlur(gray, (3, 3), 0)),
                        ("median", cv2.medianBlur(gray, 3)),
                        ("morphology", cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))),
                        ("threshold", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
                    ]
                    
                    for method_name, processed_img in preprocessing_methods:
                        try:
                            # Try OpenCV with preprocessed image
                            qr_detector = cv2.QRCodeDetector()
                            result = qr_detector.detectAndDecode(processed_img)
                            
                            # Handle different OpenCV versions
                            if len(result) == 3:
                                retval, decoded_info, points = result
                            elif len(result) == 4:
                                retval, decoded_info, points, _ = result
                            else:
                                self.logger.warning(f"Unexpected OpenCV result format: {len(result)} values")
                                retval, decoded_info, points = False, None, None
                            
                            # Check if we have valid detection results
                            has_valid_result = (
                                isinstance(retval, (bool, np.bool_)) and bool(retval) and 
                                decoded_info is not None and decoded_info != ""
                            )
                            
                            if has_valid_result:
                                qr_info = {
                                    "data": decoded_info,
                                    "type": "QRCODE",
                                    "rect": {"x": 0, "y": 0, "width": 0, "height": 0},
                                    "method": f"opencv_{method_name}"
                                }
                                results.append(qr_info)
                                self.logger.info(f"📱 QR Code detected (OpenCV {method_name}): {decoded_info[:50]}...")
                                
                        except Exception as e:
                            self.logger.warning(f"Preprocessing method {method_name} failed", error=str(e))
                            
                except Exception as e:
                    self.logger.warning("Enhanced detection failed", error=str(e))
            
            # Remove duplicates based on data content
            unique_results = []
            seen_data = set()
            for result in results:
                if result["data"] not in seen_data:
                    seen_data.add(result["data"])
                    unique_results.append(result)
            
            self.logger.info(f"✅ QR detection complete: {len(unique_results)} unique codes found")
            return unique_results
            
        except Exception as e:
            self.logger.error("QR code detection failed", error=str(e))
            return []

    def parse_qr_content(self, qr_data: str) -> dict:
        """Parse QR code content and extract structured information"""
        parsed_info = {
            "content_type": "unknown",
            "title": "",
            "details": {},
            "raw_data": qr_data
        }
        
        try:
            # URL detection
            if qr_data.startswith(('http://', 'https://', 'www.')):
                parsed_info["content_type"] = "url"
                parsed_info["title"] = "Website Link"
                parsed_info["details"] = {"url": qr_data}
                
            # Email detection
            elif '@' in qr_data and '.' in qr_data:
                parsed_info["content_type"] = "email"
                parsed_info["title"] = "Email Address"
                parsed_info["details"] = {"email": qr_data}
                
            # Phone number detection
            elif re.match(r'^\+?[\d\s\-\(\)]{10,}$', qr_data):
                parsed_info["content_type"] = "phone"
                parsed_info["title"] = "Phone Number"
                parsed_info["details"] = {"phone": qr_data}
                
            # vCard format (business card)
            elif qr_data.startswith('BEGIN:VCARD'):
                parsed_info["content_type"] = "vcard"
                parsed_info["title"] = "Business Card"
                parsed_info["details"] = self.parse_vcard(qr_data)
                
            # WiFi network
            elif qr_data.startswith('WIFI:'):
                parsed_info["content_type"] = "wifi"
                parsed_info["title"] = "WiFi Network"
                parsed_info["details"] = self.parse_wifi(qr_data)
                
            # SMS
            elif qr_data.startswith('sms:'):
                parsed_info["content_type"] = "sms"
                parsed_info["title"] = "SMS Message"
                parsed_info["details"] = self.parse_sms(qr_data)
                
            # Plain text
            else:
                parsed_info["content_type"] = "text"
                parsed_info["title"] = "Text Content"
                parsed_info["details"] = {"text": qr_data}
                
        except Exception as e:
            self.logger.warning(f"QR content parsing failed: {e}")
            parsed_info["content_type"] = "error"
            parsed_info["title"] = "Unknown Content"
            parsed_info["details"] = {"error": str(e)}
        
        return parsed_info

    def parse_vcard(self, vcard_data: str) -> dict:
        """Parse vCard format QR code"""
        vcard_info = {}
        lines = vcard_data.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('FN:'):
                vcard_info['name'] = line[3:]
            elif line.startswith('ORG:'):
                vcard_info['company'] = line[4:]
            elif line.startswith('TEL:'):
                vcard_info['phone'] = line[4:]
            elif line.startswith('EMAIL:'):
                vcard_info['email'] = line[6:]
            elif line.startswith('URL:'):
                vcard_info['website'] = line[4:]
            elif line.startswith('ADR:'):
                vcard_info['address'] = line[4:]
        
        return vcard_info

    def parse_wifi(self, wifi_data: str) -> dict:
        """Parse WiFi QR code format"""
        wifi_info = {}
        # Format: WIFI:T:WPA;S:NetworkName;P:Password;H:false;;
        parts = wifi_data[5:].split(';')
        
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                if key == 'S':
                    wifi_info['ssid'] = value
                elif key == 'P':
                    wifi_info['password'] = value
                elif key == 'T':
                    wifi_info['security'] = value
        
        return wifi_info

    def parse_sms(self, sms_data: str) -> dict:
        """Parse SMS QR code format"""
        # Format: sms:+1234567890:Message text
        if ':' in sms_data:
            parts = sms_data.split(':', 2)
            if len(parts) >= 3:
                return {
                    'phone': parts[1],
                    'message': parts[2]
                }
        return {'raw': sms_data}

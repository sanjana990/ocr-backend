#!/usr/bin/env python3
"""
Image Processing Configuration
Centralized configuration for image preprocessing methods
"""

import cv2
import numpy as np
from typing import List, Tuple, Callable


class ImageConfig:
    """Configuration for image preprocessing methods"""
    
    @staticmethod
    def get_preprocessing_methods() -> List[Tuple[str, Callable]]:
        """Get list of preprocessing methods for image enhancement"""
        return [
            ("original", lambda img: img),
            ("grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            ("blurred", lambda img: cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)),
            ("threshold", lambda img: cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive", lambda img: cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("morphology", lambda img: cv2.morphologyEx(cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.MORPH_CLOSE, np.ones((2,2), np.uint8)))
        ]
    
    @staticmethod
    def get_qr_preprocessing_methods(gray_image) -> List[Tuple[str, np.ndarray]]:
        """Get preprocessing methods specifically for QR code detection"""
        return [
            ("original", gray_image),
            ("gaussian", cv2.GaussianBlur(gray_image, (3, 3), 0)),
            ("median", cv2.medianBlur(gray_image, 3)),
            ("morphology", cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))),
            ("threshold", cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        ]

#!/usr/bin/env python3
"""
Business Card Parsing Service
Handles business card text parsing and structured data extraction
"""

import re
import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)


class BusinessCardService:
    """Service for business card text parsing and data extraction"""
    
    def __init__(self):
        self.logger = logger
    
    def extract_business_card_info(self, text: str) -> dict:
        """Extract structured business card information from OCR text"""
        info = {
            "name": "",
            "title": "",
            "company": "",
            "phone": "",
            "email": "",
            "website": "",
            "address": "",
            "otherInfo": []
        }

        if not text:
            return info

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Name detection (look for clean name patterns, not mixed with other data)
        name_pattern = r'^[A-Z][a-z]+ [A-Z][a-z]+$'  # Exact match for clean names
        for line in lines:
            clean_line = line.strip()
            # Check if line is just a name (no numbers, emails, or other data)
            if (re.match(name_pattern, clean_line) and 
                '@' not in clean_line and 
                '+' not in clean_line and 
                'www' not in clean_line.lower() and
                'http' not in clean_line.lower() and
                len(clean_line) < 50):  # Reasonable name length
                info["name"] = clean_line
                break
        
        # Fallback: try to extract name from first few words of first line
        if not info["name"] and lines:
            first_line = lines[0]
            words = first_line.split()
            if len(words) >= 2:
                # Take first 2 words that look like a name
                potential_name = ' '.join(words[:2])
                if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', potential_name):
                    info["name"] = potential_name

        # Email detection
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_match = re.search(email_pattern, text)
        if email_match:
            info["email"] = email_match.group()

        # Phone detection
        phone_pattern = r'(\+?[\d\s\-\(\)]{10,})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            info["phone"] = phone_match.group().strip()

        # Website detection
        website_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
        website_match = re.search(website_pattern, text, re.IGNORECASE)
        if website_match:
            info["website"] = website_match.group()

        # Company detection (look for common company suffixes)
        company_pattern = r'([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd|Company|Co\.|Pvt|Private))'
        company_match = re.search(company_pattern, text, re.IGNORECASE)
        if company_match:
            info["company"] = company_match.group().strip()

        # Title detection (look for job titles in clean lines)
        title_pattern = r'(Manager|Director|CEO|CTO|Analyst|Engineer|Developer|Consultant|Specialist|Executive|President|Vice President|VP|Designer|UI|UX|Developer|Coordinator|Assistant|Lead|Senior|Junior)'
        
        # Look for title in lines that don't contain other data
        for line in lines:
            clean_line = line.strip()
            if (re.search(title_pattern, clean_line, re.IGNORECASE) and 
                '@' not in clean_line and 
                '+' not in clean_line and 
                'www' not in clean_line.lower() and
                len(clean_line) < 100):  # Reasonable title length
                info["title"] = clean_line
                break
        
        # Fallback: try to find title in the text
        if not info["title"]:
            title_match = re.search(title_pattern, text, re.IGNORECASE)
            if title_match:
                info["title"] = title_match.group()

        # Address detection (look for street patterns)
        address_pattern = r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd))'
        address_match = re.search(address_pattern, text, re.IGNORECASE)
        if address_match:
            info["address"] = address_match.group().strip()

        return info

"""
AI-powered structured data extraction service using OpenAI
"""
import json
import logging
from typing import Dict, Any, Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class AIExtractionService:
    def __init__(self):
        """Initialize the AI extraction service with OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and OpenAI:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            if not OpenAI:
                logger.warning("OpenAI module not available, will use fallback extraction")
            else:
                logger.warning("OpenAI API key not found, will use fallback extraction")
        
    async def extract_business_card_data(self, ocr_text: str) -> Dict[str, Any]:
        """
        Extract structured business card information using AI
        """
        try:
            if not self.client:
                logger.warning("OpenAI client not available, falling back to regex")
                return self._fallback_extraction(ocr_text)
            
            # Create the prompt for structured extraction
            prompt = self._create_extraction_prompt(ocr_text)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using the more cost-effective model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured information from business card text. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            # Parse the JSON response
            extracted_data = json.loads(response.choices[0].message.content)
            
            logger.info("✅ AI extraction successful", extracted_data=extracted_data)
            return extracted_data
            
        except json.JSONDecodeError as e:
            logger.error("❌ Failed to parse AI response as JSON", error=str(e))
            return self._fallback_extraction(ocr_text)
        except Exception as e:
            logger.error("❌ AI extraction failed", error=str(e))
            return self._fallback_extraction(ocr_text)
    
    def _create_extraction_prompt(self, ocr_text: str) -> str:
        """Create a structured prompt for business card data extraction"""
        return f"""
Extract structured information from this business card text. Return ONLY a valid JSON object with these exact fields:

{{
    "name": "Full name (first and last name only)",
    "title": "Job title or position",
    "company": "Company name",
    "email": "Email address",
    "phone": "Phone number",
    "website": "Website URL",
    "address": "Physical address",
    "otherInfo": ["Any additional relevant information"]
}}

Rules:
- For "name": Extract only the person's full name (e.g., "John Smith"), not titles or other text
- For "title": Extract job title (e.g., "Software Engineer", "CEO", "Marketing Manager")
- For "company": Extract company name only
- For "email": Extract email address if present
- For "phone": Extract phone number if present
- For "website": Extract website URL if present
- For "address": Extract physical address if present
- For "otherInfo": Array of any other relevant information
- If a field is not found, use empty string ""
- Be precise and avoid including extra text in fields

Business card text:
{ocr_text}

JSON response:"""
    
    def _fallback_extraction(self, ocr_text: str) -> Dict[str, Any]:
        """Fallback to regex-based extraction if AI fails"""
        import re
        
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
        
        if not ocr_text:
            return info
        
        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        
        # Name detection (look for clean name patterns)
        name_pattern = r'^[A-Z][a-z]+ [A-Z][a-z]+$'
        for line in lines:
            clean_line = line.strip()
            if (re.match(name_pattern, clean_line) and 
                '@' not in clean_line and 
                '+' not in clean_line and 
                'www' not in clean_line.lower() and
                len(clean_line) < 50):
                info["name"] = clean_line
                break
        
        # Fallback name extraction
        if not info["name"] and lines:
            first_line = lines[0]
            words = first_line.split()
            if len(words) >= 2:
                potential_name = ' '.join(words[:2])
                if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', potential_name):
                    info["name"] = potential_name
        
        # Email detection
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_match = re.search(email_pattern, ocr_text)
        if email_match:
            info["email"] = email_match.group()
        
        # Phone detection
        phone_pattern = r'(\+?[\d\s\-\(\)]{10,})'
        phone_match = re.search(phone_pattern, ocr_text)
        if phone_match:
            info["phone"] = phone_match.group().strip()
        
        # Website detection
        website_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
        website_match = re.search(website_pattern, ocr_text, re.IGNORECASE)
        if website_match:
            info["website"] = website_match.group()
        
        # Title detection
        title_pattern = r'(Manager|Director|CEO|CTO|Analyst|Engineer|Developer|Consultant|Specialist|Executive|President|Vice President|VP|Designer|UI|UX|Developer|Coordinator|Assistant|Lead|Senior|Junior)'
        for line in lines:
            clean_line = line.strip()
            if (re.search(title_pattern, clean_line, re.IGNORECASE) and 
                '@' not in clean_line and 
                '+' not in clean_line and 
                'www' not in clean_line.lower() and
                len(clean_line) < 100):
                info["title"] = clean_line
                break
        
        # Company detection
        company_pattern = r'([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd|Company|Co\.|Pvt|Private))'
        company_match = re.search(company_pattern, ocr_text, re.IGNORECASE)
        if company_match:
            info["company"] = company_match.group().strip()
        
        # Address detection
        address_pattern = r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd))'
        address_match = re.search(address_pattern, ocr_text, re.IGNORECASE)
        if address_match:
            info["address"] = address_match.group().strip()
        
        return info

# Create singleton instance
ai_extraction_service = AIExtractionService()

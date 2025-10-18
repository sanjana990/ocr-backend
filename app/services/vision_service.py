#!/usr/bin/env python3
"""
Vision-based Business Card Analysis Service
Uses GPT-4o-mini with vision capabilities for structured business card analysis
"""

import base64
import json
import structlog
from typing import Dict, Any, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = structlog.get_logger(__name__)

class VisionService:
    """Service for analyzing business cards using GPT-4o-mini vision capabilities"""
    
    def __init__(self):
        """Initialize the vision service with OpenAI client"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"
        
    async def analyze_business_card(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze business card image using GPT-4o-mini vision
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict containing structured business card information
        """
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create the analysis prompt
            prompt = self._create_business_card_prompt()
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            structured_data = self._parse_analysis_response(analysis_text)
            
            logger.info("✅ Vision analysis completed", 
                       confidence=structured_data.get('confidence', 0),
                       fields_extracted=len([k for k, v in structured_data.get('contact_info', {}).items() if v]))
            
            return {
                "success": True,
                "method": "vision_analysis",
                "confidence": structured_data.get('confidence', 0.0),
                "structured_info": structured_data,
                "raw_analysis": analysis_text
            }
            
        except Exception as e:
            logger.error("❌ Vision analysis failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "method": "vision_analysis",
                "confidence": 0.0,
                "structured_info": {}
            }
    
    def _create_business_card_prompt(self) -> str:
        """Create the prompt for business card analysis"""
        return """
Analyze this business card image and extract structured information. Return your analysis as a JSON object with the following structure:

{
  "confidence": 0.95,
  "contact_info": {
    "name": "John Smith",
    "title": "Software Engineer",
    "company": "Tech Corp",
    "phone": "+1-555-123-4567",
    "email": "john@techcorp.com",
    "website": "www.techcorp.com",
    "address": "123 Main St, City, State 12345"
  },
  "analysis_notes": "Clear, well-formatted business card with all standard contact information",
  "quality_assessment": {
    "image_quality": "good",
    "text_clarity": "excellent",
    "layout_complexity": "simple"
  }
}

Guidelines:
- Extract ALL visible contact information
- Use null for missing fields
- Provide confidence score (0.0-1.0) based on clarity and completeness
- Include analysis notes about the card's quality and any challenges
- Be precise with phone numbers, emails, and addresses
- If you see multiple people or companies, extract the primary contact
- Handle both traditional and digital business card formats

Return ONLY the JSON object, no additional text.
"""
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the GPT response and extract structured data"""
        try:
            # Try to extract JSON from the response
            # Look for JSON object in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing if JSON extraction fails
                return self._fallback_parse(response_text)
                
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON response", error=str(e))
            return self._fallback_parse(response_text)
        except Exception as e:
            logger.error("Error parsing analysis response", error=str(e))
            return {
                "confidence": 0.0,
                "contact_info": {},
                "analysis_notes": f"Parsing error: {str(e)}",
                "quality_assessment": {}
            }
    
    def _fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        # Simple text-based extraction as fallback
        contact_info = {}
        
        # Extract common patterns
        import re
        
        # Email extraction
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', response_text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone extraction
        phone_match = re.search(r'[\+]?[1-9]?[0-9]{7,15}', response_text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        return {
            "confidence": 0.5,
            "contact_info": contact_info,
            "analysis_notes": "Fallback parsing used - may be incomplete",
            "quality_assessment": {
                "image_quality": "unknown",
                "text_clarity": "unknown",
                "layout_complexity": "unknown"
            }
        }
    
    async def compare_with_ocr(self, image_data: bytes, ocr_text: str) -> Dict[str, Any]:
        """
        Compare vision analysis with OCR results for enhanced accuracy
        
        Args:
            image_data: Raw image bytes
            ocr_text: Text extracted by OCR
            
        Returns:
            Combined analysis with confidence scoring
        """
        try:
            # Get vision analysis
            vision_result = await self.analyze_business_card(image_data)
            
            # Compare with OCR text
            comparison = self._compare_results(
                vision_result.get('structured_info', {}).get('contact_info', {}),
                ocr_text
            )
            
            return {
                "success": True,
                "vision_analysis": vision_result,
                "ocr_text": ocr_text,
                "comparison": comparison,
                "recommended_result": self._get_recommended_result(vision_result, ocr_text, comparison)
            }
            
        except Exception as e:
            logger.error("❌ Comparison analysis failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def _compare_results(self, vision_contact: Dict[str, str], ocr_text: str) -> Dict[str, Any]:
        """Compare vision analysis with OCR text"""
        matches = 0
        total_fields = 0
        
        for field, value in vision_contact.items():
            if value and value.lower() in ocr_text.lower():
                matches += 1
            if value:
                total_fields += 1
        
        accuracy = matches / total_fields if total_fields > 0 else 0
        
        return {
            "accuracy": accuracy,
            "matches": matches,
            "total_fields": total_fields,
            "confidence": "high" if accuracy > 0.8 else "medium" if accuracy > 0.5 else "low"
        }
    
    def _get_recommended_result(self, vision_result: Dict, ocr_text: str, comparison: Dict) -> Dict[str, Any]:
        """Get the recommended result based on comparison"""
        if comparison.get('confidence') == 'high':
            return vision_result
        else:
            # Fall back to OCR with some vision insights
            return {
                "method": "hybrid",
                "confidence": 0.7,
                "text": ocr_text,
                "vision_insights": vision_result.get('structured_info', {})
            }

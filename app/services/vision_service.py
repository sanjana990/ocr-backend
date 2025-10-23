#!/usr/bin/env python3
"""
Vision-based Business Card Analysis Service
Uses Gemini 2.5 Flash with vision capabilities for structured business card analysis
with stricter JSON output and image preprocessing to reduce gibberish results.
"""

import io
import base64
import json
import structlog
from typing import Dict, Any, Optional
from google import genai
import os
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageEnhance

# Load environment variables from .env file
load_dotenv()

logger = structlog.get_logger(__name__)

class VisionService:
    """Service for analyzing business cards using Gemini 2.5 Flash vision capabilities"""
    
    def __init__(self):
        """Initialize the vision service with Gemini client"""
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('gemini_api_key')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not configured in environment variables")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        
    async def analyze_business_card(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze business card image using Gemini 2.5 Flash with preprocessing and strict JSON forcing.
        """
        try:
            from google.genai import types
            
            prompt = self._create_business_card_prompt()
            # Preprocess image
            pre_bytes, mime_type, dbg = self._preprocess_image(image_data)

            # Call Gemini with proper API structure
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=pre_bytes
                        )
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                    # response_mime_type removed - doesn't work reliably with this SDK version
                ),
            )

            analysis_text = (response.text or "").strip()
            structured_data = self._parse_analysis_response(analysis_text)
            
            logger.info("✅ Vision analysis completed", 
                       confidence=structured_data.get('confidence', 0),
                       fields_extracted=len([k for k, v in structured_data.get('contact_info', {}).items() if v]),
                       preprocess=dbg)

            # Fallback if sparse or empty
            needs_fallback = not structured_data or not structured_data.get('contact_info')
            if needs_fallback:
                transcript = await self._transcribe_text_from_image(pre_bytes, mime_type)
                structured_from_text = await self._structure_from_text(transcript)
                base_count = len([v for v in structured_data.get('contact_info', {}).values() if v]) if structured_data else 0
                text_count = len([v for v in structured_from_text.get('contact_info', {}).values() if v]) if structured_from_text else 0
                final_struct = structured_from_text if text_count >= base_count else structured_data
                return {
                    "success": True,
                    "method": "vision_analysis_gemini_fallback" if final_struct is structured_from_text else "vision_analysis_gemini",
                    "confidence": final_struct.get('confidence', 0.0),
                    "structured_info": final_struct,
                    "raw_analysis": analysis_text,
                    "transcript_used": True,
                }
            else:
                return {
                    "success": True,
                    "method": "vision_analysis_gemini",
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
        # Emphasize strict JSON only in response
        return """You are an expert OCR and document analysis system specializing in business card extraction. Analyze the provided business card image with precision.

CRITICAL: Return ONLY a valid JSON object. No text before or after the JSON. No markdown code blocks. No backticks. No explanations.

START YOUR RESPONSE WITH { AND END WITH }

TASK: Extract all contact information from the business card and return ONLY a valid JSON object.

REQUIRED JSON STRUCTURE:
{
 "confidence": 0.95,
 "contact_info": {
   "name": "string or null",
   "title": "string or null", 
   "company": "string or null",
   "phone": "string or null",
   "mobile": "string or null",
   "fax": "string or null",
   "email": "string or null",
   "website": "string or null",
   "linkedin": "string or null",
   "address": "string or null",
   "additional_contacts": []
 },
 "visual_elements": {
   "logo_present": true/false,
   "card_color": "string",
   "layout_orientation": "horizontal/vertical"
 },
 "analysis_notes": "string",
 "quality_assessment": {
   "image_quality": "excellent/good/fair/poor",
   "text_clarity": "excellent/good/fair/poor",
   "layout_complexity": "simple/moderate/complex",
   "ocr_challenges": "string or null"
 },
 "language_detected": "string"
}

CRITICAL EXTRACTION RULES:
1. Read EVERY piece of text visible on the card, no matter how small
2. Scan the ENTIRE image including edges, corners, and reverse side if visible
3. Look for text in multiple orientations (some cards have vertical text)
4. Distinguish between different types of phone numbers (office, mobile, fax)
5. Extract social media handles if present (LinkedIn, Twitter, etc.)
6. Preserve exact formatting for phone numbers and addresses as shown
7. If multiple contacts are present, extract ALL and note the primary one

OUTPUT REQUIREMENTS:
- Return ONLY the JSON object (no backticks, no markdown fences)
- No explanatory text before or after the JSON
- Ensure valid JSON syntax (proper quotes, commas, brackets)
- Use null (not "null" or empty string) for missing fields
- Escape special characters properly
- If NO business card is detected in the image, return:
  {
    "error": "No business card detected in image",
    "confidence": 0.0
  }

Now analyze the business card image provided."""
    
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

    def _preprocess_image(self, image_data: bytes):
        """Lightweight preprocessing with PIL to improve OCR robustness."""
        debug = {}
        try:
            im = Image.open(io.BytesIO(image_data))
            debug["orig_size"] = im.size
            # Auto-orient
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            # Grayscale and normalize
            im = im.convert("L")
            im = ImageOps.autocontrast(im)
            im = ImageEnhance.Contrast(im).enhance(1.3)
            im = ImageEnhance.Sharpness(im).enhance(1.15)
            # Simple binarization
            im = im.point(lambda p: 255 if p > 175 else 0)
            # Resize longest side to <= 1600px
            max_side = 1600
            w, h = im.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            debug["proc_size"] = im.size
            # Encode JPEG
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=92, optimize=True)
            return out.getvalue(), "image/jpeg", debug
        except Exception as e:
            logger.warning("Preprocess failed, using original bytes", error=str(e))
            return image_data, self._guess_mime_type(image_data), {"preprocess": "failed"}

    def _guess_mime_type(self, image_data: bytes) -> str:
        try:
            im = Image.open(io.BytesIO(image_data))
            fmt = (im.format or "").upper()
            if fmt == "PNG":
                return "image/png"
            if fmt in ("JPG", "JPEG"):
                return "image/jpeg"
            if fmt == "WEBP":
                return "image/webp"
            if fmt == "GIF":
                return "image/gif"
            return "image/jpeg"
        except Exception:
            return "image/jpeg"

    async def _transcribe_text_from_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Ask Gemini to transcribe ALL text from the image (no structure)."""
        try:
            from google.genai import types
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "Transcribe ALL text exactly as it appears in the image. Return plain text only.",
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1500,
                ),
            )
            return (response.text or "").strip()
        except Exception as e:
            logger.warning("Transcript step failed", error=str(e))
            return ""

    async def _structure_from_text(self, transcript: str) -> Dict[str, Any]:
        """Ask Gemini to map plain text transcript to the JSON schema only."""
        if not transcript:
            return {}
        from google.genai import types
        
        instruction = (
            "Return ONLY a valid JSON object mapping this business card text to the required schema. "
            "Use null for missing fields. No prose."
        )
        schema_prompt = (
            instruction
            + "\n\nTEXT:\n" + transcript[:6000]
            + "\n\nREQUIRED JSON KEYS: confidence, contact_info{ name,title,company,phone,mobile,fax,email,website,linkedin,address,additional_contacts }, visual_elements{ logo_present,card_color,layout_orientation }, analysis_notes, quality_assessment{ image_quality,text_clarity,layout_complexity,ocr_challenges }, language_detected"
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=schema_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                ),
            )
            text = (response.text or "").strip()
            return self._parse_analysis_response(text)
        except Exception as e:
            logger.warning("Structure-from-text failed", error=str(e))
            return {}

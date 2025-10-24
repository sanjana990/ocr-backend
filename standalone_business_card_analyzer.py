#!/usr/bin/env python3
"""
Standalone Business Card Analyzer
Extracts text and QR code information from business card images using OpenAI Vision API
"""

import base64
import json
import os
import sys
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

class BusinessCardAnalyzer:
    """Standalone business card analyzer using OpenAI Vision API"""
    
    def __init__(self):
        """Initialize the analyzer with OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Cost-effective model with vision capabilities
    
    def analyze_business_card(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze business card image and extract all information
        
        Args:
            image_path: Path to the business card image
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            return self.analyze_business_card_from_bytes(image_data)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "structured_info": {}
            }
    
    def analyze_business_card_from_bytes(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze business card from image bytes
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create comprehensive analysis prompt
            prompt = self._create_analysis_prompt()
            
            # Call OpenAI Vision API
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
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            structured_data = self._parse_analysis_response(analysis_text)
            
            return {
                "success": True,
                "structured_info": structured_data,
                "raw_analysis": analysis_text
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "structured_info": {}
            }
    
    def _create_analysis_prompt(self) -> str:
        """Create comprehensive analysis prompt for business card extraction"""
        return """
Analyze this business card image and extract ALL information including text content and QR codes. 

INSTRUCTIONS:
1. Extract all visible text information from the business card
2. If there are QR codes visible, describe what they contain (you can see QR codes but cannot decode them directly)
3. Look for any URLs, social media handles, or other digital contact information
4. Extract contact information in a structured format

Return your analysis as a JSON object with this EXACT structure:

{
  "confidence": 0.95,
  "contact_info": {
    "name": "Full name",
    "title": "Job title or position", 
    "company": "Company name",
    "phone": "Phone number",
    "email": "Email address",
    "website": "Website URL",
    "address": "Physical address",
    "social_media": "Social media handles if any"
  },
  "qr_codes": [
    {
      "visible": true,
      "description": "Description of what the QR code appears to contain",
      "location": "Where on the card the QR code is located"
    }
  ],
  "additional_info": {
    "design_notes": "Any notable design elements",
    "digital_contacts": "Any digital contact methods found",
    "other_text": "Any other text or information visible"
  }
}

GUIDELINES:
- Extract ALL visible contact information accurately
- If you see QR codes, describe their likely content (contact info, website, etc.)
- Use null for missing fields
- Provide confidence score (0.0-1.0) based on clarity and completeness
- Be precise with phone numbers, emails, and addresses
- Look for both traditional and digital contact information
- If multiple people/companies, extract the primary contact

Return ONLY the JSON object, no additional text.
"""
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the OpenAI response and extract structured data"""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback parsing if JSON extraction fails
                return self._fallback_parse(response_text)
                
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            return self._fallback_parse(response_text)
        except Exception as e:
            print(f"Error parsing analysis response: {e}")
            return {
                "confidence": 0.0,
                "contact_info": {},
                "qr_codes": [],
                "additional_info": {}
            }
    
    def _fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        import re
        
        contact_info = {}
        
        # Extract common patterns
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', response_text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        phone_match = re.search(r'[\+]?[1-9]?[0-9]{7,15}', response_text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        return {
            "confidence": 0.5,
            "contact_info": contact_info,
            "qr_codes": [],
            "additional_info": {
                "design_notes": "Fallback parsing used",
                "digital_contacts": "",
                "other_text": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
        }
    
    def format_output(self, analysis_result: Dict[str, Any]) -> str:
        """Format the analysis result in the requested style"""
        if not analysis_result.get("success", False):
            return f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}"
        
        structured_info = analysis_result.get("structured_info", {})
        contact_info = structured_info.get("contact_info", {})
        qr_codes = structured_info.get("qr_codes", [])
        
        # Format the output
        output = []
        output.append("📋 EXTRACTED INFORMATION:")
        output.append("━" * 38)
        
        # Contact information
        if contact_info.get("name"):
            output.append(f"👤 Name: {contact_info['name']}")
        
        if contact_info.get("title"):
            output.append(f"💼 Title: {contact_info['title']}")
        
        if contact_info.get("company"):
            output.append(f"🏢 Company: {contact_info['company']}")
        
        if contact_info.get("phone"):
            output.append(f"📞 Phone: {contact_info['phone']}")
        
        if contact_info.get("email"):
            output.append(f"📧 Email: {contact_info['email']}")
        
        if contact_info.get("website"):
            output.append(f"🌐 Website: {contact_info['website']}")
        
        if contact_info.get("address"):
            output.append(f"📍 Address: {contact_info['address']}")
        
        if contact_info.get("social_media"):
            output.append(f"📱 Social Media: {contact_info['social_media']}")
        
        # QR Code information
        if qr_codes:
            output.append("")
            output.append("📱 QR CODE INFORMATION:")
            output.append("━" * 25)
            for i, qr in enumerate(qr_codes, 1):
                if qr.get("visible"):
                    output.append(f"🔗 QR Code {i}: {qr.get('description', 'Unknown content')}")
                    if qr.get("location"):
                        output.append(f"   📍 Location: {qr['location']}")
        
        # Additional information
        additional_info = structured_info.get("additional_info", {})
        if additional_info.get("digital_contacts"):
            output.append("")
            output.append("💻 DIGITAL CONTACTS:")
            output.append("━" * 20)
            output.append(f"🔗 {additional_info['digital_contacts']}")
        
        # Confidence score
        confidence = structured_info.get("confidence", 0)
        output.append("")
        output.append(f"🎯 Confidence: {confidence:.1%}")
        
        return "\n".join(output)

def main():
    """Main function to run the business card analyzer"""
    parser = argparse.ArgumentParser(description="Analyze business card images using OpenAI Vision")
    parser.add_argument("image_path", help="Path to the business card image file")
    parser.add_argument("--output", "-o", help="Output file to save results (optional)")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file '{args.image_path}' not found")
        sys.exit(1)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = BusinessCardAnalyzer()
        
        print("🔍 Analyzing business card...")
        print("━" * 30)
        
        # Analyze the image
        result = analyzer.analyze_business_card(args.image_path)
        
        # Format and display results
        formatted_output = analyzer.format_output(result)
        print(formatted_output)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print(f"\n💾 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

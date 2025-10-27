import time 

business_card_image_url = "https://jmnqgntwkmbrztxzsfek.supabase.co/storage/v1/object/public/ocr_bucket/scanned_cards/WhatsApp%20Image%202025-10-18%20at%2023.20.34(1).jpeg"

prompt = """
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
URL of the business card image:
"""


from google import genai
import os 
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
t0 = time.time()
response = client.chat.completions.create(
    model="gpt-4o-mini",
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
                        "url": business_card_image_url
                    }
                }
            ]
        }
    ],
    max_tokens=1500,
    temperature=0.1
)
            
t1 = time.time()

print(t1-t0,response.choices[0].message.content)


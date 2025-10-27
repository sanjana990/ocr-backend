from google import genai
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Validate the API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment")

# Initialize the GenAI client
client = genai.Client(api_key=api_key)

# Define the company name
company_name = "Microsoft Corporation"

# Define the prompt
prompt = f"""You are a research assistant. Extract factual company information and return ONLY valid JSON.

COMPANY TO RESEARCH: {company_name}

JSON STRUCTURE (required):
{{
  "Company name": "",
  "Description/tagline": "",
  "Products/services": "",
  "Location/headquarters": "",
  "Industry": "",
  "Number of employees": "",
  "Revenue": "",
  "Market Share": "",
  "Competitors": "",
  "Suggested sources": []
}}

CRITICAL QUALITY RULES:
1. ONLY include verifiable information from reliable sources
2. Use "N/A" for unavailable data - NEVER guess or estimate
3. Include year/date for temporal data: "500 (2023)" or "Approximately 1000 (as of 2024)"
4. For ranges, use LinkedIn-style: "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10000+"

RESEARCH PRIORITY (check in this order):
1. Official company website (About, Press)
2. LinkedIn company page
3. Crunchbase profile
4. Wikipedia (for established companies)
5. Recent reputable news articles

FORMATTING SPECIFICS:
- Description: Clear 1-2 sentence summary of what the company does
- Products/services: Specific offerings, not generic descriptions
- Location: City, State/Province, Country format
- Industry: Be specific (e.g., "SaaS - HR Technology" not just "Technology")
- Revenue: Include currency, amount, and year: "$10M USD (2023)" or "N/A"
- Market Share: Include % and scope: "12% in North American CRM market (2024)" or "N/A"
- Competitors: 3-5 direct competitors (companies offering similar products/services)
- Suggested sources: Array of actual URLs you can verify

QUALITY CHECKS:
✓ Every field has real data or "N/A"
✓ No fabricated statistics
✓ Sources are actual URLs
✓ JSON is valid and parseable
✓ Industry is specific, not generic

OUTPUT: Return ONLY the JSON object. No markdown, no explanations, no code blocks."""

print(f"🔍 Researching: {company_name}\n")

# Measure the time taken for the API call
t0 = time.perf_counter()

try:
    # Generate content using the GenAI client
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",  # Use newer, faster model
        contents=prompt,
        config={
            "temperature": 0.2,  # Slightly higher for better quality
            "max_output_tokens": 1500,  # Increased for complete response
            "top_p": 0.9,
            "top_k": 40
        }
    )
    
    t1 = time.perf_counter()
    print(f"✅ Total time: {t1 - t0:.3f}s\n")
    
    # Debug: Print the type and attributes of response
    print(f"Response type: {type(response)}")
    print(f"Response attributes: {dir(response)}\n")
    
    # Try multiple ways to extract text
    output_text = None
    
    # Method 1: Direct .text attribute
    if hasattr(response, 'text') and response.text:
        output_text = response.text
        print("✓ Extracted using .text attribute")
    
    # Method 2: Check candidates
    elif hasattr(response, 'candidates') and response.candidates:
        print(f"Found {len(response.candidates)} candidates")
        candidate = response.candidates[0]
        
        if hasattr(candidate, 'content'):
            content = candidate.content
            
            # Check for parts
            if hasattr(content, 'parts') and content.parts:
                output_text = ''.join([part.text for part in content.parts if hasattr(part, 'text')])
                print("✓ Extracted using candidates[0].content.parts")
    
    # Method 3: Try to access via dict
    elif isinstance(response, dict):
        output_text = response.get('text') or str(response)
        print("✓ Extracted from dict response")
    
    # If still no output, print full response for debugging
    if not output_text:
        print("❌ Could not extract text. Full response object:")
        print(response)
        
        # Try to convert to dict
        if hasattr(response, '__dict__'):
            print("\nResponse.__dict__:")
            print(response.__dict__)
    else:
        print(f"\n{'='*80}")
        print("RAW OUTPUT:")
        print('='*80)
        print(output_text)
        print('='*80)
        
        # Try to parse as JSON
        try:
            # Clean potential markdown code blocks
            cleaned_text = output_text
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_text:
                cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()
            
            company_data = json.loads(cleaned_text)
            
            print(f"\n{'='*80}")
            print("PARSED JSON OUTPUT:")
            print('='*80)
            print(json.dumps(company_data, indent=2))
            
            # Save to file
            filename = f"{company_name.replace(' ', '_').replace('.', '')}_research.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(company_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved to: {filename}")
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parsing error: {e}")
            print("The output might not be valid JSON")

except Exception as e:
    print(f"❌ Error during API call: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
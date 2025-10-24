#!/usr/bin/env python3
"""
Clean OCR Server - Minimal implementation with AI business card analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog
from datetime import datetime
import json
import re
import time
import os
import asyncio

# Import standalone business card analyzer
from standalone_business_card_analyzer import BusinessCardAnalyzer

# Import database service
try:
    from database_service import database_service
    DATABASE_SERVICE_AVAILABLE = database_service.available
    if DATABASE_SERVICE_AVAILABLE:
        print("✅ Database service available")
    else:
        print("⚠️ Database service not configured (Supabase credentials not set)")
except ImportError as e:
    DATABASE_SERVICE_AVAILABLE = False
    print(f"⚠️ Database service not available: {e}")

# Import MongoDB service
try:
    from mongodb_service import mongodb_service
    MONGODB_SERVICE_AVAILABLE = mongodb_service.available
    if MONGODB_SERVICE_AVAILABLE:
        print("✅ MongoDB service available")
    else:
        print("⚠️ MongoDB service not configured (MONGODB_URL not set)")
except ImportError as e:
    MONGODB_SERVICE_AVAILABLE = False
    print(f"⚠️ MongoDB service not available: {e}")

# Import Gemini for company research
try:
    from google import genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini available for company research")
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"⚠️ Gemini not available: {e}")
    print("Install with: pip install google-generativeai")

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Rate limiter for company research
class SimpleRateLimiter:
    """Simple rate limiter - 3 requests per 60 seconds with 2 second delays"""
    
    def __init__(self):
        self.requests = []
        self.last_request = 0
        self.max_requests = 3    # Max 3 requests per minute
        self.time_window = 60    # 60 seconds
        self.min_delay = 2       # 2 seconds between requests
    
    async def wait_if_needed(self):
        """Wait if we need to respect rate limits"""
        now = time.time()
        
        # Remove old requests (older than 60 seconds)
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # If we've hit the limit, wait until the oldest request expires
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (now - self.requests[0]) + 1
            logger.info(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
            now = time.time()
        
        # Wait minimum delay between requests
        if self.last_request > 0:
            time_since_last = now - self.last_request
            if time_since_last < self.min_delay:
                delay = self.min_delay - time_since_last
                logger.info(f"⏸️ Waiting {delay:.1f}s between requests...")
                await asyncio.sleep(delay)
                now = time.time()
        
        # Record this request
        self.requests.append(now)
        self.last_request = now
        logger.info(f"✅ Request approved ({len(self.requests)}/{self.max_requests} requests used)")

# Initialize rate limiter
rate_limiter = SimpleRateLimiter()

# Company extraction functions
def _company_extraction_instruction() -> str:
    return """You are a research assistant tasked with gathering factual company information. Follow these instructions carefully:
 
TASK: Extract company information and return ONLY a valid JSON object.
 
REQUIRED JSON STRUCTURE:
{
"Company name": "string",
"Description/tagline": "string",
"Products/services": "string",
"Location/headquarters": "string",
"Industry": "string",
"Number of employees": "string",
"Revenue": "string",
"Market Share": "string",
"Competitors": "string",
"Suggested sources": ["string"]
}
 
CRITICAL RULES TO PREVENT HALLUCINATION:
1. ONLY include information you can verify from actual sources
2. If you cannot find specific information, write "N/A" - do NOT guess or estimate
3. When stating facts, you must be able to cite where that information came from
4. If information is outdated, include the year (e.g., "500-1000 (as of 2023)")
5. For ambiguous data, use qualifiers like "approximately" or "estimated"
 
RESEARCH METHODOLOGY:
Priority order for sources:
1. Company's official website (About page, Press releases)
2. Company's LinkedIn page (About section)
3. Crunchbase profile
4. Wikipedia article (for established companies)
5. Recent news articles from reputable sources
6. SEC filings/10-K reports (for public companies only)
 
FORMATTING GUIDELINES:
- Number of employees: Use ranges "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10000+"
- Products/services: 1-2 sentence summary of main offerings
- Revenue: Include currency and year (e.g., "$50M USD (2023)" or "N/A")
- Market Share: Include percentage and geographic scope if available (e.g., "15% in North America (2023)" or "N/A")
- Competitors: List 3-5 direct competitors, comma-separated
- Suggested sources: Include actual URLs you referenced, prioritize LinkedIn and official website
 
OUTPUT REQUIREMENTS:
- Return ONLY the JSON object
- No explanatory text before or after
- Ensure valid JSON syntax (proper quotes, commas, brackets)
- Use double quotes for strings
- Escape special characters if needed
 
VERIFICATION CHECKLIST BEFORE RESPONDING:
- [ ] Every field has either real data or "N/A"
- [ ] No invented statistics or figures
- [ ] Sources list contains actual URLs
- [ ] JSON is valid and parseable
- [ ] No additional commentary outside JSON
 
Company to research: [INSERT COMPANY NAME HERE]"""

async def _extract_company_via_gemini(company_name: str) -> dict:
    """
    Use Gemini to produce structured JSON based on web knowledge.
    If your account has Google-grounded search enabled, Gemini will leverage it automatically.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
 
    client = genai.Client(api_key=gemini_key)
 
    instruction = _company_extraction_instruction()
    prompt = (
        f"{instruction}\n\n"
        f'Query: "{company_name}" company overview, about, products, HQ, industry, employees, revenue, market share, competitors.\n'
        f"Return ONLY the JSON."
    )
 
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    ai_text = (response.text or "").strip()
 
    # Clean JSON code fences if present
    cleaned = ai_text
    if "```json" in cleaned:
        m = re.search(r'```json\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
    elif "```" in cleaned:
        m = re.search(r'```\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
 
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {}
 
    required = [
        "Company name",
        "Description/tagline",
        "Products/services",
        "Location/headquarters",
        "Industry",
        "Number of employees",
        "Revenue",
        "Market Share",
        "Competitors",
    ]
    structured = {k: (str(parsed.get(k)) if parsed.get(k) not in [None, ""] else "N/A") for k in required}
 
    return {
        "structured_data": structured,
        "raw_ai_response": ai_text,
        "extraction_successful": bool(parsed),
        "parsing_successful": bool(parsed),
    }

# Initialize AI analyzer
try:
    ai_analyzer = BusinessCardAnalyzer()
    AI_ANALYZER_AVAILABLE = True
    print("✅ AI Business Card Analyzer available")
except Exception as e:
    AI_ANALYZER_AVAILABLE = False
    print(f"⚠️ AI Business Card Analyzer not available: {e}")
    print("Make sure OPENAI_API_KEY is set in environment variables")

# Create FastAPI application
app = FastAPI(
    title="Clean OCR Server",
    version="1.0.0",
    description="Minimal OCR server with AI business card analysis"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://localhost:5174",
        "http://localhost:8000",
        "https://fe-ocr-fe1.vercel.app",
        "https://*.onrender.com",
        "https://projects-main-liard.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Clean OCR Server",
        "status": "running",
        "ai_analyzer_available": AI_ANALYZER_AVAILABLE,
        "database_available": DATABASE_SERVICE_AVAILABLE,
        "mongodb_available": MONGODB_SERVICE_AVAILABLE,
        "gemini_available": GEMINI_AVAILABLE,
        "endpoints": {
            "ai_business_card": "/ai-business-card",
            "crawl_company": "/crawl-company",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_analyzer_available": AI_ANALYZER_AVAILABLE,
        "database_available": DATABASE_SERVICE_AVAILABLE,
        "mongodb_available": MONGODB_SERVICE_AVAILABLE,
        "gemini_available": GEMINI_AVAILABLE
    }

@app.post("/ai-business-card")
async def process_ai_business_card(file: UploadFile = File(...)):
    """Process business card using AI Vision analysis (OpenAI GPT-4o-mini)"""
    try:
        # Check if AI analyzer is available
        if not AI_ANALYZER_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="AI Business Card Analyzer not available. Please check OPENAI_API_KEY configuration."
            )
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🤖 Processing business card with AI Vision", 
                   filename=file.filename, 
                   file_size=len(content))
        
        # Process with AI analyzer
        result = ai_analyzer.analyze_business_card_from_bytes(content)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {result.get('error', 'Unknown error')}")
        
        structured_info = result.get("structured_info", {})
        contact_info = structured_info.get("contact_info", {})
        qr_codes = structured_info.get("qr_codes", [])
        
        logger.info("✅ AI business card analysis completed", 
                   filename=file.filename, 
                   success=result["success"],
                   confidence=structured_info.get("confidence", 0),
                   qr_count=len(qr_codes))
        
        # Save to Supabase if available
        saved_to_database = False
        if result["success"] and DATABASE_SERVICE_AVAILABLE:
            try:
                # Prepare data for saving
                card_data = {
                    **contact_info,
                    "raw_analysis": result.get("raw_analysis", ""),
                    "confidence": structured_info.get("confidence", 0.0),
                    "method": "ai_vision",
                    "qr_codes": qr_codes,
                    "qr_count": len(qr_codes),
                    "extracted_at": datetime.now().isoformat()
                }
                
                # Save to Supabase
                saved_card = await database_service.save_business_card_data(card_data)
                if saved_card:
                    logger.info(f"✅ AI business card data saved to Supabase: {contact_info.get('name', 'Unknown')}")
                    saved_to_database = True
                    
                    # Automatically trigger company research if company name is available
                    company_name = contact_info.get('company')
                    if company_name and MONGODB_SERVICE_AVAILABLE and GEMINI_AVAILABLE:
                        try:
                            logger.info(f"🏢 Auto-triggering company research for: {company_name}")
                            
                            # Apply rate limiting
                            await rate_limiter.wait_if_needed()
                            start_time = time.time()
                            
                            # Research company using Gemini
                            extracted = await _extract_company_via_gemini(company_name.strip())
                            elapsed = time.time() - start_time
                            
                            # Prepare company data for MongoDB
                            company_research_data = {
                                "company_name": company_name,
                                "url": None,
                                "platform": "gemini-search",
                                "content": None,
                                "ai_extracted_data": extracted,
                                "crawl_time": elapsed,
                                "crawled_at": datetime.now().isoformat(),
                                "company_details":card_data,
                                "triggered_by": "business_card_auto_research",
                                "business_card_id": saved_card.get('id') if isinstance(saved_card, dict) else None
                            }
                            
                            # Save company research to MongoDB
                            saved_company = await mongodb_service.save_crawl_data(company_research_data)
                            if saved_company:
                                logger.info(f"✅ Auto company research completed and saved to MongoDB: {company_name}")
                            else:
                                logger.warning(f"⚠️ Auto company research failed to save to MongoDB: {company_name}")
                                
                        except Exception as research_error:
                            logger.warning(f"⚠️ Auto company research failed: {research_error}")
                    elif company_name and not MONGODB_SERVICE_AVAILABLE:
                        logger.warning(f"⚠️ MongoDB not available for auto company research: {company_name}")
                    elif company_name and not GEMINI_AVAILABLE:
                        logger.warning(f"⚠️ Gemini not available for auto company research: {company_name}")
                    elif not company_name:
                        logger.info("ℹ️ No company name found in business card, skipping auto research")
                else:
                    logger.warning(f"⚠️ Failed to save AI business card data to Supabase")
                    
            except Exception as db_error:
                logger.warning(f"⚠️ Supabase save failed: {db_error}")
                # Continue with response even if database save fails
        
        return {
            "success": result["success"],
            "filename": file.filename,
            "method": "ai_vision",
            "structured_data": contact_info,
            "confidence": structured_info.get("confidence", 0.0),
            "qr_codes": qr_codes,
            "qr_count": len(qr_codes),
            "formatted_output": ai_analyzer.format_output(result),
            "raw_analysis": result.get("raw_analysis", ""),
            "additional_info": structured_info.get("additional_info", {}),
            "saved_to_database": saved_to_database,
            "database_available": DATABASE_SERVICE_AVAILABLE,
            "error": result.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ AI business card processing failed", err=str(e))
        return {
            "success": False,
            "error": str(e),
            "structured_data": {},
            "confidence": 0.0
        }

@app.post("/crawl-company")
async def crawl_company(
    company_name: str = Query(..., description="Company name to find"),
    use_ai_extraction: bool = Query(True, description="Kept for compatibility; ignored"),
    platform: str = Query("search", description="Deprecated. LLM search is always used.")
):
    """
    LLM-only company lookup (no crawl4ai):
    - Uses Gemini to return strict JSON with required fields.
    - Stores result in Supabase.
    """
    try:
        if not GEMINI_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Gemini not available. Install with: pip install google-generativeai"
            )
        
        if not company_name or not company_name.strip():
            raise HTTPException(status_code=400, detail="Company name is required")
 
        await rate_limiter.wait_if_needed()
        start = time.time()
 
        extracted = await _extract_company_via_gemini(company_name.strip())
        elapsed = time.time() - start
 
        response_data = {
            "success": True,
            "company_name": company_name,
            "url": None,
            "platform": "gemini-search",
            "crawl_time": elapsed,
            "content_length": 0,
            "company_data": extracted.get("structured_data"),
            "raw_ai_response": extracted.get("raw_ai_response"),
            "extraction_successful": extracted.get("extraction_successful", False),
            "parsing_successful": extracted.get("parsing_successful", False),
            "crawled_at": datetime.now().isoformat()
        }
 
        # Save to MongoDB if available
        if MONGODB_SERVICE_AVAILABLE:
            try:
                # Prepare data for saving to MongoDB
                company_data = {
                    "company_name": company_name,
                    "url": None,
                    "platform": "gemini-search",
                    "content": None,
                    "ai_extracted_data": extracted,                   
                    "crawl_time": elapsed,
                    "crawled_at": datetime.now().isoformat()
                }
                
                # Save to MongoDB using the MongoDB service
                saved_company = await mongodb_service.save_crawl_data(company_data)
                if saved_company:
                    logger.info(f"✅ Company data saved to MongoDB: {company_name}")
                    response_data["saved_to_database"] = True
                    response_data["database_type"] = "MongoDB"
                else:
                    logger.warning(f"⚠️ Failed to save company data to MongoDB")
                    response_data["saved_to_database"] = False
                    response_data["database_type"] = "MongoDB"
                    
            except Exception as db_err:
                logger.warning("⚠️ MongoDB save failed", err=str(db_err))
                response_data["saved_to_database"] = False
                response_data["database_error"] = str(db_err)
                response_data["database_type"] = "MongoDB"
        else:
            response_data["saved_to_database"] = False
            response_data["database_type"] = "MongoDB"
            logger.warning("⚠️ MongoDB not available, skipping data storage")
 
        logger.info("✅ LLM company lookup completed", company=company_name)
        return response_data
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ LLM company lookup failed", err=str(e))
        raise HTTPException(status_code=500, detail=f"Company lookup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Clean OCR Server...")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("🤖 AI Business Card: http://localhost:8000/ai-business-card")
    uvicorn.run(app, host="0.0.0.0", port=8000)

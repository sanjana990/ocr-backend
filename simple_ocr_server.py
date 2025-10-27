#!/usr/bin/env python3
"""
Simple OCR Server - Clean and reliable OCR implementation
"""
import sys
import os
from google import genai
import json
import re
# Make crawl4ai optional and expose availability flag
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except Exception:
    AsyncWebCrawler = None  # type: ignore
    CRAWL4AI_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Form, Response
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import structlog
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import time
from collections import deque
import asyncio
import json
import re

# QR code detection will use OpenCV only

# Import the new OCR service
from app.services.ocr_service import OCRService

# Import new modular services
from app.services.qr_service import QRService
from app.services.business_card_service import BusinessCardService
from app.services.image_processing_service import ImageProcessingService

# Import Apollo.io service
try:
    from apollo_service import apollo_service
    APOLLO_SERVICE_AVAILABLE = apollo_service.available
    if APOLLO_SERVICE_AVAILABLE:
        print("✅ Apollo.io service available")
    else:
        print("⚠️ Apollo.io service not configured (APOLLO_API_KEY not set)")
except ImportError as e:
    APOLLO_SERVICE_AVAILABLE = False
    print(f"⚠️ Apollo.io service not available: {e}")
    print("Install with: pip install apolloio aiohttp")

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
    print("Install with: pip install motor pymongo")

# Import Web scraping service
try:
    from web_scraping_service import web_scraping_service
    WEB_SCRAPING_AVAILABLE = web_scraping_service.available
    if WEB_SCRAPING_AVAILABLE:
        print("✅ Web scraping service available")
    else:
        print("⚠️ Web scraping service not configured (playwright not installed)")
except ImportError as e:
    WEB_SCRAPING_AVAILABLE = False
    print(f"⚠️ Web scraping service not available: {e}")
    print("Install with: pip install playwright")


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

# Rate limiter for crawl4ai requests
class SimpleRateLimiter:
    """Simple rate limiter - 3 requests per 60 seconds with 2 second delays"""
    
    def __init__(self):
        self.requests = deque()  # Track request times
        self.last_request = 0
        self.max_requests = 3    # Max 3 requests per minute
        self.time_window = 60    # 60 seconds
        self.min_delay = 2       # 2 seconds between requests
    
    async def wait_if_needed(self):
        """Wait if we need to respect rate limits"""
        now = time.time()
        
        # Remove old requests (older than 60 seconds)
        while self.requests and now - self.requests[0] > self.time_window:
            self.requests.popleft()
        
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

# Initialize services
ocr_service = OCRService()
qr_service = QRService()
business_card_service = BusinessCardService()
image_processing_service = ImageProcessingService()

# Initialize rate limiter for crawl4ai
rate_limiter = SimpleRateLimiter()

def detect_qr_codes(image_data: bytes) -> list:
    """Detect QR codes in the image using multiple methods"""
    return qr_service.detect_qr_codes(image_data)


def parse_qr_content(qr_data: str) -> dict:
    """Parse QR code content and extract structured information"""
    return qr_service.parse_qr_content(qr_data)


# QR parsing functions moved to QRService


def extract_business_card_info(text: str) -> dict:
    """Extract structured business card information from OCR text"""
    return business_card_service.extract_business_card_info(text)

# Create FastAPI application
app = FastAPI(
    title="Simple OCR Server",
    version="1.0.0",
    description="Clean and reliable OCR functionality"
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
        "https://*.onrender.com",  # Allow all Render domains
        "https://projects-main-liard.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Functions
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple OCR Server",
        "status": "running",
        "whatsapp_webhook": "/api/v1/whatsapp/webhook"
    }

@app.post("/")
async def root_post(request: Request):
    """Handle POST requests to root - likely from Twilio webhook misconfiguration"""
    try:
        # Check if this looks like a Twilio webhook request
        form_data = await request.form()
        if "From" in form_data and "whatsapp:" in str(form_data.get("From", "")):
            # Redirect to proper webhook endpoint
            from app.services.twilio_whatsapp_service import twilio_whatsapp_service
            return twilio_whatsapp_service.create_webhook_response(
                "⚠️ Webhook misconfigured. Please update Twilio webhook URL to: /api/v1/whatsapp/webhook"
            )
        else:
            return {"error": "Method Not Allowed - Use GET for root endpoint"}
    except Exception as e:
        return {"error": f"Invalid request: {str(e)}"}
@app.get("/health") 
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

def enhance_image_for_ocr(cv_image):
    """Apply multiple image enhancement techniques for better OCR"""
    return image_processing_service.enhance_image_for_ocr(cv_image)


def process_image_with_tesseract(image, cv_image):
    """Process image with Tesseract using multiple enhancement techniques"""
    return image_processing_service.process_image_with_tesseract(image, cv_image)
    


@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    """Process OCR with uploaded image using multi-engine OCR service"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🚀 Processing OCR request", 
                   filename=file.filename, 
                   file_size=len(content))
        
        # Debug: Log image details
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(content))
            logger.info("📸 Received image details", 
                       format=img.format, 
                       mode=img.mode, 
                       size=f"{img.width}x{img.height}")
        except Exception as e:
            logger.warning("⚠️ Could not analyze image", err=str(e))
        
        # Process with multi-engine OCR service
        result = await ocr_service.process_image(content, engine='auto')
        
        # Detect QR codes
        logger.info("🔍 Detecting QR codes...")
        qr_codes = detect_qr_codes(content)
        logger.info(f"📱 Found {len(qr_codes)} QR codes")
        
        logger.info("✅ OCR processing completed", 
                   filename=file.filename, 
                   success=result["success"],
                   text_length=len(result.get("text", "")),
                   confidence=result.get("confidence", 0),
                   engine=result.get("engine", "unknown"),
                   qr_count=len(qr_codes))
        
        return {
            "success": result["success"],
            "filename": file.filename,
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "engine": result.get("engine", "unknown"),
            "qr_codes": qr_codes,
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error("❌ OCR processing failed", err=str(e))
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "confidence": 0.0
        }


@app.post("/business-card")
async def process_business_card(file: UploadFile = File(...)):
    """Process business card with OCR + Vision analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🚀 Processing business card", 
                   filename=file.filename, 
                   file_size=len(content))
        
        # Process with enhanced business card extraction (OCR + Vision)
        result = await ocr_service.extract_business_card_data(content, use_vision=True)
        
        logger.info("✅ Business card processing completed", 
                   filename=file.filename, 
                   success=result["success"],
                   confidence=result.get("confidence", 0),
                   vision_available=result.get("vision_available", False),
                   qr_count=result.get("qr_count", 0))
        
        # Save OCR data to Supabase
        if result["success"]:
            try:
                from database_service import database_service
                
                # Prepare data for saving
                card_data = {
                    **result.get("data", {}),
                    "raw_text": result.get("raw_text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "engine_used": result.get("engine_used", "unknown"),
                    "qr_codes": result.get("qr_codes", []),
                    "qr_count": result.get("qr_count", 0),
                    "extracted_at": datetime.now().isoformat()
                }
                
                # Save to Supabase
                saved_card = await database_service.save_business_card_data(card_data)
                if saved_card:
                    logger.info(f"✅ Business card data saved to Supabase: {card_data.get('name', 'Unknown')}")
                else:
                    logger.warning(f"⚠️ Failed to save business card data to Supabase")
                    
            except Exception as db_error:
                logger.warning(f"⚠️ Supabase save failed: {db_error}")
                # Continue with response even if database save fails
        
        return {
            "success": result["success"],
            "filename": file.filename,
            "structured_data": result.get("data", {}),
            "confidence": result.get("confidence", 0.0),
            "raw_text": result.get("raw_text", ""),
            "engine_used": result.get("engine_used", "unknown"),
            "qr_codes": result.get("qr_codes", []),
            "qr_count": result.get("qr_count", 0),
            "vision_analysis": result.get("vision_analysis"),
            "vision_available": result.get("vision_available", False),
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error("❌ Business card processing failed", err=str(e))
        return {
            "success": False,
            "error": str(e),
            "structured_data": {},
            "confidence": 0.0
        }


@app.post("/qr-scan-goqr")
async def scan_qr_codes_goqr(
    file: UploadFile = File(...)
):
    """Scan QR codes using goQR.me API (backend proxy to avoid CORS)"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🔍 Starting goQR.me API scan", 
                   filename=file.filename, 
                   file_size=len(content))
        
        # Call goQR.me API from backend
        import aiohttp
        import io
        
        form_data = aiohttp.FormData()
        form_data.add_field('file', io.BytesIO(content), filename=file.filename, content_type=file.content_type)
        
        async with aiohttp.ClientSession() as session:
            async with session.post('https://api.qrserver.com/v1/read-qr-code/', data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"📊 goQR.me API response: {result}")
                    
                    # Parse the response
                    qr_codes = []
                    if result and len(result) > 0 and result[0].get('symbol'):
                        for symbol in result[0]['symbol']:
                            if symbol.get('data') and not symbol.get('error'):
                                qr_codes.append({
                                    "data": symbol['data'],
                                    "type": "QRCODE",
                                    "rect": {"x": 0, "y": 0, "width": 0, "height": 0},
                                    "method": "goQR.me API (backend proxy)"
                                })
                    
                    logger.info(f"✅ goQR.me API found {len(qr_codes)} QR codes")
                    return {
                        "success": True,
                        "filename": file.filename,
                        "qr_codes": qr_codes,
                        "count": len(qr_codes),
                        "method": "goQR.me API (backend proxy)"
                    }
                else:
                    # Get detailed error information
                    error_text = await response.text()
                    logger.warning(f"goQR.me API failed: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "filename": file.filename,
                        "qr_codes": [],
                        "count": 0,
                        "error": f"API request failed: {response.status} - {error_text}"
                    }
                    
    except Exception as e:
        logger.error(f"goQR.me API scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"QR scan failed: {str(e)}")




@app.post("/qr-scan")
async def scan_qr_codes(
    file: UploadFile = File(...),
    fetch_url_details: bool = False
):
    """Scan for QR codes in the image and optionally fetch URL details"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🔍 Starting QR code scan", 
                   filename=file.filename, 
                   file_size=len(content),
                   fetch_url_details=fetch_url_details)
        
        # Detect QR codes
        qr_codes = detect_qr_codes(content)
        
        # Parse QR codes
        parsed_data = {}
        qr_results = []
        
        for qr in qr_codes:
            qr_data = qr.get("data", "")
            qr_type = qr.get("type", "QRCODE")
            
            # Parse QR content
            parsed_info = parse_qr_content(qr_data)
            
            qr_results.append({
                "data": qr_data,
                "type": qr_type,
                "parsed": parsed_info,
                "rect": qr.get("rect", {})
            })
            
            # Add to parsed data if it contains useful information
            if parsed_info.get("details"):
                parsed_data.update(parsed_info["details"])
        
        # Fetch URL details if requested and URLs found
        url_details = {}
        if fetch_url_details and qr_codes:
            urls_to_fetch = []
            for qr in qr_codes:
                parsed_info = qr.get("parsed", {})
                if parsed_info.get("content_type") == "url" or qr.get("data", "").startswith(('http://', 'https://', 'www.')):
                    urls_to_fetch.append(qr["data"])
            
            if urls_to_fetch:
                try:
                    from app.services.qr_fetch_service import fetch_multiple_qr_urls
                    url_details = await fetch_multiple_qr_urls(urls_to_fetch)
                    logger.info("URL details fetched", 
                               urls_fetched=len(url_details),
                               successful_fetches=sum(1 for details in url_details.values() if details.get("success")))
                except Exception as e:
                    logger.warning("Failed to fetch URL details", err=str(e))
                    url_details = {"error": str(e)}
        
        logger.info("✅ QR code scanning completed", 
                   filename=file.filename, 
                   qr_count=len(qr_codes),
                   url_details_fetched=bool(url_details))
        
        return {
            "success": True,
            "filename": file.filename,
            "qr_codes": qr_results,
            "parsed_data": parsed_data,
            "count": len(qr_codes),
            "url_details": url_details if fetch_url_details else {},
            "error": None
        }
        
    except Exception as e:
        logger.error("❌ QR code scanning failed", filename=file.filename, err=str(e))
        return {
            "success": False,
            "filename": file.filename,
            "qr_codes": [],
            "parsed_data": {},
            "count": 0,
            "url_details": {},
            "error": str(e)
        }



@app.post("/batch-ocr")
async def batch_ocr(files: List[UploadFile] = File(...)):
    """Process multiple files for OCR"""
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
        
        results = []
        
        for file in files:
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "File must be an image",
                        "text": "",
                        "confidence": 0.0
                    })
                    continue
                
                # Read file content
                content = await file.read()
                
                logger.info("🚀 Processing batch file", 
                           filename=file.filename, 
                           file_size=len(content))
                
                # Process with multi-engine OCR service
                result = await ocr_service.process_image(content, engine='auto')
                
                # Detect QR codes
                qr_codes = detect_qr_codes(content)
                
                # Extract structured information: prefer Vision pipeline, fallback to AI text-only
                structured_info = None
                try:
                    vision_card = await ocr_service.extract_business_card_data(content, use_vision=True)
                    if vision_card.get("success"):
                        structured_info = vision_card.get("data", {})
                    else:
                        # Fallback to AI extraction from OCR text
                        if result.get("success") and result.get("text"):
                            from app.services.ai_extraction_service import ai_extraction_service
                            structured_info = await ai_extraction_service.extract_business_card_data(result.get("text", ""))
                except Exception as ex:
                    logger.warning("Structured extraction failed, using AI text-only fallback", error=str(ex))
                    if result.get("success") and result.get("text"):
                        from app.services.ai_extraction_service import ai_extraction_service
                        structured_info = await ai_extraction_service.extract_business_card_data(result.get("text", ""))
                
                results.append({
                    "filename": file.filename,
                    "success": result["success"],
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "engine": result.get("engine", "unknown"),
                    "qr_codes": qr_codes,
                    "structuredInfo": structured_info,
                    "error": result.get("error")
                })
                
            except Exception as e:
                logger.error("❌ Batch file processing failed", 
                           filename=file.filename, 
                           error=str(e))
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                    "text": "",
                    "confidence": 0.0
                })
        
        successful_count = sum(1 for r in results if r["success"])
        logger.info("✅ Batch processing completed", 
                   total_files=len(files),
                   successful=successful_count)
        
        return {
            "success": True,
            "total_files": len(files),
            "successful_files": successful_count,
            "results": results
        }
        
    except Exception as e:
        logger.error("❌ Batch OCR processing failed", err=str(e))
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


@app.post("/extract-url-content")
async def extract_url_content(request: dict):
    """Extract detailed information from a URL"""
    try:
        url = request.get('url')
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        logger.info(f"🌐 Extracting content from URL: {url}")
        
        # Use the URL content service
        from app.services.url_content_service import url_content_service
        result = await url_content_service.extract_contact_info(url)
        
        logger.info(f"✅ URL content extraction completed", 
                   success=result.get('success'),
                   title=result.get('title'),
                   emails_found=len(result.get('contact_info', {}).get('emails', [])),
                   phones_found=len(result.get('contact_info', {}).get('phones', [])))
        
        return result
        
    except Exception as e:
        logger.error(f"URL content extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")



@app.post("/comprehensive-business-card-analysis")
async def comprehensive_business_card_analysis(file: UploadFile = File(...)):
    """
    Comprehensive business card analysis with full data enrichment workflow:
    1. OCR extraction → Save to Supabase
    2. Extract company name → Apollo.io search → Save to MongoDB
    3. Web scraping → Save to MongoDB
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🚀 Starting comprehensive business card analysis", 
                   filename=file.filename, 
                   file_size=len(content))
        
        # Step 1: OCR Processing
        logger.info("📝 Step 1: OCR Processing")
        ocr_result = await ocr_service.extract_business_card_data(content, use_vision=True)
        
        if not ocr_result["success"]:
            raise HTTPException(status_code=400, detail="OCR processing failed")
        
        # Save OCR data to Supabase
        card_data = {
            **ocr_result.get("data", {}),
            "raw_text": ocr_result.get("raw_text", ""),
            "confidence": ocr_result.get("confidence", 0.0),
            "engine_used": ocr_result.get("engine_used", "unknown"),
            "qr_codes": ocr_result.get("qr_codes", []),
            "qr_count": ocr_result.get("qr_count", 0),
            "extracted_at": datetime.now().isoformat()
        }
        
        saved_card = None
        try:
            from database_service import database_service
            saved_card = await database_service.save_business_card_data(card_data)
            if saved_card:
                logger.info(f"✅ OCR data saved to Supabase: {card_data.get('name', 'Unknown')}")
        except Exception as db_error:
            logger.warning(f"⚠️ Supabase save failed: {db_error}")
        
        # Step 2: Company Enrichment with Apollo.io
        company_name = card_data.get("company")
        apollo_data = None
        web_scraped_data = None
        
        if company_name:
            logger.info(f"🏢 Step 2: Company Enrichment for {company_name}")
            
            # Apollo.io search
            if APOLLO_SERVICE_AVAILABLE:
                apollo_data = await apollo_service.search_company(company_name)
                logger.info(f"✅ Apollo.io data retrieved for {company_name}")
            else:
                logger.warning("⚠️ Apollo.io not available, skipping company enrichment")
            
            # Web scraping
            if WEB_SCRAPING_AVAILABLE and apollo_data:
                logger.info(f"🌐 Step 3: Web Scraping for {company_name}")
                website_url = apollo_data.get("website")
                web_scraped_data = await web_scraping_service.scrape_company_website(company_name, website_url)
                logger.info(f"✅ Web scraping completed for {company_name}")
            else:
                logger.warning("⚠️ Web scraping not available, skipping website analysis")
            
            # Save enriched data to MongoDB
            if MONGODB_SERVICE_AVAILABLE and (apollo_data or web_scraped_data):
                logger.info(f"💾 Step 4: Saving enriched data to MongoDB")
                
                enriched_data = {
                    "company_name": company_name,
                    "apollo_data": apollo_data,
                    "web_scraped_data": web_scraped_data,
                    "enriched_at": datetime.now().isoformat()
                }
                
                try:
                    saved_enriched = await mongodb_service.save_enriched_company_data(enriched_data)
                    if saved_enriched:
                        logger.info(f"✅ Enriched data saved to MongoDB: {company_name}")
                except Exception as mongo_error:
                    logger.warning(f"⚠️ MongoDB save failed: {mongo_error}")
            else:
                logger.warning("⚠️ MongoDB not available, skipping enriched data storage")
        
        # Prepare comprehensive response
        response_data = {
            "success": True,
            "filename": file.filename,
            "ocr_data": {
                "structured_data": card_data,
                "confidence": ocr_result.get("confidence", 0.0),
                "engine_used": ocr_result.get("engine_used", "unknown"),
                "qr_codes": ocr_result.get("qr_codes", []),
                "qr_count": ocr_result.get("qr_count", 0),
                "saved_to_supabase": saved_card is not None
            },
            "company_enrichment": {
                "company_name": company_name,
                "apollo_data": apollo_data,
                "web_scraped_data": web_scraped_data,
                "enrichment_successful": apollo_data is not None or web_scraped_data is not None
            },
            "data_storage": {
                "supabase_saved": saved_card is not None,
                "mongodb_available": MONGODB_SERVICE_AVAILABLE,
                "apollo_available": APOLLO_SERVICE_AVAILABLE,
                "web_scraping_available": WEB_SCRAPING_AVAILABLE
            },
            "analysis_completed_at": datetime.now().isoformat()
        }
        
        logger.info("✅ Comprehensive business card analysis completed")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


@app.post("/crawl-company")
async def crawl_company(
    company_name: str = Query(..., description="Company name to find"),
    use_ai_extraction: bool = Query(True, description="Kept for compatibility; ignored"),
    platform: str = Query("search", description="Deprecated. LLM search is always used.")
):
    """
    LLM-only company lookup (no crawl4ai):
    - Uses Gemini to return strict JSON with required fields.
    - Stores result in MongoDB.
    """
    try:
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

        # Save to MongoDB as before
        try:
            from mongodb_service import mongodb_service
            crawl_doc = {
                "company_name": company_name,
                "url": None,
                "platform": "gemini-search",
                "content": None,
                "ai_extracted_data": extracted,
                "crawl_time": elapsed,
                "crawled_at": datetime.now().isoformat()
            }
            saved = await mongodb_service.save_crawl_data(crawl_doc)
            response_data["saved_to_database"] = bool(saved)
            response_data["database_type"] = "MongoDB"
        except Exception as db_err:
            logger.warning("⚠️ MongoDB save failed", err=str(db_err))
            response_data["saved_to_database"] = False
            response_data["database_error"] = str(db_err)
            response_data["database_type"] = "MongoDB"

        logger.info("✅ LLM company lookup completed", company=company_name)
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ LLM company lookup failed", err=str(e))
        raise HTTPException(status_code=500, detail=f"Company lookup failed: {str(e)}")


@app.post("/crawl-website")
async def crawl_website(
    url: str = Query(..., description="Full URL of the website to crawl (include http/https)"),
    use_ai_extraction: bool = Query(True, description="Whether to use AI for data extraction"),
    extraction_profile: str = Query("generic", description="Extraction profile: generic | company | contact")
):
    """
    Crawl any website URL using crawl4ai and optionally run AI-powered data extraction.

    Returns:
    - success, url, crawl_time, content_length
    - markdown content length and optional AI-extracted structured data
    - saves crawl to MongoDB if service is available
    """
    try:
        if not url or not isinstance(url, str):
            raise HTTPException(status_code=400, detail="Valid URL is required")

        # Basic normalization
        target_url = url.strip()
        if not target_url.startswith(("http://", "https://")):
            target_url = f"https://{target_url}"

        logger.info("🔍 Starting website crawl", url=target_url, ai_extraction=use_ai_extraction, profile=extraction_profile)

        # Rate limiting
        await rate_limiter.wait_if_needed()

        start_time = time.time()

        # Ensure crawler availability
        if not CRAWL4AI_AVAILABLE:
            raise HTTPException(status_code=503, detail="Crawl4AI not available. Install with: pip install crawl4ai")

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=target_url)

        elapsed = time.time() - start_time
        logger.info("⏱️ Website crawling completed", elapsed=f"{elapsed:.1f}s")

        if not getattr(result, "success", False):
            logger.error("❌ Crawl unsuccessful", url=target_url)
            return {
                "success": False,
                "error": "Failed to crawl the website",
                "url": target_url,
                "crawl_time": elapsed
            }

        markdown_content = getattr(result, "markdown", "") or ""
        logger.info("📄 Extracted content", characters=len(markdown_content))

        # Optional AI extraction
        extracted_data = None
        if use_ai_extraction:
            try:
                gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
                if not gemini_key:
                    logger.warning("⚠️ Gemini API key not found, skipping AI extraction")
                    extracted_data = {"error": "Gemini API key not configured"}
                else:
                    client = genai.Client(api_key=gemini_key)

                    # Define profiles
                    if extraction_profile.lower() == "company":
                        instruction = """Extract company-focused information from the page and return ONLY valid JSON:
                        {
                          "Company name": "string",
                          "Description/tagline": "string",
                          "Products/services": "string",
                          "Location/headquarters": "string",
                          "Industry": "string",
                          "Number of employees": "string"
                        }
                        Use "N/A" when unknown. Be concise.
                        """
                    elif extraction_profile.lower() == "contact":
                        instruction = """Extract contact information and return ONLY valid JSON:
                        {
                          "Emails": ["string"],
                          "Phones": ["string"],
                          "Addresses": ["string"],
                          "Social links": ["string"],
                          "Contact page URLs": ["string"]
                        }
                        Return arrays (possibly empty). No extra text.
                        """
                    else:  # generic
                        instruction = """Summarize page structure and contact details. Return ONLY valid JSON:
                        {
                          "Title": "string",
                          "Description": "string",
                          "Headings": ["string"],
                          "Main topics": ["string"],
                          "Emails": ["string"],
                          "Phones": ["string"],
                          "Social links": ["string"]
                        }
                        Use concise values; arrays may be empty. No extra text.
                        """

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"{instruction}\n\n{markdown_content}"
                    )

                    ai_response = (response.text or "").strip()

                    # Clean possible fenced code blocks
                    cleaned = ai_response
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
                        extracted_data = {
                            "structured_data": parsed,
                            "raw_ai_response": ai_response,
                            "extraction_successful": True,
                            "parsing_successful": True
                        }
                    except json.JSONDecodeError as je:
                        logger.warning("⚠️ Failed to parse AI response as JSON", err=str(je))
                        extracted_data = {
                            "structured_data": None,
                            "raw_ai_response": ai_response,
                            "extraction_successful": True,
                            "parsing_successful": False,
                            "parsing_error": str(je)
                        }
            except Exception as ai_err:
                logger.error("❌ AI extraction failed", err=str(ai_err))
                extracted_data = {
                    "error": str(ai_err),
                    "extraction_successful": False,
                    "parsing_successful": False
                }

        response_data = {
            "success": True,
            "url": target_url,
            "crawl_time": elapsed,
            "content_length": len(markdown_content),
            "content_preview": markdown_content[:5000],  # limit preview size
            "extraction_profile": extraction_profile,
            "extracted_data": extracted_data if use_ai_extraction else None,
            "crawled_at": datetime.now().isoformat()
        }

        # Save to MongoDB if available
        try:
            if MONGODB_SERVICE_AVAILABLE:
                from mongodb_service import mongodb_service
                crawl_doc = {
                    "url": target_url,
                    "content": markdown_content,
                    "ai_extracted_data": extracted_data,
                    "extraction_profile": extraction_profile,
                    "crawl_time": elapsed,
                    "crawled_at": datetime.now().isoformat()
                }
                saved = await mongodb_service.save_crawl_data(crawl_doc)
                response_data["saved_to_database"] = bool(saved)
                response_data["database_type"] = "MongoDB"
            else:
                response_data["saved_to_database"] = False
                response_data["database_type"] = "MongoDB"
        except Exception as db_err:
            logger.warning("⚠️ MongoDB save failed", err=str(db_err))
            response_data["saved_to_database"] = False
            response_data["database_error"] = str(db_err)
            response_data["database_type"] = "MongoDB"

        logger.info("✅ Website crawl completed successfully", url=target_url)
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Website crawl failed", err=str(e))
        raise HTTPException(status_code=500, detail=f"Website crawl failed: {str(e)}")

@app.post("/crawl-multiple-companies")
async def crawl_multiple_companies(
    company_names: List[str] = Query(..., description="List of company names to crawl"),
    use_ai_extraction: bool = Query(True, description="Whether to use AI for data extraction"),
    platform: str = Query("linkedin", description="Platform to crawl")
):
    """
    Crawl multiple companies with rate limiting and batch processing
    """
    try:
        if not CRAWL4AI_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Crawl4AI not available. Install with: pip install crawl4ai google-generativeai"
            )
        
        if len(company_names) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 companies allowed per batch")
        
        logger.info(f"🚀 Starting batch crawl for {len(company_names)} companies")
        
        results = []
        
        for i, company_name in enumerate(company_names):
            try:
                logger.info(f"📋 Processing company {i+1}/{len(company_names)}: {company_name}")
                
                # Use the single company crawl logic
                result = await crawl_company(company_name, use_ai_extraction, platform)
                results.append(result)
                
                # Add delay between companies to respect rate limits
                if i < len(company_names) - 1:  # Don't delay after the last company
                    logger.info("⏸️ Waiting between companies to respect rate limits...")
                    await asyncio.sleep(3)  # 3 second delay between companies
                    
            except Exception as e:
                logger.error(f"❌ Failed to crawl {company_name}: {e}")
                results.append({
                    "success": False,
                    "company_name": company_name,
                    "error": str(e)
                })
        
        successful_crawls = sum(1 for r in results if r.get("success", False))
        
        logger.info(f"✅ Batch crawl completed: {successful_crawls}/{len(company_names)} successful")
        
        return {
            "success": True,
            "total_companies": len(company_names),
            "successful_crawls": successful_crawls,
            "results": results,
            "batch_completed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch crawl failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch crawl failed: {str(e)}")

# @app.post("/scrape-website-details")


@app.get("/api/v1/crawl-data")
async def get_crawl_data(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of records to return (max 100)"),
    company_name: str = Query(None, description="Filter by company name"),
    sort_by: str = Query("crawled_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order: asc or desc")
):
    """
    Fetch crawl data records from MongoDB visitor_intelligence.crawl_data collection
    
    Query Parameters:
    - skip: Number of records to skip for pagination (default: 0)
    - limit: Number of records to return (default: 10, max: 100)
    - company_name: Optional filter by company name (partial match)
    - sort_by: Field to sort by (default: crawled_at)
    - sort_order: Sort order - asc or desc (default: desc)
    """
    try:
        if not MONGODB_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service not available. Please configure MONGODB_URL environment variable."
            )
        
        logger.info("📊 Fetching crawl data from MongoDB",
                   skip=skip,
                   limit=limit,
                   company_name=company_name,
                   sort_by=sort_by,
                   sort_order=sort_order)
        
        # Build filter
        filter_query = {}
        if company_name:
            filter_query["company_name"] = {"$regex": company_name, "$options": "i"}  # Case-insensitive search
        
        # Build sort
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        
        # Access MongoDB collection directly
        collection = mongodb_service.db["crawl_data"]
        
        # Execute query
        cursor = collection.find(filter_query).sort(sort_by, sort_direction).skip(skip).limit(limit)
        
        # Convert to list and handle ObjectId
        records = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            records.append(doc)
        
        # Get total count for pagination
        total_count = await collection.count_documents(filter_query)
        
        logger.info("✅ Crawl data fetched successfully",
                   records_count=len(records),
                   total_count=total_count)
        
        return {
            "success": True,
            "data": records,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total_count,
                "has_more": (skip + limit) < total_count
            },
            "filters": {
                "company_name": company_name,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to fetch crawl data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch crawl data: {str(e)}"
        )


@app.get("/api/v1/crawl-data/{record_id}")
async def get_crawl_data_by_id(record_id: str):
    """
    Fetch a single crawl data record by ID from MongoDB
    """
    try:
        if not MONGODB_SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service not available. Please configure MONGODB_URL environment variable."
            )
        
        from bson import ObjectId
        
        logger.info("🔍 Fetching crawl data by ID", record_id=record_id)
        
        # Access MongoDB collection directly
        collection = mongodb_service.db["crawl_data"]
        
        # Fetch single record
        doc = await collection.find_one({"_id": ObjectId(record_id)})
        
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Crawl data record not found: {record_id}"
            )
        
        # Convert ObjectId to string
        doc["_id"] = str(doc["_id"])
        
        logger.info("✅ Crawl data record fetched", record_id=record_id)
        
        return {
            "success": True,
            "data": doc
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to fetch crawl data by ID", record_id=record_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch crawl data: {str(e)}"
        )


# WhatsApp webhook endpoint
@app.post("/api/v1/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(None),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None)
):
    """Handle incoming WhatsApp messages via Twilio"""
    try:
        # Import WhatsApp service
        from app.services.twilio_whatsapp_service import twilio_whatsapp_service
        
        # Extract phone number (remove whatsapp: prefix)
        sender_number = From.replace("whatsapp:", "")
        
        logger.info("📱 WhatsApp message received", 
                   from_number=sender_number, 
                   has_media=NumMedia != "0",
                   body=Body,
                   num_media=NumMedia,
                   body_is_none=Body is None)
        
        # Handle text messages
        if Body and NumMedia == "0":
            body_lower = Body.lower().strip()
            
            # Welcome messages
            if body_lower in ["hi", "hello", "start"]:
                welcome_message = (
                    "👋 *Welcome to Business Card Scanner!*\n\n"
                    "📸 Send me a photo of a business card and I'll extract all the information for you!\n\n"
                    "✨ *Features:*\n"
                    "• Extract contact details (name, title, company)\n"
                    "• Scan phone numbers and emails\n"
                    "• Detect QR codes and websites\n"
                    "• Company data enrichment\n"
                    "• Industry analysis\n\n"
                    "Just send a clear photo of any business card to get started! 🚀"
                )
                twiml_response = twilio_whatsapp_service.create_webhook_response(welcome_message)
                return Response(content=twiml_response, media_type="application/xml")
            
            # Help message
            elif body_lower in ["help", "commands", "?"]:
                help_message = (
                    "📸 *How to use Business Card Scanner:*\n\n"
                    "1. Take a clear photo of a business card\n"
                    "2. Send the photo to this chat\n"
                    "3. Get instant analysis with:\n"
                    "   • Contact information\n"
                    "   • QR code data\n"
                    "   • Company enrichment\n"
                    "   • Industry details\n\n"
                    "💡 *Tips for best results:*\n"
                    "• Ensure good lighting\n"
                    "• Keep the card flat\n"
                    "• Avoid shadows and glare\n"
                    "• Make sure text is readable"
                )
                twiml_response = twilio_whatsapp_service.create_webhook_response(help_message)
                return Response(content=twiml_response, media_type="application/xml")
            
            # Default text response
            else:
                help_message = (
                    "📸 Please send a photo of a business card to get started!\n\n"
                    "I can extract:\n"
                    "• Names, titles, companies\n"
                    "• Phone numbers and emails\n"
                    "• QR codes and websites\n"
                    "• Company enrichment data\n\n"
                    "Type 'help' for more information."
                )
                twiml_response = twilio_whatsapp_service.create_webhook_response(help_message)
                return Response(content=twiml_response, media_type="application/xml")
        
        # Handle image messages
        if NumMedia and NumMedia != "0" and int(NumMedia) > 0 and MediaUrl0:
            try:
                # Validate content type
                if not MediaContentType0 or not MediaContentType0.startswith('image/'):
                    twiml_response = twilio_whatsapp_service.create_webhook_response(
                        "❌ Please send a valid image file (JPG, PNG, etc.)"
                    )
                    return Response(content=twiml_response, media_type="application/xml")
                
                logger.info("📷 Processing business card image", 
                           media_url=MediaUrl0, 
                           content_type=MediaContentType0)
                
                # Download image from Twilio
                image_data = await twilio_whatsapp_service.download_media(MediaUrl0)
                if not image_data:
                    twiml_response = twilio_whatsapp_service.create_webhook_response(
                        "❌ Failed to download image. Please try again."
                    )
                    return Response(content=twiml_response, media_type="application/xml")
                
                logger.info("📥 Image downloaded", size=len(image_data))
                
                # Process with OCR service
                result = await ocr_service.extract_business_card_data(image_data, use_vision=True)
                
                if not result["success"]:
                    error_message = (
                        "❌ Sorry, I couldn't process the image.\n\n"
                        "💡 *Tips for better results:*\n"
                        "• Ensure good lighting\n"
                        "• Keep the card flat and straight\n"
                        "• Avoid shadows and glare\n"
                        "• Make sure all text is visible\n\n"
                        "Please try with a clearer photo!"
                    )
                    twiml_response = twilio_whatsapp_service.create_webhook_response(error_message)
                    return Response(content=twiml_response, media_type="application/xml")
                
                logger.info("✅ OCR processing completed", 
                           confidence=result.get("confidence", 0),
                           qr_count=result.get("qr_count", 0))
                
                # Get company enrichment if company found
                company_name = result.get("data", {}).get("company")
                if company_name and APOLLO_SERVICE_AVAILABLE:
                    try:
                        logger.info("🏢 Starting company enrichment", company=company_name)
                        enrichment = await apollo_service.search_company(company_name)
                        result["enrichment"] = enrichment
                        logger.info("✅ Company enrichment completed", company=company_name)
                    except Exception as e:
                        logger.warning("⚠️ Company enrichment failed", company=company_name, error=str(e))
                elif company_name and not APOLLO_SERVICE_AVAILABLE:
                    logger.info("⚠️ Apollo.io not available, skipping enrichment")
                
                # Send formatted response via WhatsApp API (async)
                try:
                    await twilio_whatsapp_service.send_business_card_analysis(sender_number, result)
                    logger.info("📤 Business card analysis sent", to=sender_number)
                except Exception as e:
                    logger.error("❌ Failed to send analysis", error=str(e))
                    twiml_response = twilio_whatsapp_service.create_webhook_response(
                        "❌ Analysis completed but failed to send results. Please try again."
                    )
                    return Response(content=twiml_response, media_type="application/xml")
                
                # Return empty response (we sent message via API)
                twiml_response = twilio_whatsapp_service.create_webhook_response("")
                return Response(content=twiml_response, media_type="application/xml")
                
            except Exception as e:
                logger.error("❌ Error processing image", error=str(e))
                twiml_response = twilio_whatsapp_service.create_webhook_response(
                    "❌ Error processing image. Please try again with a different photo."
                )
                return Response(content=twiml_response, media_type="application/xml")
        
        # Default response
        twiml_response = twilio_whatsapp_service.create_webhook_response(
            "📸 Please send a photo of a business card to get started!"
        )
        return Response(content=twiml_response, media_type="application/xml")
        
    except Exception as e:
        logger.error("❌ WhatsApp webhook error", error=str(e))
        twiml_response = twilio_whatsapp_service.create_webhook_response(
            "❌ An error occurred. Please try again."
        )
        return Response(content=twiml_response, media_type="application/xml")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple OCR Server...")
    print("📱 Frontend: http://localhost:5173")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("📱 WhatsApp Webhook: http://localhost:8000/api/v1/whatsapp/webhook")
    uvicorn.run(app, host="0.0.0.0", port=8000)

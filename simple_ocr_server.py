#!/usr/bin/env python3
"""
Simple OCR Server - Clean and reliable OCR implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Form, Response
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import structlog
import cv2
import numpy as np
from PIL import Image
import io
import pytesseract
from datetime import datetime
import time
from collections import deque
import asyncio

# Import crawl4ai and genai for company crawling
try:
    from crawl4ai import AsyncWebCrawler
    from google import genai
    CRAWL4AI_AVAILABLE = True
    print("✅ Crawl4AI and Gemini available for company crawling")
except ImportError as e:
    CRAWL4AI_AVAILABLE = False
    print(f"⚠️ Crawl4AI not available: {e}")
    print("Install with: pip install crawl4ai google-generativeai")

# QR code detection will use OpenCV only

# Import the new OCR service
from app.services.ocr_service import OCRService

# Import new modular services
from app.services.qr_service import QRService
from app.services.business_card_service import BusinessCardService
from app.services.image_processing_service import ImageProcessingService

# Import standalone business card analyzer
from standalone_business_card_analyzer import BusinessCardAnalyzer

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

# Initialize standalone AI analyzer
try:
    ai_analyzer = BusinessCardAnalyzer()
    AI_ANALYZER_AVAILABLE = True
    print("✅ AI Business Card Analyzer available")
except Exception as e:
    AI_ANALYZER_AVAILABLE = False
    print(f"⚠️ AI Business Card Analyzer not available: {e}")
    print("Make sure OPENAI_API_KEY is set in environment variables")

# Initialize rate limiter for crawl4ai
rate_limiter = SimpleRateLimiter()

def detect_qr_codes(image_data: bytes) -> list:
    """Detect QR codes in the image using multiple methods"""
    return qr_service.detect_qr_codes(image_data)


def parse_qr_content(qr_data: str) -> dict:
    """Parse QR code content and extract structured information"""
    return qr_service.parse_qr_content(qr_data)


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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple OCR Server",
        "status": "running",
        "tesseract_version": pytesseract.get_tesseract_version(),
        "ai_analyzer_available": AI_ANALYZER_AVAILABLE,
        "endpoints": {
            "ocr": "/ocr",
            "business_card": "/business-card",
            "ai_business_card": "/ai-business-card",
            "qr_scan": "/qr-scan",
            "qr_scan_goqr": "/qr-scan-goqr",
            "batch_ocr": "/batch-ocr",
            "extract_url_content": "/extract-url-content",
            "comprehensive_business_card_analysis": "/comprehensive-business-card-analysis",
            "crawl_company": "/crawl-company",
            "crawl_multiple_companies": "/crawl-multiple-companies",
            "whatsapp_webhook": "/api/v1/whatsapp/webhook"
        }
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
        "tesseract_available": True,
        "tesseract_version": pytesseract.get_tesseract_version()
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
async def process_business_card(
    file: UploadFile = File(...), 
    use_ai: bool = Query(False, description="Use AI Vision analysis instead of traditional OCR")
):
    """Process business card with OCR + Vision analysis or AI Vision analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        logger.info("🚀 Processing business card", 
                   filename=file.filename, 
                   file_size=len(content),
                   use_ai=use_ai)
        
        # Choose processing method
        if use_ai and AI_ANALYZER_AVAILABLE:
            # Use AI Vision analysis
            logger.info("🤖 Using AI Vision analysis")
            ai_result = ai_analyzer.analyze_business_card_from_bytes(content)
            
            if ai_result["success"]:
                structured_info = ai_result.get("structured_info", {})
                contact_info = structured_info.get("contact_info", {})
                qr_codes = structured_info.get("qr_codes", [])
                
                result = {
                    "success": True,
                    "data": contact_info,
                    "confidence": structured_info.get("confidence", 0.0),
                    "raw_text": ai_result.get("raw_analysis", ""),
                    "engine_used": "ai_vision",
                    "qr_codes": qr_codes,
                    "qr_count": len(qr_codes),
                    "vision_analysis": structured_info,
                    "vision_available": True
                }
            else:
                result = {
                    "success": False,
                    "error": ai_result.get("error", "AI analysis failed"),
                    "data": {},
                    "confidence": 0.0,
                    "raw_text": "",
                    "engine_used": "ai_vision",
                    "qr_codes": [],
                    "qr_count": 0,
                    "vision_analysis": {},
                    "vision_available": False
                }
        else:
            # Use traditional OCR + Vision
            if use_ai and not AI_ANALYZER_AVAILABLE:
                logger.warning("⚠️ AI analyzer not available, falling back to traditional OCR")
            
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
        
        # Save to database if available
        if result["success"] and MONGODB_SERVICE_AVAILABLE:
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
                
                # Save to MongoDB
                saved_card = await mongodb_service.save_business_card_data(card_data)
                if saved_card:
                    logger.info(f"✅ AI business card data saved to MongoDB: {contact_info.get('name', 'Unknown')}")
                else:
                    logger.warning(f"⚠️ Failed to save AI business card data to MongoDB")
                    
            except Exception as db_error:
                logger.warning(f"⚠️ MongoDB save failed: {db_error}")
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
                
                # Extract structured information if OCR was successful
                structured_info = None
                if result.get("success") and result.get("text"):
                    # Use AI-powered extraction instead of regex
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
    company_name: str = Query(..., description="Company name to crawl from LinkedIn"),
    use_ai_extraction: bool = Query(True, description="Whether to use AI for data extraction"),
    platform: str = Query("linkedin", description="Platform to crawl (linkedin, website, etc.)")
):
    """
    Crawl company information using crawl4ai with AI-powered data extraction
    
    This endpoint:
    1. Crawls LinkedIn company pages using crawl4ai
    2. Extracts structured company data using Gemini AI
    3. Respects rate limits to avoid being blocked
    """
    try:
        if not CRAWL4AI_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Crawl4AI not available. Install with: pip install crawl4ai google-generativeai"
            )
        
        # Validate inputs
        if not company_name or not company_name.strip():
            raise HTTPException(status_code=400, detail="Company name is required")
        
        # Clean company name for URL generation
        company_name_clean = company_name.strip().lower()
        
        logger.info(f"🔍 Starting company crawl", 
                   company=company_name, 
                   platform=platform,
                   ai_extraction=use_ai_extraction)
        
        # Apply rate limiting
        await rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        # Construct URL based on platform
        if platform.lower() == "linkedin":
            # Handle company names with spaces for LinkedIn URLs
            if " " in company_name_clean:
                # Try both formats: with hyphens and without spaces
                company_name_hyphen = company_name_clean.replace(" ", "-")
                company_name_no_spaces = company_name_clean.replace(" ", "")
                
                # Try hyphenated version first (more common on LinkedIn)
                url = f"https://www.linkedin.com/company/{company_name_hyphen}/"
                logger.info(f"🌐 Using hyphenated LinkedIn URL: {url}")
            else:
                url = f"https://www.linkedin.com/company/{company_name_clean}/"
            
        elif platform.lower() == "website":
            # Try common website patterns
            url = f"https://www.{company_name}.com"
        else:
            url = company_name  # Assume it's already a full URL
        
        logger.info(f"🌐 Crawling URL: {url}")
        
        # Crawl the website
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Crawling completed in {elapsed:.1f} seconds")
            
            if not result.success:
                logger.error(f"❌ Crawl unsuccessful for {url}")
                return {
                    "success": False,
                    "error": "Failed to crawl the website",
                    "url": url,
                    "company_name": company_name,
                    "crawl_time": elapsed
                }
            
            # Extract content
            markdown_content = result.markdown
            logger.info(f"📄 Extracted {len(markdown_content)} characters of content")
            
            # AI-powered data extraction
            extracted_data = None
            if use_ai_extraction:
                try:
                    logger.info("🤖 Starting AI-powered data extraction...")
                    
                    # Check if Gemini API key is available
                    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
                    if not gemini_key:
                        logger.warning("⚠️ Gemini API key not found, skipping AI extraction")
                        extracted_data = {"error": "Gemini API key not configured"}
                    else:
                        # Use exact same format as crawling.py
                        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                        
                        # Enhanced strategy for structured JSON output
                        strategy = {
                            "instruction": """Extract the following company information and return as a valid JSON object with these exact keys:
                            {
                                "Company name": "string",
                                "Description/tagline": "string", 
                                "Products/services": "string",
                                "Location/headquarters": "string",
                                "Industry": "string",
                                "Number of employees": "string"
                            }
                            
                            Rules:
                            - Return ONLY valid JSON, no additional text
                            - Use "N/A" if information is not available
                            - For products/services, provide a brief summary
                            - For number of employees, use ranges like "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10000+"
                            - Be concise but informative"""
                        }
                        
                        # Use exact same format as crawling.py
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=f"{strategy} {markdown_content}"
                        )
                        
                        # Parse the AI response into structured JSON
                        ai_response = response.text.strip()
                        
                        try:
                            # Try to parse as JSON
                            import json
                            import re
                            
                            # Clean the response - remove markdown code blocks if present
                            cleaned_response = ai_response
                            if "```json" in cleaned_response:
                                # Extract JSON from markdown code blocks
                                json_match = re.search(r'```json\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                                if json_match:
                                    cleaned_response = json_match.group(1).strip()
                            elif "```" in cleaned_response:
                                # Handle generic code blocks
                                json_match = re.search(r'```\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
                                if json_match:
                                    cleaned_response = json_match.group(1).strip()
                            
                            parsed_data = json.loads(cleaned_response)
                            
                            # Validate and format the response
                            formatted_data = {
                                "Company name": parsed_data.get("Company name", "N/A"),
                                "Description/tagline": parsed_data.get("Description/tagline", "N/A"),
                                "Products/services": parsed_data.get("Products/services", "N/A"),
                                "Location/headquarters": parsed_data.get("Location/headquarters", "N/A"),
                                "Industry": parsed_data.get("Industry", "N/A"),
                                "Number of employees": parsed_data.get("Number of employees", "N/A")
                            }
                            
                            extracted_data = {
                                "structured_data": formatted_data,
                                "raw_ai_response": ai_response,
                                "extraction_successful": True,
                                "parsing_successful": True
                            }
                            
                            logger.info("✅ AI extraction and JSON parsing completed")
                            
                        except json.JSONDecodeError as json_error:
                            logger.warning(f"⚠️ Failed to parse AI response as JSON: {json_error}")
                            # Fallback: return raw response with parsing error
                            extracted_data = {
                                "structured_data": None,
                                "raw_ai_response": ai_response,
                                "extraction_successful": True,
                                "parsing_successful": False,
                                "parsing_error": str(json_error)
                            }
                        
                except Exception as ai_error:
                    logger.error(f"❌ AI extraction failed: {ai_error}")
                    extracted_data = {
                        "error": str(ai_error),
                        "extraction_successful": False,
                        "parsing_successful": False
                    }
            
            # Prepare response with structured company data
            response_data = {
                "success": True,
                "company_name": company_name,
                "url": url,
                "platform": platform,
                "crawl_time": elapsed,
                "content_length": len(markdown_content),
                "company_data": extracted_data.get("structured_data") if extracted_data else None,  # This will be your formatted JSON
                "raw_ai_response": extracted_data.get("raw_ai_response") if extracted_data else None,
                "extraction_successful": extracted_data.get("extraction_successful", False) if extracted_data else False,
                "parsing_successful": extracted_data.get("parsing_successful", False) if extracted_data else False,
                "crawled_at": datetime.now().isoformat()
            }
            
            # Save to MongoDB if available
            try:
                from mongodb_service import mongodb_service
                
                # Prepare data for saving
                crawl_data = {
                    "company_name": company_name,
                    "url": url,
                    "platform": platform,
                    "content": markdown_content,
                    "ai_extracted_data": extracted_data,
                    "crawl_time": elapsed,
                    "crawled_at": datetime.now().isoformat()
                }
                
                # Save crawl data to MongoDB
                saved_crawl = await mongodb_service.save_crawl_data(crawl_data)
                if saved_crawl:
                    logger.info(f"✅ Crawl data saved to MongoDB: {company_name}")
                    response_data["saved_to_database"] = True
                    response_data["database_type"] = "MongoDB"
                else:
                    logger.warning(f"⚠️ Failed to save crawl data to MongoDB")
                    response_data["saved_to_database"] = False
                    response_data["database_type"] = "MongoDB"
                    
            except Exception as db_error:
                logger.warning(f"⚠️ MongoDB save failed: {db_error}")
                response_data["saved_to_database"] = False
                response_data["database_error"] = str(db_error)
                response_data["database_type"] = "MongoDB"
            
            logger.info(f"✅ Company crawl completed successfully: {company_name}")
            return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Company crawl failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Company crawl failed: {str(e)}")


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

"""
OCR processing endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Dict, Any, Optional
import structlog

from app.services.ocr_service import OCRService

router = APIRouter()
logger = structlog.get_logger(__name__)

# Initialize OCR service
ocr_service = OCRService()


@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    engine: str = Query("auto", description="OCR engine to use: auto, tesseract, easyocr, paddleocr")
):
    """Process image with OCR"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Process with OCR
        result = await ocr_service.process_image(content, engine=engine)
        
        logger.info("OCR processing completed", 
                   filename=file.filename, 
                   engine=engine, 
                   success=result["success"])
        
        return {
            "success": result["success"],
            "filename": file.filename,
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "engine_used": result.get("engine", engine),
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error("OCR processing failed", filename=file.filename, err=str(e))
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@router.post("/business-card")
async def process_business_card(
    file: UploadFile = File(...),
    engine: str = Query("auto", description="OCR engine to use: auto, tesseract, easyocr, paddleocr")
):
    """Process business card image and extract structured data"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Process business card
        result = await ocr_service.extract_business_card_data(content)
        
        logger.info("Business card processing completed", 
                   filename=file.filename, 
                   success=result["success"])
        
        return {
            "success": result["success"],
            "filename": file.filename,
            "data": result.get("data", {}),
            "confidence": result.get("confidence", 0.0),
            "raw_text": result.get("raw_text", ""),
            "engine_used": result.get("engine_used", engine),
            "qr_codes": result.get("qr_codes", []),
            "qr_count": result.get("qr_count", 0),
            "qr_parsed_data": result.get("qr_parsed_data", {}),
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error("Business card processing failed", filename=file.filename, err=str(e))
        raise HTTPException(status_code=500, detail=f"Business card processing failed: {str(e)}")


@router.post("/qr-scan")
async def scan_qr_codes(
    file: UploadFile = File(...),
    fetch_url_details: bool = Query(False, description="Fetch additional details from QR code URLs")
):
    """Scan for QR codes in the image and optionally fetch URL details"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Scan for QR codes
        result = await ocr_service.scan_qr_codes(content)
        
        # Fetch URL details if requested and URLs found
        url_details = {}
        if fetch_url_details and result.get("success"):
            urls_to_fetch = []
            for qr in result.get("qr_codes", []):
                if qr.get("parsed", {}).get("type") == "url":
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
        
        logger.info("QR code scanning completed", 
                   filename=file.filename, 
                   success=result["success"],
                   qr_count=result.get("count", 0),
                   url_details_fetched=bool(url_details))
        
        return {
            "success": result["success"],
            "filename": file.filename,
            "qr_codes": result.get("qr_codes", []),
            "parsed_data": result.get("parsed_data", {}),
            "count": result.get("count", 0),
            "url_details": url_details if fetch_url_details else {},
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error("QR code scanning failed", filename=file.filename, err=str(e))
        raise HTTPException(status_code=500, detail=f"QR code scanning failed: {str(e)}")


@router.get("/engines")
async def get_available_engines():
    """Get list of available OCR engines"""
    try:
        engines = list(ocr_service.engines.keys())
        return {
            "available_engines": engines,
            "default_engine": "auto",
            "total_engines": len(engines)
        }
    except Exception as e:
        logger.error("Failed to get OCR engines", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get OCR engines: {str(e)}")

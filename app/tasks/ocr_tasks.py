"""
OCR background tasks
"""

from app.core.celery_app import celery_app
from app.services.ocr_service import OCRService
import structlog

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True)
def process_image_ocr(self, image_data: bytes, visitor_id: int):
    """Process image with OCR in background"""
    try:
        logger.info("Starting OCR processing", visitor_id=visitor_id)
        
        ocr_service = OCRService()
        result = ocr_service.process_image(image_data)
        
        logger.info("OCR processing completed", visitor_id=visitor_id, success=result["success"])
        return result
        
    except Exception as e:
        logger.error("OCR processing failed", visitor_id=visitor_id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True)
def extract_business_card_data(self, image_data: bytes, visitor_id: int):
    """Extract business card data in background"""
    try:
        logger.info("Starting business card extraction", visitor_id=visitor_id)
        
        ocr_service = OCRService()
        result = ocr_service.extract_business_card_data(image_data)
        
        logger.info("Business card extraction completed", visitor_id=visitor_id, success=result["success"])
        return result
        
    except Exception as e:
        logger.error("Business card extraction failed", visitor_id=visitor_id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

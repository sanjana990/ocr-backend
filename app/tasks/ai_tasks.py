"""
AI background tasks
"""

from app.core.celery_app import celery_app
from app.services.ai_service import AIService
import structlog

logger = structlog.get_logger(__name__)


@celery_app.task(bind=True)
def generate_profile_summary(self, visitor_data: dict, visitor_id: int):
    """Generate AI profile summary in background"""
    try:
        logger.info("Starting AI summarization", visitor_id=visitor_id)
        
        ai_service = AIService()
        result = ai_service.generate_profile_summary(visitor_data)
        
        logger.info("AI summarization completed", visitor_id=visitor_id, success=result["success"])
        return result
        
    except Exception as e:
        logger.error("AI summarization failed", visitor_id=visitor_id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True)
def generate_engagement_suggestions(self, visitor_data: dict, visitor_id: int):
    """Generate engagement suggestions in background"""
    try:
        logger.info("Starting engagement suggestions", visitor_id=visitor_id)
        
        ai_service = AIService()
        result = ai_service.generate_engagement_suggestions(visitor_data)
        
        logger.info("Engagement suggestions completed", visitor_id=visitor_id, success=result["success"])
        return result
        
    except Exception as e:
        logger.error("Engagement suggestions failed", visitor_id=visitor_id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

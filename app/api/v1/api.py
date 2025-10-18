"""
API v1 router configuration
"""

from fastapi import APIRouter
from app.api.v1.endpoints import visitors, companies, ocr, ai, research

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(visitors.router, prefix="/visitors", tags=["visitors"])
api_router.include_router(companies.router, prefix="/companies", tags=["companies"])
api_router.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
api_router.include_router(research.router, prefix="/research", tags=["research"])

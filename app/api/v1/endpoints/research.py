"""
Research and enrichment endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import structlog

from app.services.research_service import ResearchService

router = APIRouter()
logger = structlog.get_logger(__name__)

# Initialize research service
research_service = ResearchService()


@router.post("/enrich")
async def enrich_company_data(
    company_name: str = Query(..., description="Company name to enrich"),
    domain: Optional[str] = Query(None, description="Company domain (optional)")
):
    """Enrich company data with external research"""
    try:
        logger.info("Starting company enrichment", company=company_name)
        
        # Enrich company data
        result = await research_service.enrich_company_data(company_name, domain)
        
        logger.info("Company enrichment completed", 
                   company=company_name, 
                   success=result["success"])
        
        return result
        
    except Exception as e:
        logger.error("Company enrichment failed", company=company_name, err=str(e))
        raise HTTPException(status_code=500, detail=f"Company enrichment failed: {str(e)}")


@router.post("/social-discovery")
async def discover_social_profiles(
    company_name: str = Query(..., description="Company name to search for"),
    domain: Optional[str] = Query(None, description="Company domain (optional)")
):
    """Discover social media profiles for a company"""
    try:
        logger.info("Starting social discovery", company=company_name)
        
        # Enrich company data (includes social profiles)
        result = await research_service.enrich_company_data(company_name, domain)
        
        if result["success"]:
            social_profiles = result["data"].get("social_profiles", {})
            return {
                "success": True,
                "company_name": company_name,
                "social_profiles": social_profiles,
                "total_profiles": len(social_profiles)
            }
        else:
            return result
            
    except Exception as e:
        logger.error("Social discovery failed", company=company_name, err=str(e))
        raise HTTPException(status_code=500, detail=f"Social discovery failed: {str(e)}")


@router.get("/status")
async def get_research_status():
    """Get research service status"""
    try:
        return {
            "status": "active",
            "available_services": [
                "company_enrichment",
                "social_discovery",
                "news_search",
                "website_analysis"
            ],
            "supported_platforms": [
                "LinkedIn",
                "Twitter",
                "Facebook",
                "Google News"
            ]
        }
    except Exception as e:
        logger.error("Failed to get research status", err=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get research status: {str(e)}")

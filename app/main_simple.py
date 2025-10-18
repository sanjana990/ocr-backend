"""
Visitor Intelligence Platform - Simple FastAPI Application (for testing)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Visitor Intelligence Platform (Simple Mode)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Visitor Intelligence Platform")


# Create FastAPI application
app = FastAPI(
    title="Visitor Intelligence Platform",
    version="1.0.0",
    description="AI-driven Visitor Intelligence Platform (Simple Mode)",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Visitor Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "mode": "simple"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": "development",
        "mode": "simple"
    }


@app.get("/api/v1/ocr/engines")
async def get_available_engines():
    """Get list of available OCR engines (mock)"""
    return {
        "available_engines": ["tesseract", "easyocr", "paddleocr"],
        "default_engine": "auto",
        "total_engines": 3,
        "note": "Running in simple mode - OCR engines not loaded"
    }


@app.get("/api/v1/research/status")
async def get_research_status():
    """Get research service status (mock)"""
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
        ],
        "note": "Running in simple mode - Research services not loaded"
    }


@app.post("/api/v1/visitors/")
async def create_visitor_mock():
    """Create visitor (mock)"""
    return {
        "success": True,
        "message": "Visitor creation not implemented yet",
        "visitor_id": "mock_123",
        "note": "Running in simple mode"
    }


@app.get("/api/v1/visitors/")
async def get_visitors_mock():
    """Get visitors (mock)"""
    return {
        "success": True,
        "visitors": [],
        "total": 0,
        "note": "Running in simple mode"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    logger.error("HTTP Exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception", err=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

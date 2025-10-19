#!/usr/bin/env python3
"""
Run script for Render deployment
"""

import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app
from simple_ocr_server import app

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

#!/bin/bash

# Visitor Intelligence Platform - Backend Installation Script

echo "🚀 Installing Visitor Intelligence Platform Backend..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    exit 1
fi

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Error: Python 3.11+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check if root virtual environment exists
if [ ! -d "../venv" ]; then
    echo "📦 Creating virtual environment..."
    cd ..
    python3 -m venv venv
    cd backend
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install system dependencies for OCR
echo "📚 Installing system dependencies for OCR..."

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS, installing dependencies with Homebrew..."
    
    # Install Tesseract
    if ! command -v tesseract &> /dev/null; then
        echo "Installing Tesseract..."
        brew install tesseract
    fi
    
    # Install system dependencies for OpenCV
    if ! command -v pkg-config &> /dev/null; then
        echo "Installing pkg-config..."
        brew install pkg-config
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux, installing dependencies with apt..."
    
    # Install Tesseract
    if ! command -v tesseract &> /dev/null; then
        echo "Installing Tesseract..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    fi
    
    # Install system dependencies for OpenCV
    sudo apt-get install -y libopencv-dev python3-opencv
    
else
    echo "⚠️  Unsupported OS: $OSTYPE"
    echo "Please install Tesseract and OpenCV manually"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads
mkdir -p logs

# Copy environment template
if [ ! -f ".env" ]; then
    echo "📋 Creating environment file..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your configuration"
else
    echo "✅ Environment file already exists"
fi

# Test installation
echo "🧪 Testing installation..."
python -c "
try:
    import fastapi
    import sqlalchemy
    import redis
    import celery
    import opencv-python
    import pillow
    import tesseract
    import easyocr
    import paddleocr
    import playwright
    import scrapy
    import beautifulsoup4
    print('✅ All dependencies installed successfully')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Backend installation completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Set up PostgreSQL database"
    echo "3. Run: python run.py"
    echo ""
    echo "🔧 Available commands:"
    echo "- python run.py          # Start FastAPI server"
    echo "- python run_celery.py   # Start Celery worker"
    echo "- playwright install     # Install browser dependencies"
    echo ""
    echo "📚 Documentation:"
    echo "- API Docs: http://localhost:8000/api/v1/docs"
    echo "- Health Check: http://localhost:8000/health"
else
    echo "❌ Installation failed. Please check the errors above."
    exit 1
fi

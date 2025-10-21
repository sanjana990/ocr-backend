# OCR Business Card Processing Platform - Backend

AI-driven business card processing platform with company data enrichment using Apollo.io.

## Features

- **FastAPI** - Modern, fast web framework
- **OCR Integration** - Text extraction from business cards (Tesseract, EasyOCR)
- **QR Code Detection** - Extract contact info from QR codes
- **Apollo.io Integration** - Company data enrichment (replaces LinkedIn scraping)
- **Company Research** - Automated data enrichment
- **Database Storage** - Supabase integration for data persistence

## Project Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/     # API endpoints
│   ├── core/                # Core configuration
│   ├── models/              # Database models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic services
├── requirements.txt         # Python dependencies
├── run.py                   # Development server
└── env.example             # Environment variables template
```

## Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source ../venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
# Set database URLs, API keys, etc.
```

### 3. Apollo.io Configuration

```bash
# Get your Apollo.io API key from https://app.apollo.io/
# Add to .env file:
APOLLO_API_KEY=your_apollo_api_key_here

# Optional: Scrapfly for LinkedIn fallback
SCRAPFLY_API_KEY=your_scrapfly_api_key_here
```

### 3. Database Setup

```bash
# Create database (PostgreSQL)
createdb visitor_intelligence

# Run migrations (when implemented)
alembic upgrade head
```

### 4. Start Services

```bash
# Terminal 1: Start FastAPI server
python run.py

```

### 5. Access API

- **API Documentation**: http://localhost:8000/api/v1/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

## API Endpoints

### OCR Processing
- `POST /ocr` - Process image with OCR
- `POST /business-card` - Extract business card data with AI analysis
- `POST /batch-ocr` - Process multiple images
- `POST /qr-scan` - Scan QR codes in images

### Company Data Enrichment (Apollo.io)
- `POST /search-company` - Search company information using Apollo.io
- `POST /search-companies` - Search multiple companies
- `POST /search-company-contacts` - Get contacts at a company

### URL Content Extraction
- `POST /extract-url-content` - Extract contact info from URLs

### Health & Debug
- `GET /health` - Health check
- `GET /debug-qr` - Debug QR code detection

## Development

### Code Quality

```bash
# Format code
black app/

# Sort imports
isort app/

# Lint code
flake8 app/

# Run tests
pytest
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## Production Deployment

### Railway Deployment

1. Connect your GitHub repository to Railway
2. Railway will automatically detect Python and install dependencies
3. Set environment variables in Railway dashboard
4. Deploy!

### Environment Variables

Required environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `OPENAI_API_KEY` - OpenAI API key
- `CELERY_BROKER_URL` - Celery broker URL
- `CELERY_RESULT_BACKEND` - Celery result backend

## Next Steps

1. **Step 2**: Implement database models and migrations
2. **Step 3**: Enhance OCR service with multiple engines
3. **Step 4**: Integrate AI services
4. **Step 5**: Build research automation
5. **Step 6**: Connect frontend to backend

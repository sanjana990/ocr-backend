# OCR Business Card Processing Platform - Backend

AI-driven business card processing platform with company data enrichment using Gemini AI.

## Features

- **FastAPI** - Modern, fast web framework
- **OCR Integration** - Text extraction from business cards (Tesseract, EasyOCR)
- **QR Code Detection** - Extract contact info from QR codes
- **Gemini AI Integration** - Company data enrichment using Google's Gemini AI
- **Company Research** - Automated data enrichment
- **Database Storage** - Supabase integration for data persistence
- **Image Storage** - Supabase Storage for efficient business card image handling

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

### 3. Gemini AI Configuration

```bash
# Get your Gemini API key from https://aistudio.google.com/app/apikey
# Add to .env file:
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Database Setup (Supabase)

```bash
# 1. Create Supabase project at https://supabase.com
# 2. Create required tables in Supabase SQL Editor:

-- Create scanned_cards table
CREATE TABLE scanned_cards (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT,
    phone TEXT,
    company TEXT,
    email TEXT,
    website TEXT,
    address TEXT,
    social_media JSONB,
    qr_codes JSONB,
    additional_info JSONB,
    image TEXT, -- Stores Supabase Storage URL
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create company_enrichment table
CREATE TABLE company_enrichment (
    id SERIAL PRIMARY KEY,
    company_name TEXT NOT NULL,
    description TEXT,
    products TEXT,
    location TEXT,
    industry TEXT,
    num_of_emp TEXT,
    revenue TEXT,
    market_share TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Disable RLS for easier API access
ALTER TABLE scanned_cards DISABLE ROW LEVEL SECURITY;
ALTER TABLE company_enrichment DISABLE ROW LEVEL SECURITY;

# 3. Create Storage bucket for business card images:
# - Go to Storage → Buckets → New bucket
# - Name: business_card_images
# - Make it Public for direct URL access
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

### Company Data Enrichment (Gemini AI)
- `POST /crawl-company` - Research company information using Gemini AI
- `POST /search-companies` - Search multiple companies
- `POST /search-company-contacts` - Get contacts at a company

### URL Content Extraction
- `POST /extract-url-content` - Extract contact info from URLs

### Health & Debug
- `GET /health` - Health check
- `GET /debug-qr` - Debug QR code detection

## Testing Storage Functionality

Test the new Supabase Storage implementation:

```bash
# Run the storage test script
python test_storage.py
```

This will:
1. Upload a test image to Supabase Storage
2. Save business card data with the image URL
3. Test image deletion functionality

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

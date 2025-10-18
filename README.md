# Visitor Intelligence Platform - Backend

AI-driven Visitor Intelligence Platform backend built with FastAPI, PostgreSQL, and Celery.

## Features

- **FastAPI** - Modern, fast web framework
- **PostgreSQL** - Primary database
- **Redis** - Caching and message broker
- **Celery** - Background task processing
- **OCR Integration** - Text extraction from images
- **AI Services** - LLM integration for insights
- **Research Automation** - Company data enrichment

## Project Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/     # API endpoints
│   ├── core/                # Core configuration
│   ├── models/              # Database models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic services
│   └── tasks/               # Celery background tasks
├── requirements.txt         # Python dependencies
├── run.py                   # Development server
├── run_celery.py           # Celery worker
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

# Terminal 2: Start Celery worker
python run_celery.py

# Terminal 3: Start Redis (if not running)
redis-server
```

### 5. Access API

- **API Documentation**: http://localhost:8000/api/v1/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

## API Endpoints

### Visitors
- `GET /api/v1/visitors/` - List visitors
- `POST /api/v1/visitors/` - Create visitor
- `GET /api/v1/visitors/{id}` - Get visitor
- `PUT /api/v1/visitors/{id}` - Update visitor
- `DELETE /api/v1/visitors/{id}` - Delete visitor

### OCR
- `POST /api/v1/ocr/process` - Process image with OCR
- `POST /api/v1/ocr/business-card` - Extract business card data

### AI
- `POST /api/v1/ai/summarize` - Generate profile summary
- `POST /api/v1/ai/suggestions` - Get engagement suggestions

### Research
- `POST /api/v1/research/enrich` - Enrich company data
- `POST /api/v1/research/social-discovery` - Discover social profiles

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

### Docker

```bash
# Build image
docker build -t visitor-intelligence-backend .

# Run container
docker run -p 8000:8000 visitor-intelligence-backend
```

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

# IntelliFone

IntelliFone is a smartphone marketplace and AI verification system for the Pakistani used-phone market. It combines a Next.js web app, a FastAPI AI backend, MongoDB-backed market datasets, Supabase storage/auth, YOLO damage detection, DeepSeek LLM extraction, OLX scraping, price prediction, and YouTube-based recommendations.

## What It Does

- Lets sellers list used phones with images and verification data.
- Detects visible damage from phone images using a YOLO segmentation model.
- Converts damage into a 0-20 condition score.
- Predicts used-phone price ranges from scraped OLX Pakistan listings.
- Recommends phones from YouTube review data based on budget and priority.
- Provides a smartphone-focused AI chatbot.

## Repository Layout

```text
IntelliFone/
  web/                         Next.js frontend
  ai-backend/                  FastAPI AI/ML backend
    main.py                    API entrypoint
    models.py                  shared Pydantic models
    Dockerfile                 API-only Docker image
    requirements.ai-backend.txt
    DamageDetection/           YOLO inference
    ConditionScoring/          damage-to-score logic
    PricePrediction/           OLX-based price prediction
    RecommendationEngine/      recommendation endpoint logic
    DataCronJob/               OLX and YouTube scheduled jobs
    ChatBot/                   MongoDB-backed AI assistant
  CODEBASE_MODULE_ANALYSIS.md  detailed module-by-module architecture
```

Use the files under `ai-backend/` for backend development. Any old root-level duplicate backend folders, if present, are not the active backend.

## Main Architecture

```text
Frontend web app
  -> Supabase for auth, marketplace data, image storage
  -> FastAPI ai-backend for AI verification, price prediction, recommendations, chatbot

FastAPI ai-backend
  -> MongoDB for OLX listings, YouTube recommendation data, chatbot history
  -> Supabase Storage for generated damage reports
  -> DeepSeek for LLM extraction, classification, recommendations, chatbot replies
  -> ScrapingBee for OLX Pakistan proxy fetching when configured

Separate cron jobs
  -> OLX scraper writes used_mobiles into MongoDB
  -> YouTube watcher writes videos and phones into MongoDB
```

## AI Backend Features

| Module | Purpose |
| --- | --- |
| `DamageDetection` | Runs YOLO on phone images and returns detected cracks, dots, and lines |
| `ConditionScoring` | Converts damage JSON into a condition score and AI damage flags |
| `PricePrediction` | Trains a Random Forest from matching OLX listings and predicts price range |
| `DataCronJob/olx_scraper_service.py` | Scrapes OLX listings with ScrapingBee-first, direct-fetch fallback |
| `DataCronJob/cron_scraper.py` | Round-robin scheduled OLX scraper by brand/model |
| `DataCronJob/youtube_watcher_service.py` | Monitors YouTube channels for list-style phone recommendation videos |
| `DataCronJob/recommender_data_service.py` | Extracts phone recommendations from transcripts |
| `RecommendationEngine` | Ranks candidate phones for budget/priority requests |
| `ChatBot` | Smartphone-focused assistant with MongoDB conversation history |

## Required Services

- MongoDB Atlas or local MongoDB
- Supabase project with Storage bucket for reports
- DeepSeek API key
- YouTube Data API key for the YouTube watcher
- ScrapingBee API key if OLX needs Pakistan proxy access

## AI Backend Environment

Create `ai-backend/.env`:

```env
MONGO_CONNECTION_STRING=mongodb+srv://USER:PASSWORD@HOST/MobileDB

DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
SUPABASE_REPORTS_BUCKET=phone-reports
SUPABASE_REPORTS_FOLDER=damage_reports

YOUTUBE_API_KEY=your_youtube_api_key
SCRAPINGBEE_API_KEY=your_scrapingbee_key

ALLOWED_ORIGINS=http://localhost:3000
MAX_IMAGE_BYTES=10485760
```

Notes:

- `SCRAPINGBEE_API_KEY` is optional, but useful for OLX Pakistan.
- `DEEPSEEK_MODEL` defaults to `deepseek-chat`.
- `DEEPSEEK_BASE_URL` defaults to `https://api.deepseek.com`.
- `ALLOWED_ORIGINS` is comma-separated for deployment, for example:

```env
ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

## Run AI Backend Locally

```powershell
cd ai-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.ai-backend.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API root: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- FastAPI docs: `http://localhost:8000/docs`

## Run Cron Jobs Separately

Cron jobs are intentionally separate from the FastAPI API process. Do not run them inside `main.py` or inside the web server process.

OLX scraper:

```powershell
cd ai-backend
.\.venv\Scripts\python.exe DataCronJob\cron_scraper.py
```

YouTube watcher:

```powershell
cd ai-backend
.\.venv\Scripts\python.exe DataCronJob\youtube_watcher_service.py
```

Suggested deployment schedules:

| Job | Command | Suggested Frequency |
| --- | --- | --- |
| OLX scraper | `python DataCronJob/cron_scraper.py` | every 6-12 hours or daily |
| YouTube watcher | `python DataCronJob/youtube_watcher_service.py` | every 6-12 hours or daily |

For free/cheap cron hosting, GitHub Actions is usually enough. For more reliable scheduled jobs, use Render Cron Jobs or a small VPS cron/systemd timer.

## Docker

The Dockerfile builds the API service only.

```powershell
cd ai-backend
docker build -t intellifone-ai-backend .
docker run --env-file .env -p 8000:8000 intellifone-ai-backend
```

The Docker image installs `requirements.ai-backend.txt` and starts:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Deploy cron jobs as separate scheduled commands or separate services.

## Main API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Welcome response |
| `/health` | GET | Lightweight health check |
| `/damage-detection/` | POST | Download image URLs, run YOLO, create PDF report, return condition score |
| `/condition-scoring/` | POST | Score existing damage JSON |
| `/price-prediction/` | POST | Predict price range from phone details and condition |
| `/full-verification/` | POST | Upload images, detect damage, score condition, predict price |
| `/recommend/` | GET | Recommend phones by budget and priority |
| `/chat` | POST | Smartphone AI assistant |
| `/chat/{conversation_id}` | GET | Fetch saved assistant chat history |

## Current Backend Hardening

- OLX scraper test execution is commented, so imports do not start scraping.
- Cron jobs run separately from the API deployment.
- MongoDB index creation is explicit and wrapped in safe setup functions.
- API startup prepares only API-needed indexes.
- OLX and YouTube cron scripts prepare their own indexes before running.
- Damage detection uses per-request temporary folders.
- Image uploads/downloads have content-type and size checks.
- FastAPI CORS is controlled by `ALLOWED_ORIGINS`.
- `/health` exists for deployment health checks.

## Data Retention

OLX pricing data:

- Stored in `MobileDB.used_mobiles`.
- Older listings are intentionally preserved.
- Price prediction uses latest 60-day listings first, then older listings if recent data is too small.

YouTube recommendation data:

- Stored in `MobileDB.phones` and `MobileDB.videos`.
- Phone recommendation records expire after 60 days because recommendation content gets stale.

## Frontend

The frontend lives in `web/`.

Typical local setup:

```powershell
cd web
npm install
npm run dev
```

Set the frontend AI backend URL to the FastAPI deployment/local URL according to the frontend environment variables used in `web/`.

## Deployment Recommendation

Cheapest practical split:

```text
Frontend: Vercel
AI API: Koyeb free, Render free/starter, Railway hobby, or small VPS
Database: MongoDB Atlas M0 to start
Cron jobs: GitHub Actions free or Render Cron Jobs
Reports/images: Supabase Storage
```

For a more stable low-cost setup:

```text
Frontend: Vercel
AI API: Render Starter / Railway Hobby / small VPS
Cron jobs: Render Cron Jobs or VPS cron
MongoDB: Atlas M0 first, upgrade when storage/traffic grows
```

## More Details

Read `CODEBASE_MODULE_ANALYSIS.md` for the full module-by-module explanation of how the scraper, price prediction, damage detection, recommendation engine, and chatbot work.

# IntelliFone

IntelliFone is an AI-powered used-smartphone marketplace for Pakistan. It combines a Next.js web marketplace with a FastAPI AI backend that verifies phone condition from images, generates damage reports, predicts fair resale price ranges from live OLX data, recommends phones from YouTube review content, and provides a smartphone-focused AI assistant.

The project is also documented as a final-year research system: the research paper describes a YOLOv11-seg damage-detection pipeline trained on iteratively expanded marketplace image datasets, survey-calibrated condition scoring, an OLX-backed Random Forest pricing model, and a YouTube-review recommendation engine.

## Core Capabilities

- Seller listings with Supabase auth, profiles, image storage, and marketplace data.
- AI damage detection on required front/back phone images using a YOLO segmentation model.
- Condition scoring on a `0-20` scale with AI flags for screen cracks, panel dots, and panel lines.
- Branded PDF damage reports uploaded to Supabase Storage.
- Used-phone price prediction from MongoDB OLX market listings.
- GSMArena specification lookup and MongoDB caching through the specs fetcher.
- Buyer/seller chat backed by Supabase data and Pusher realtime events.
- AI assistant and recommendation flows powered by DeepSeek-compatible OpenAI clients.
- YouTube transcript ingestion for budget/priority phone recommendations.

## Repository Layout

```text
IntelliFone/
  README.md
  CODEBASE_MODULE_ANALYSIS.md
  web/                         Next.js 16 web marketplace
  ai-backend/                  FastAPI AI/ML backend
    main.py                    full API: verification, specs, recommendations, chat
    main_api.py                smaller API: verification, specs, price prediction
    models.py                  shared Pydantic models
    Dockerfile                 container starts main_api:app
    best4.pt                   YOLO model file
    DamageDetection/           YOLO segmentation inference
    ConditionScoring/          damage-to-condition scoring
    ReportGenerator/           PDF report generation and Supabase upload
    PricePrediction/           OLX-backed Random Forest pricing
    RecommendationEngine/      DeepSeek recommendation ranking
    SpecsFetcher/              GSMArena lookup/cache service
    DataCronJob/               OLX and YouTube scheduled jobs
    ChatBot/                   MongoDB-backed AI assistant
```

Use `ai-backend/` for backend development. The active repository no longer has active root-level backend folders outside `ai-backend/`.

## UI Screenshots

<img width="1600" height="744" alt="image 2" src="https://github.com/user-attachments/assets/1f81dcd2-a567-4bcd-abb6-ae4a95ac18e9" />

--

<img width="1600" height="801" alt="image 3" src="https://github.com/user-attachments/assets/e9db2842-4094-4735-a3b3-15a3d2f27a7d" />

--

<img width="1600" height="789" alt="image 4" src="https://github.com/user-attachments/assets/eb070aab-e9e4-47c6-ae11-64285e8ecf41" />

--

<img width="1600" height="777" alt="image1" src="https://github.com/user-attachments/assets/e5586c55-f257-48ff-881a-95ff58335372" />

--

<img width="1600" height="733" alt="image 5" src="https://github.com/user-attachments/assets/02f1ebc3-714f-480d-9c64-5a5b3e84c83d" />

---

## Architecture

```text
Web client on Vercel
  -> Supabase Auth, Database, and Storage
  -> Next.js API routes for marketplace/chat proxies
  -> FastAPI AI backend endpoints

FastAPI AI backend
  -> YOLO damage detection and PDF report generation
  -> Random Forest price prediction from MongoDB used-phone listings
  -> DeepSeek LLM calls for scraping, recommendations, and AI chat
  -> GSMArena specs lookup through SerpApi

Background jobs
  -> OLX scraper writes used_mobiles into MongoDB
  -> YouTube watcher writes videos and recommendation phones into MongoDB
```

Current deployment notes from the codebase:

- `main.py` exposes the full backend, including chat and recommendations.
- `main_api.py` exposes verification, specs, and price prediction only.
- `ai-backend/Dockerfile` currently starts `main_api:app`, so Docker deployments do not expose chat or recommendation routes unless changed.
- Web proxy routes currently hardcode `http://127.0.0.1:8000`; production should replace this with environment-based backend URLs.

## Backend Environment

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

SERPAPI_API_KEY=your_serpapi_key
YOUTUBE_API_KEY=your_youtube_api_key
SCRAPINGBEE_API_KEY=your_scrapingbee_key

ALLOWED_ORIGINS=http://localhost:3000
MAX_IMAGE_BYTES=10485760
```

`SCRAPINGBEE_API_KEY` is optional for OLX scraping but useful for Pakistan proxy access. `SERPAPI_API_KEY` is required by `/mobile-specs/` when a cached GSMArena result is not already available.

## Run Locally

Backend:

```powershell
cd ai-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd web
npm install
npm run dev
```

Useful backend URLs:

- API root: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- FastAPI docs: `http://localhost:8000/docs`
- Web app: `http://localhost:3000`

## Main FastAPI Endpoints

| Endpoint | Method | Available In | Purpose |
| --- | --- | --- | --- |
| `/` | GET | `main.py`, `main_api.py` | Welcome response |
| `/health` | GET | `main.py`, `main_api.py` | Lightweight health check |
| `/mobile-specs/` | POST | `main.py`, `main_api.py` | Fetch/cache GSMArena specs |
| `/damage-detection/` | POST | `main.py`, `main_api.py` | Run YOLO on front/back image URLs and return score/report data |
| `/condition-scoring/` | POST | `main.py`, `main_api.py` | Score existing damage JSON |
| `/price-prediction/` | POST | `main.py`, `main_api.py` | Predict used phone price range |
| `/full-verification/` | POST | `main.py`, `main_api.py` | Upload images, detect damage, score condition, predict price |
| `/recommend/` | GET | `main.py` | Non-streaming recommendations |
| `/recommend-stream/` | GET | `main.py` | Streaming recommendations |
| `/chat` | POST | `main.py` | Non-streaming AI assistant |
| `/chat-stream` | POST | `main.py` | Streaming AI assistant |
| `/chat/{conversation_id}` | GET | `main.py` | Fetch assistant chat history |
| `/conversations/{user_id}` | GET | `main.py` | List assistant conversations for a user |

## Background Jobs

Run these separately from the API process.

```powershell
cd ai-backend
.\.venv\Scripts\python.exe DataCronJob\cron_scraper.py
.\.venv\Scripts\python.exe DataCronJob\youtube_watcher_service.py
```

Suggested scheduling:

| Job | Purpose | Suggested Schedule |
| --- | --- | --- |
| `DataCronJob/cron_scraper.py` | Round-robin OLX listing ingestion by brand/model | Daily or every 6-12 hours |
| `DataCronJob/youtube_watcher_service.py` | YouTube video classification and transcript extraction | Weekly or daily depending on quota |

## Docker

```powershell
cd ai-backend
docker build -t intellifone-ai-backend .
docker run --env-file .env -p 8000:8000 intellifone-ai-backend
```

Current Docker behavior:

- Installs `requirements.txt`.
- Starts `uvicorn main_api:app --host 0.0.0.0 --port 8000`.
- Does not expose chat or recommendation endpoints unless the command is changed to `main:app` or `main_api.py` is expanded.

## Important Implementation Notes

- The active damage flow requires exactly two images in this order: `front`, `back`.
- Damage detection writes per-request temporary files, which avoids cross-request collisions.
- Condition scoring currently weights `front` as `1.0` and `back` as `0.8`, then applies logarithmic penalties by damage type.
- The research paper mentions broader six-view capture, sensor diagnostics, and battery health; those are product/research concepts, while the current backend verification code uses the front/back image workflow.
- `SpecsFetcher/specs_service.py` currently contains a demo `print(fetch_mobile_specs("Samsung", "Galaxy S21"))` at import time. Remove or guard it before production deployment because it can trigger external requests during API startup.
- `web/app/api/phones/add/route.ts` expects `pdf_path`, while FastAPI returns `pdf_url`. This contract needs alignment before relying on listing verification end-to-end.

## More Detail

Read `CODEBASE_MODULE_ANALYSIS.md` for the full module-by-module explanation, current risks, and improvement roadmap.

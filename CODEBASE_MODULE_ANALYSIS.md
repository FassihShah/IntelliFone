# IntelliFone Codebase Module Analysis

## Scope

This document explains the current IntelliFone repository as it exists now, with special attention to the updated AI backend pipelines.

It covers:

- What each major module does
- Which data each module reads and writes
- How modules call each other
- Why important design choices exist
- Which environment variables and external services are required
- Current limitations and operational notes

The real backend code lives under `ai-backend/`. There are older root-level duplicate backend files in folders such as `DataCronJob/`, `PricePrediction/`, and `RecommendationEngine/`; those are not the active tracked backend and should not be used for development.

## High-Level Architecture

IntelliFone has two main runtimes:

1. `web/`
   - Next.js frontend
   - Supabase-backed auth, marketplace listings, storage, and buyer/seller chat
   - API routes that proxy some AI requests to the FastAPI backend

2. `ai-backend/`
   - FastAPI service
   - ML, LLM, scraping, and data-ingestion code
   - MongoDB-backed AI datasets and chat history

Storage is split by responsibility:

| Storage | Used For |
| --- | --- |
| Supabase | user auth, marketplace listings, profiles, storage buckets, buyer/seller chat |
| MongoDB | OLX used-phone market data, YouTube recommendation data, AI assistant conversations |

The AI backend currently uses DeepSeek through OpenAI-compatible clients:

- LangChain `ChatOpenAI` with `base_url=https://api.deepseek.com`
- OpenAI SDK `OpenAI(..., base_url=https://api.deepseek.com)` in the YouTube watcher

This does not mean requests go to OpenAI. The `base_url` sends them to DeepSeek.

## Current Deployment Topology

The codebase is deployed as multiple runtime services instead of one single backend process.

| Runtime | Hosting | Schedule / Role | Main Code |
| --- | --- | --- | --- |
| Web frontend | Vercel | Public web marketplace and admin/user UI | `web/` |
| Mobile app | React Native app deployment/distribution | Native mobile client for IntelliFone | React Native app code, outside or separate from the shown `web/` Next.js app if not present in this repository |
| OLX cron job | Render Cron Job | Runs daily at 2:00 AM Pakistan time | `ai-backend/DataCronJob/cron_scraper.py` |
| YouTube cron job | Render Cron Job | Runs weekly on Saturday at 2:00 AM Pakistan time | `ai-backend/DataCronJob/youtube_watcher_service.py` |
| Chatbot + recommendation service | Separate EC2 server | Serves AI chat and recommendation endpoints together because chatbot calls the recommendation engine for recommendation-style messages | `ai-backend/main.py`, `ChatBot/`, `RecommendationEngine/` |
| Remaining FastAPI service | Separate EC2 server | Serves the other AI endpoints such as damage detection, condition scoring, full verification, report generation, and price prediction | `ai-backend/main.py` or `ai-backend/main_api.py`, depending on deployed entrypoint |

Why chatbot and recommendation are deployed together:

- `ChatBot/chatbot.py` imports and calls `RecommendationEngine/recommendation_service.py`.
- Recommendation-style chatbot messages are routed into `get_recommendations()` or `stream_recommendations()`.
- Keeping these two modules on the same EC2 server avoids cross-service calls inside a single chatbot request.

Operational implication:

- The frontend must know which backend base URL to call for each API family.
- Chat and recommendation proxy routes should point to the chatbot/recommendation EC2 server.
- Damage detection, condition scoring, price prediction, and verification proxy routes should point to the remaining FastAPI EC2 server.
- Cron jobs do not run inside either EC2 API process; Render runs them independently on their schedules.
- The Vercel web frontend and React Native mobile app should share the same backend API contracts so marketplace, verification, recommendation, and chat behavior stays consistent across clients.

## Required Backend Environment Variables

The backend reads these variables from `ai-backend/.env`:

| Variable | Required For |
| --- | --- |
| `MONGO_CONNECTION_STRING` | MongoDB access for scraping, recommendations, price prediction, and chat |
| `DEEPSEEK_API_KEY` | All LLM calls after migration to DeepSeek |
| `DEEPSEEK_MODEL` | Optional; defaults to `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | Optional; defaults to `https://api.deepseek.com` |
| `SCRAPINGBEE_API_KEY` | Optional; OLX scraper uses it first, then falls back to direct requests |
| `YOUTUBE_API_KEY` | YouTube watcher |
| `SUPABASE_URL` | Damage report upload |
| `SUPABASE_SERVICE_ROLE_KEY` | Damage report upload |
| `SUPABASE_REPORTS_BUCKET` | Optional; defaults to `phone-reports` |
| `SUPABASE_REPORTS_FOLDER` | Optional; defaults to `damage_reports` |
| `ALLOWED_ORIGINS` | Optional comma-separated frontend origins for FastAPI CORS |
| `MAX_IMAGE_BYTES` | Optional max image upload/download size; defaults to 10 MB |

## AI Backend Entry Point

File: `ai-backend/main.py`

The FastAPI app exposes these routes:

| Endpoint | Purpose | Main Modules |
| --- | --- | --- |
| `GET /` | Basic welcome route | none |
| `GET /health` | Lightweight deployment health check | none |
| `POST /damage-detection/` | Download image URLs, run YOLO, create report, score condition | `DamageDetection`, `report_generator`, `ConditionScoring` |
| `POST /condition-scoring/` | Score damage JSON directly | `ConditionScoring` |
| `POST /price-prediction/` | Predict used phone price range | `PricePrediction` |
| `POST /full-verification/` | Run damage detection, scoring, and price prediction in one flow | damage, scoring, price modules |
| `GET /recommend/` | Recommend phones by budget and priority | `RecommendationEngine` |
| `GET /recommend-stream/` | Stream recommendation text chunk by chunk | `RecommendationEngine.stream_recommendations` |
| `POST /chat` | AI assistant chat | `ChatBot` |
| `POST /chat-stream` | Stream AI assistant replies and save the completed response | `ChatBot.generate_stream_reply` |
| `GET /chat/{conversation_id}` | Get saved AI assistant conversation | `ChatBot.crud` |
| `GET /conversations/{user_id}` | List a user's saved AI assistant conversations | `ChatBot.crud` |

Startup checks require:

- `MONGO_CONNECTION_STRING`
- `DEEPSEEK_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `ai-backend/best3.pt`

The YOLO import is lazy-loaded inside the damage detection function so the FastAPI app can import cleanly without immediately initializing Ultralytics.

The API also configures CORS from `ALLOWED_ORIGINS`. If that variable is missing, local frontend origins are allowed by default:

- `http://localhost:3000`
- `http://127.0.0.1:3000`

Image inputs are hardened with:

- HTTP/HTTPS URL validation for image URL workflows
- image content-type checks
- max image size enforcement through `MAX_IMAGE_BYTES`
- per-request temporary folders for uploaded images, annotated outputs, and generated reports

Why this matters:

- concurrent damage-detection requests no longer share `uploads/`, `outputs/`, or `reports`
- one user request cannot delete another request's temporary files
- deployment health checks can call `/health` without loading YOLO or touching external services

### `main.py` vs `main_api.py`

There are two FastAPI entrypoint files:

| File | Role | Important Difference |
| --- | --- | --- |
| `ai-backend/main.py` | Full AI backend | Includes damage detection, condition scoring, price prediction, recommendations, streaming recommendations, chatbot, streaming chatbot, and conversation list endpoints |
| `ai-backend/main_api.py` | Smaller API variant | Includes damage detection, condition scoring, price prediction, and full verification, but does not include chatbot or recommendation routes |

Current Docker behavior:

- `ai-backend/Dockerfile` starts `uvicorn main_api:app`.
- That means a container built from the current Dockerfile will not expose `/recommend/`, `/recommend-stream/`, `/chat`, `/chat-stream`, `/chat/{conversation_id}`, or `/conversations/{user_id}`.
- If the deployed AI backend is expected to support the current web chat and recommendation pages, Docker should start `main:app` instead of `main_api:app`, or `main_api.py` should be expanded to include those routes.

Dependency-file note:

- `README.md` says the Docker image installs `requirements.ai-backend.txt`.
- The current Dockerfile copies and installs `requirements.txt`.
- Keep these aligned before deployment so the local and container runtime dependencies do not drift.

## Shared Python Models

File: `ai-backend/models.py`

### `UsedMobile`

This is the core schema shared by:

- OLX scraper
- price prediction endpoint
- full verification endpoint

Important fields:

- `brand`, `model`, `ram`, `storage`
- `condition`, `condition_score`
- PTA and damage flags
- `price`, `city`
- `listing_source`, `images`, `post_date`

### Chat Models

- `ChatRequest`
- `ChatResponse`
- `ChatMessage`
- `ChatHistoryResponse`

These structure the MongoDB-backed AI assistant chat endpoints.

### `NewMobile`

`NewMobile` is also defined in both:

- `ai-backend/models.py`
- `ai-backend/DataCronJob/models.py`

It models new-phone specification data such as OS, release year, cameras, chipset, network, sensors, dimensions, and price. The current runtime code does not actively use this schema in the API routes, OLX scraper, recommendation engine, or chatbot. It is a future-ready model for a possible new-phone specification catalog.

The duplicate `DataCronJob/models.py` should be treated carefully:

- It repeats `UsedMobile` and `NewMobile`.
- The active scraper imports `UsedMobile` from root `models.py`, not from `DataCronJob/models.py`.
- If schemas change later, update or remove the duplicate to avoid silent drift.

## Damage Detection Pipeline

Primary file: `ai-backend/DamageDetection/Damage_Detection.py`

Purpose:

Detect physical damage from phone images using a YOLO segmentation model.

Inputs:

- YOLO model path, usually `ai-backend/best3.pt`
- dictionary of side names to local image paths:
  - `front`
  - `back`
  - `left`
  - `right`
  - `top`
  - `bottom`

Flow:

1. `analyze_phone_images()` receives image paths and an optional `output_dir`.
2. It lazy-imports `YOLO` from `ultralytics`.
3. It runs prediction for each valid image.
4. For each mask, `process_yolo_result()` converts the segmentation polygon into damage metrics.
5. Dots are measured by polygon area.
6. Cracks and lines are measured by the max span of polygon bounds.
7. It returns:

```json
{
  "damages": {
    "front": {
      "crack": [{"length_px": 123.4}],
      "dot": [{"area_px": 55.2}]
    }
  }
}
```

Why this design:

- Each side is processed independently, making it simple to map damage severity to visible phone areas.
- The output is compact and usable by condition scoring.

Operational note:

- The module sets local runtime config paths for Matplotlib and Ultralytics under `ai-backend/.runtime/` to avoid Windows permission issues.
- The FastAPI app passes a per-request output directory, so annotated images are isolated per request.

## Condition Scoring Pipeline

Primary file: `ai-backend/ConditionScoring/condition_scoring.py`

Purpose:

Convert raw detected damage into a numeric condition score and boolean AI damage flags.

Flow:

1. Accepts damage JSON.
2. Applies side weights:
   - front is most important
   - back is medium
   - edges are lower impact
3. Applies damage severity weights:
   - crack
   - line
   - dot
4. Uses logarithmic scaling so many small detections do not explode the penalty linearly.
5. Returns:

```json
{
  "condition_score": 17.2,
  "penalty_total": 12.5,
  "ai_detected": {
    "screen_crack": true,
    "panel_dot": false,
    "panel_line": false
  }
}
```

Why this design:

- The price module needs both a numeric condition score and simple boolean damage flags.
- Log scaling makes large damage matter without letting a single noisy mask dominate too much.

## Report Generation

Primary file: `ai-backend/report_generator.py`

Purpose:

Create a PDF report for detected damages and upload it to Supabase.

Flow:

1. `generate_damage_report()` writes an A4 PDF.
2. It lists detected damages by phone side.
3. If annotated images exist, they are embedded.
4. `upload_report_to_supabase()` uploads the PDF to Supabase Storage.

Environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_REPORTS_BUCKET`
- `SUPABASE_REPORTS_FOLDER`

## OLX Used-Mobile Scraping Pipeline

Primary files:

- `ai-backend/DataCronJob/olx_scraper_service.py`
- `ai-backend/DataCronJob/cron_scraper.py`

Purpose:

Collect used-phone market listings from OLX Pakistan and store structured data in MongoDB for price prediction.

MongoDB collection:

- database: `MobileDB`
- collection: `used_mobiles`

Indexes are created by `ensure_olx_indexes()`, not automatically at import time:

```python
collection.create_index([("link", 1)], unique=True)
collection.create_index([("extraction_date", 1)])
```

Old TTL cleanup:

The scraper and price module both drop any old TTL index on `extraction_date` when their index setup functions run. This matters because older listings are now intentionally preserved for fallback training data.

Why index setup is explicit:

- importing scraper or prediction modules should not fail just because MongoDB is briefly unavailable
- cron jobs and API startup can decide when to prepare indexes
- index setup failures are logged without crashing unrelated imports

### Fetching Strategy

`fetch(url)` works like this:

1. If `SCRAPINGBEE_API_KEY` is set:
   - request goes through ScrapingBee
   - `country_code` is `pk`
   - `render_js` is `false`
   - `block_resources` is `true`
2. If ScrapingBee fails:
   - it falls back to a direct `requests.get()`
3. If no ScrapingBee key exists:
   - it uses direct request immediately

Why:

- OLX Pakistan may behave better with Pakistan-based proxy traffic.
- Direct fallback keeps the scraper usable if ScrapingBee is unavailable.

### Search and Detail Flow

`scrape_used_data(model, brand)`:

1. Starts at page 1.
2. Calls `get_ads_from_page(page_num, model, brand)`.
3. Stops when OLX returns zero ads on a page.
4. Stops after 150 successfully saved listings.
5. Sleeps randomly between pages.

`get_ads_from_page()`:

1. Builds query such as `Google Pixel 6A`.
2. Fetches the OLX search page.
3. Selects listings using `li[aria-label='Listing']`.
4. Counts `ads_found`.
5. For each listing:
   - extracts title, price, location, link
   - runs a cheap title pre-filter
   - fetches detail page only if title looks relevant
   - extracts description, details, images
   - calls LLM extraction

The pagination fix is important:

- old behavior stopped when zero records were saved
- new behavior stops only when zero ads are found
- this prevents stopping early when a page contains only duplicates or LLM-skipped listings

### Title Pre-Filter

`title_matches_model(title, model_query, brand)` prevents obvious mismatches before detail fetch.

Example:

- scraping `Google Pixel 6A`
- listing title `Redmi Note 13 Pro`
- skip without detail page fetch or LLM call

Why:

- Saves ScrapingBee credits
- Saves DeepSeek tokens
- Speeds up scraping

The LLM still performs final strict validation later.

### LLM Extraction

The scraper uses:

```python
ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
```

The prompt asks DeepSeek to:

- verify exact brand/model
- allow only `4G`/`5G` suffix differences
- reject multi-model listings
- return either `skip` or raw JSON

The prompt intentionally rejects listings like:

```text
Pixel 6A / Pixel 7 / Pixel 8 Pro all available
```

Why:

- Multi-model shop listings are bad training data for price prediction.

### Save to MongoDB

`save_to_db()`:

1. Converts the `UsedMobile` object to dict.
2. Adds:
   - `_id`
   - `link`
   - `extraction_date`
3. Inserts into `used_mobiles`.
4. Duplicate links are skipped via unique index.

Stored data is later used by the price prediction module.

### Cron Scraper

`cron_scraper.py` reads collection:

- database: `MobileDB`
- collection: `mobile_brands`

Expected brand document shape:

```json
{
  "brand": "Google",
  "models": ["Pixel 6A", "Pixel 7", "..."],
  "model_index": 0
}
```

Flow:

1. Call `ensure_olx_indexes()`.
2. Fetch all brand docs.
3. For each brand, select next batch of models.
4. Call `scrape_used_data(model, brand)` for each model.
5. Move `model_index` forward.

Why:

- Round-robin scraping prevents one brand/model set from dominating collection time.
- The OLX scraper no longer executes a test scrape when imported; the old `scrape_used_data("Pixel 6A", "Google")` line is commented.
- Cron jobs should be deployed as separate scheduled commands, not inside the FastAPI web process.

## Price Prediction Pipeline

Primary file: `ai-backend/PricePrediction/predict_price_service.py`

Purpose:

Predict a used phone price range from current and historical OLX market data.

MongoDB collection:

- database: `MobileDB`
- collection: `used_mobiles`

Indexes are prepared by `ensure_price_prediction_indexes()` during FastAPI startup, with safe error handling.

Constants:

```python
RECENT_LISTINGS_DAYS = 60
MIN_RECENT_LISTINGS = 50
MIN_TRAINING_LISTINGS = 20
```

### Training Data Selection

`fetch_training_data(input_model, input_brand, db)` works in stages:

1. Query broad candidate records by model regex.
2. Apply exact normalized model matching in Python.
3. Split records into:
   - recent listings from the last 60 days
   - all listings
4. If recent valid records are at least 50:
   - use only recent data
5. Otherwise:
   - use all matching records, including older listings
6. If total valid records are below 20:
   - raise an error

Why:

- Recent data reflects current market price.
- Older data gives fallback when recent data is too sparse.
- Preserving older listings improves coverage for less common models.

### Exact Model Matching

The model matching avoids bad mixes like:

- `Pixel 6` vs `Pixel 6A`
- `Pixel 6` vs `Pixel 6 Pro`
- `A52` vs `A52s`
- `Note 13` vs `Note 13 Pro`

It normalizes model names by:

1. Lowercasing.
2. Removing brand tokens.
3. Keeping only letters and digits.

Examples:

```text
Google Pixel 6A -> pixel6a
Pixel 6A        -> pixel6a
Pixel 6         -> pixel6
Pixel 6 Pro     -> pixel6pro
```

Only exact normalized matches are used.

### Feature Engineering

`preprocess_training_data()`:

- extracts numeric RAM and storage
- converts booleans to `0` and `1`
- drops non-feature fields:
  - images
  - post date
  - source
  - city
  - model
  - brand

`preprocess_input_mobile()` performs the same feature normalization for the user input.

### Model Training and Prediction

For each request:

1. Fetch matching market records.
2. Build training dataframe.
3. Train `RandomForestRegressor`.
4. Predict base price.
5. Apply rule-based adjustments:
   - condition score
   - damage flags
   - PTA status
   - panel changed
   - camera/fingerprint status
6. Compute uncertainty from IQR of training prices.
7. Return:

```json
{
  "min_price": 75000,
  "max_price": 85000
}
```

Why train per request:

- It keeps prediction tied to the freshest MongoDB market data.

Tradeoff:

- It is slower than using cached/pretrained models.
- Later improvement could cache models per brand/model for a short time.

## YouTube Recommendation Data Pipeline

Primary files:

- `ai-backend/DataCronJob/youtube_watcher_service.py`
- `ai-backend/DataCronJob/recommender_data_service.py`
- `ai-backend/RecommendationEngine/recommendation_service.py`

Purpose:

Build a recommendation knowledge base from Pakistani phone-review YouTube channels, then use it to answer user recommendation requests.

MongoDB collections:

- `MobileDB.videos`
- `MobileDB.phones`

Recommendation indexes are prepared explicitly:

- the FastAPI app calls `ensure_recommendation_indexes()` for runtime recommendation queries
- the YouTube watcher calls `ensure_recommender_data_indexes()` before ingesting videos
- index setup is wrapped in safe error handling so imports remain lightweight

### YouTube Watcher

File: `youtube_watcher_service.py`

Required env vars:

- `YOUTUBE_API_KEY`
- `MONGO_CONNECTION_STRING`
- `DEEPSEEK_API_KEY`

Configured channels:

```python
CHANNELS = {
    "Babloo Lahori": "...",
    "ReviewsPK": "...",
    "VideoWaliSarkar": "...",
    "MAS TECH": "..."
}
```

Each channel also has a configurable trust weight:

```python
CHANNEL_WEIGHTS = {
    "Babloo Lahori": 1.0,
    "ReviewsPK": 1.0,
    "VideoWaliSarkar": 1.0,
    "MAS TECH": 1.0
}
```

How to set weights:

- `1.2` means highly trusted
- `1.0` means normal
- `0.8` means less trusted
- `0.6` means noisy or low confidence

The watcher:

1. Fetches recent videos for each channel.
2. Uses DeepSeek to classify whether the video is list/recommendation style.
3. Skips single-phone reviews, news, and irrelevant videos.
4. Inserts new relevant videos into `videos`.
5. Calls `process_video(..., channel=name, channel_weight=...)`.

### Transcript Processing

File: `recommender_data_service.py`

`fetch_transcript(video_id)`:

1. Tries English transcript first.
2. If missing, lists available transcripts.
3. Fetches another language if available.
4. Translates non-English chunks to English using `deep-translator`.

`segment_transcript()`:

- sends short transcripts in one LLM call
- splits long transcripts into overlapping chunks
- chunk overlap reduces risk of missing phones at chunk boundaries

### Phone Extraction

DeepSeek is asked to return a JSON array:

```json
[
  {
    "phone_name": "Samsung Galaxy A55",
    "description": "Summary of pros and cons",
    "price_range": 85000
  }
]
```

Validation:

- response must parse as JSON
- each item must have phone name and description
- `price_range` must be:
  - integer
  - multiple of 5000
  - between 5000 and 200000

Invalid `price_range` values become `null`.

### Duplicate Consolidation

Phones are consolidated by exact normalized phone name:

```text
iPhone 15     -> iphone15
iPhone 15 Pro -> iphone15pro
```

This avoids the old unsafe substring behavior where `iPhone 15` could merge with `iPhone 15 Pro`.

### Stored Phone Documents

The `phones` collection stores:

```json
{
  "video_id": "...",
  "phone_name": "Samsung Galaxy A55",
  "phone_name_normalized": "samsunggalaxya55",
  "description": "...",
  "price_range": 85000,
  "video_price_range": "80000_to_100000",
  "source_channel": "ReviewsPK",
  "source_weight": 1.0,
  "source_url": "https://www.youtube.com/watch?v=...",
  "source_title": "...",
  "created_at": "..."
}
```

Indexes:

```python
phones_collection.create_index("created_at", expireAfterSeconds=60 * 24 * 60 * 60)
phones_collection.create_index([("price_range", 1)])
phones_collection.create_index([("phone_name_normalized", 1)])
phones_collection.create_index([("source_channel", 1)])
phones_collection.create_index([("source_weight", -1)])
phones_collection.create_index(
    [("video_id", 1), ("phone_name_normalized", 1)],
    unique=True,
    partialFilterExpression={"phone_name_normalized": {"$exists": True}}
)
videos_collection.create_index(
    [("video_id", 1)],
    unique=True,
    partialFilterExpression={"video_id": {"$exists": True}}
)
```

The YouTube recommendation dataset still expires after 60 days. This is different from OLX pricing data, where older listings are intentionally preserved.

Why YouTube data expires:

- Recommendations from YouTube videos become stale quickly.
- Keeping the recommendation set recent is usually desirable.

### Recommendation Runtime

File: `RecommendationEngine/recommendation_service.py`

`get_recommendations(max_price, priority)`:

1. Queries phones with `price_range <= max_price`.
2. Sorts by highest price under budget first.
3. Limits candidates to 25.
4. If no under-budget phones exist:
   - searches a fallback range from `max_price - 10000` to `max_price + 5000`.
5. Builds candidate text including:
   - phone name
   - description
   - price range
   - source channel
   - source weight
6. Sends candidates to DeepSeek.
7. DeepSeek ranks phones based on user priority.

Why under-budget first:

- A user budget is normally a maximum, not a target-only number.
- Sorting descending keeps the best phones closest to the budget near the top.

Why source weight is included:

- It gives the LLM a trust signal when ranking similar candidates.
- It does not hard-code the winner; it influences reasoning.

### Streaming Recommendations

`stream_recommendations(max_price, priority)` performs the same candidate lookup and prompt construction as `get_recommendations()`, but calls `model.astream(prompt)` and yields text chunks as they arrive.

FastAPI exposes this through:

```text
GET /recommend-stream/?max_price=80000&priority=camera
```

The web route `web/app/api/phones/recommend/route.ts` calls this streaming endpoint and pipes the response body directly back to the browser as `text/plain`.

Why this exists:

- Recommendation answers may take several seconds because they require an LLM response.
- Streaming improves perceived responsiveness on the recommendation page.

Important deployment note:

- The Next.js route currently calls `http://127.0.0.1:8000/recommend-stream/` directly.
- For production, this should be controlled by an environment variable such as `AI_BACKEND_URL` or `NEXT_PUBLIC_AI_BACKEND_URL`.

## Chatbot Pipeline

Primary files:

- `ai-backend/ChatBot/chatbot.py`
- `ai-backend/ChatBot/crud.py`
- `ai-backend/ChatBot/db.py`

Purpose:

Provide a smartphone-focused AI assistant with saved conversation history.

Flow:

1. FastAPI receives `/chat`.
2. If no conversation exists, it creates one.
3. Chat history is loaded from MongoDB.
4. If the message looks like a recommendation request, it routes to `get_recommendations()`.
5. Otherwise it sends chat history and the user message to DeepSeek.
6. User and assistant messages are saved.

Recommendation detection is keyword-based:

- recommend
- suggestion
- best phone
- which phone
- buy
- purchase

Tradeoff:

- Simple and fast, but complex recommendation phrasing may be missed.

### Chat Persistence Details

MongoDB collections:

- `MobileDB.conversations`
- `MobileDB.messages`

Conversation document shape:

```json
{
  "_id": "ObjectId",
  "user_id": "supabase-user-id",
  "title": "first 40 chars of first message",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

Message document shape:

```json
{
  "_id": "ObjectId",
  "conversation_id": "ObjectId",
  "user_id": "supabase-user-id",
  "role": "user | assistant",
  "content": "message text",
  "created_at": "datetime"
}
```

Important behavior:

- `create_conversation()` creates only the conversation row.
- `save_message()` writes each message and updates the parent conversation's `updated_at`.
- `get_chat_history(conversation_id, limit=10)` returns the first 10 messages sorted ascending, because it sorts oldest-first and then limits. If the intent is "latest 10 with context", this should be revised.
- `get_chat_history_formatted()` returns all messages for a conversation in frontend-friendly `{ role, content }` form.
- `get_user_conversations()` returns conversation IDs, titles, and ISO-formatted `updated_at` values for a sidebar/history list.

### Streaming Chat

FastAPI exposes:

```text
POST /chat-stream
```

Flow:

1. Create a conversation if `conversation_id` is missing.
2. Load MongoDB history.
3. Call `generate_stream_reply()`.
4. Yield chunks to the client.
5. After the stream finishes, save the user message and full assistant reply.

Important implementation detail:

- The stream response currently does not send the created `conversation_id` back in a header or first chunk.
- The Next.js `web/app/api/chat/route.ts` tries to forward an `X-Conversation-Id` header if FastAPI provides one, but FastAPI does not currently set it.
- `web/app/components/ChatWindow.tsx` tries to discover new chat IDs from a Supabase `conversations` table after streaming, but the AI assistant stores conversations in MongoDB, not Supabase.
- Result: new streaming AI conversations may save correctly in MongoDB but not reliably appear in the frontend sidebar immediately.

Recommended fix:

- Add `X-Conversation-Id` to the FastAPI streaming response.
- Have the frontend read that header and call `onNewConversation(conversation_id)` directly.
- Remove the fallback Supabase lookup for AI assistant conversations, or create a deliberate Supabase/Mongo mapping table if both are needed.

## Frontend Overview

The frontend lives under `web/`.

The web frontend is deployed on Vercel.

There is also a React Native mobile app for IntelliFone. Its code is not represented by the current `web/` Next.js folder unless it lives in another repository or an unlisted folder. Architecturally, it should consume the same Supabase resources and AI backend APIs as the web client.

Major areas:

| Area | Purpose |
| --- | --- |
| Auth pages | Supabase sign-in, sign-up, OAuth callback |
| Marketplace | browse and filter user-listed phones |
| Add phone flow | upload images, trigger AI damage detection, save listing |
| Recommendation page | collect budget/priority and call backend `/recommend/` |
| Product detail | show phone listing, seller info, report link |
| User chat | Supabase + Pusher buyer/seller messaging |
| Admin | list/manage users and ads |

The web app and AI assistant chat are separate from buyer/seller chat:

- AI assistant chat uses MongoDB in `ai-backend/ChatBot`
- buyer/seller chat uses Supabase and Pusher in `web/`

### Frontend Runtime Stack

Primary package:

- `web/package.json`

Important dependencies:

| Package | Used For |
| --- | --- |
| `next` | App Router web framework |
| `react`, `react-dom` | UI runtime |
| `@supabase/supabase-js` | Auth, database, and storage |
| `@tanstack/react-query` | Marketplace data fetching/caching |
| `pusher`, `pusher-js` | Buyer/seller real-time chat notifications |
| `lucide-react`, `react-icons` | Icons |
| `emailjs-com` | Listing report email submission |
| `@radix-ui/react-select`, `@radix-ui/react-slider` | Select and slider UI controls |

Current frontend environment variables:

| Variable | Used By |
| --- | --- |
| `NEXT_PUBLIC_SUPABASE_URL` | browser and server Supabase client |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | browser Supabase client |
| `SUPABASE_SERVICE_ROLE_KEY` | server-only admin API routes |
| `NEXT_PUBLIC_PUSHER_KEY` | browser and server Pusher setup |
| `NEXT_PUBLIC_PUSHER_CLUSTER` | Pusher cluster |
| `PUSHER_APP_ID` | server-side Pusher trigger route |
| `PUSHER_SECRET` | server-side Pusher trigger route |
| `NEXT_PUBLIC_ADMIN_EMAIL` | client-side admin shortcut login |
| `NEXT_PUBLIC_ADMIN_PASSWORD` | client-side admin shortcut login |
| `NEXT_PUBLIC_EMAILJS_SERVICE_ID` | report-listing email |
| `NEXT_PUBLIC_EMAILJS_TEMPLATE_ID` | report-listing email |
| `NEXT_PUBLIC_EMAILJS_PUBLIC_KEY` | report-listing email |

Security note:

- `NEXT_PUBLIC_ADMIN_PASSWORD` is visible to the browser by design because of the `NEXT_PUBLIC_` prefix. This is not safe for real production admin auth.
- Replace the client-side admin shortcut with role-based authorization from Supabase profiles or app metadata.

### App Shell and Routing

Root files:

- `web/app/layout.tsx`
- `web/app/page.tsx`
- `web/app/home/page.tsx`
- `web/app/ui/AppShell.tsx`
- `web/app/ClientProvider.tsx`
- `web/next.config.ts`
- `web/tsconfig.json`
- `web/eslint.config.mjs`
- `web/postcss.config.mjs`
- `web/components.json`

Behavior:

- `layout.tsx` loads global CSS, wraps the app in `ClientProviders`, and renders every page inside `AppShell`.
- `/` renders the same component as `/home`.
- The current metadata still says `Create Next App`; update it before production.
- `ClientProvider.tsx` creates one React Query `QueryClient` and provides it through `QueryClientProvider`.
- `AppShell.tsx` owns the shared header/footer, watches Supabase auth state, shows guest vs signed-in navigation, closes the mobile menu on route change, and signs out through `supabase.auth.signOut()`.
- Guest users see Login and Sign Up; signed-in users see AI Chat, Inbox, Saved, Profile, and Logout.
- Main navigation links include Marketplace, Sell, Recommendations, About, and Contact Us.
- `next.config.ts` is currently empty.
- `tsconfig.json` enables strict TypeScript, App Router/Next plugin support, and the `@/*` path alias.
- `eslint.config.mjs` uses Next core web vitals and TypeScript presets.
- `postcss.config.mjs` enables Tailwind's PostCSS plugin.
- `components.json` configures shadcn-style UI conventions and aliases for `@/components`, `@/components/ui`, `@/lib`, and `@/hooks`.

### Supabase Clients

Files:

- `web/app/lib/supabaseClient.ts`
- `web/app/lib/supabaseAdmin.ts`

`supabaseClient.ts`:

- Uses `NEXT_PUBLIC_SUPABASE_URL`.
- Uses `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
- Is imported by browser components and some route handlers.

`supabaseAdmin.ts`:

- Uses `NEXT_PUBLIC_SUPABASE_URL`.
- Uses server-only `SUPABASE_SERVICE_ROLE_KEY`.
- Disables auth session persistence.
- Is used by admin and delete API routes for privileged reads/deletes and storage cleanup.

### Marketplace Data Model

The frontend expects a Supabase table named `mobile_phones`.

Fields referenced across pages/routes include:

```text
id
user_id
name
model
company
brand
ram
storage
description
price
pta_status
pictures
condition_score
damage_report_pdf
status
created_at
```

Important naming drift:

- Some code uses `company`; some code uses `brand`.
- `ProductDetailPage` types include `brand`, but add-listing form sends `company`.
- Similar-phone logic filters by `brand`, while marketplace filters by `company`.
- Admin ad query selects `model,company,price`, but the admin UI interface calls the field `brand`.

Recommended fix:

- Pick one canonical field, preferably `brand`, and migrate UI/API/database usage consistently.
- If old rows use `company`, add a migration or compatibility mapping.

### Add/Sell Phone Flows

There are two seller-facing pages:

| Page | Purpose | Current Behavior |
| --- | --- | --- |
| `web/app/add/page.tsx` | Listing creation with image upload and damage detection | Uploads images to `phone-images`, calls `/api/phones/add`, saves listing |
| `web/app/sell-phone/page.tsx` | More explicit AI verification and price-estimate workflow | Uploads images, calls `/api/damage-detection`, calls `/api/price-prediction`, displays estimated range |

`/add` flow:

1. Requires a logged-in Supabase user.
2. Uploads up to 6 images to Supabase Storage bucket `phone-images`.
3. Gets public URLs for uploaded images.
4. Sends `{ user_id, formData, pictureUrls }` to `POST /api/phones/add`.
5. The route calls FastAPI `/damage-detection/`.
6. The route inserts a row into `mobile_phones`.

Current issue in `web/app/api/phones/add/route.ts`:

- FastAPI `/damage-detection/` returns `pdf_url`, `condition_score`, and `ai_detected`.
- The route expects `pdf_path`.
- Because it checks `if (!fastapiData.pdf_path)`, the current route can fail even when FastAPI succeeds.
- The insert stores `damage_report_pdf: pdf_path`, but this should use `fastapiData.pdf_url`.

`/sell-phone` flow:

1. Requires a logged-in user.
2. Uploads images to `phone-images`.
3. Calls `/api/damage-detection`.
4. Receives `condition_score` and `ai_detected`.
5. Auto-fills user damage flags when AI detects crack/dot/line.
6. Sends form data to `/api/price-prediction`.
7. Displays `min_price` and `max_price`.

Difference:

- `/add` saves the listing.
- `/sell-phone` currently estimates condition and price, but does not insert the listing.

Recommended merge:

- Combine both into one seller workflow:
  - upload images
  - damage detection
  - condition flags
  - price prediction
  - final editable listing form
  - save listing with AI report URL and suggested price range

### Frontend API Routes

Files under `web/app/api/` act mostly as BFF/proxy routes.

| Route File | Method | Purpose |
| --- | --- | --- |
| `api/damage-detection/route.ts` | POST | Proxies JSON image URLs to FastAPI `/damage-detection/` and returns condition score plus AI flags |
| `api/price-prediction/route.ts` | POST | Receives form data and forwards it to FastAPI `/price-prediction/` |
| `api/phones/add/route.ts` | POST | Calls FastAPI damage detection, then inserts listing into Supabase |
| `api/phones/list/route.ts` | GET | Reads all `mobile_phones` ordered by newest first |
| `api/phones/recommend/route.ts` | GET | Streams FastAPI `/recommend-stream/` back to the browser |
| `api/phones/delete/[id]/route.ts` | DELETE | Deletes one user's ad after optional ownership check and removes Storage files |
| `api/chat/route.ts` | GET | Fetches AI conversations or one AI chat history from FastAPI |
| `api/chat/route.ts` | POST | Streams AI chat from FastAPI `/chat-stream` |
| `api/messages/send/route.ts` | POST | Triggers Pusher events after buyer/seller message insert |
| `api/admin/users/route.ts` | GET | Lists Supabase `profiles` for admin dashboard |
| `api/admin/ads/route.ts` | GET | Lists Supabase `mobile_phones` for admin dashboard |
| `api/admin/ads/delete/[id]/route.ts` | DELETE | Admin delete ad and associated files |
| `api/admin/users/delete/[id]/route.ts` | DELETE | Admin delete user profile, ads, files, and best-effort auth user |
| `api/users/[id]/route.ts` | GET | Fetches profile `full_name` and `email` |

Hardcoded backend URL:

- Several routes call `http://127.0.0.1:8000`.
- This works locally only when FastAPI is running on the same machine.
- For production, add a server-side env var such as `AI_BACKEND_URL` and use it in all proxy routes.

### Marketplace Pages

`web/app/marketplace/page.tsx`:

- Fetches `/api/phones/list` using React Query.
- Filters by search text, company, storage, and price range.
- Paginates client-side with 12 items per page.
- Uses `ProductCard` for listing cards.

`web/app/phones/[id]/page.tsx`:

- Fetches all phones from `/api/phones/list`.
- Selects one phone client-side by route ID.
- Fetches seller profile from `/api/users/{user_id}`.
- Shows image carousel, price, specs, seller, save button, AI report link, contact button, and report link.
- Creates or finds a buyer/seller conversation in Supabase table `conversation`.
- Redirects to `/chats?conversation={id}`.

Efficiency note:

- Product detail currently fetches all phones and filters client-side.
- Add `GET /api/phones/[id]` later for direct lookup.

Navigation issue:

- Some links use `/product/{id}` while the actual product route is `/phones/{id}`.
- Fix saved/report back links so navigation stays consistent.

### Saved Phones

File:

- `web/app/saved/page.tsx`

Behavior:

- Uses browser `localStorage` key `cart`.
- The "Save" action on product detail stores `{ id, model, price, image }`.
- Saved items are not synced to Supabase and are device/browser-specific.

Improvement:

- Add a `saved_phones` table keyed by `user_id` and `phone_id` so saved listings follow the user across devices.

### Buyer/Seller Chat

Primary files:

- `web/app/chats/page.tsx`
- `web/app/chats/ChatClient.tsx`
- `web/app/components/chat/Inbox.tsx`
- `web/app/components/chat/ChatWindow.tsx`
- `web/hooks/useRealtimeChat.ts`
- `web/app/api/messages/send/route.ts`

Supabase tables expected:

- `conversation`
- `messages`
- `profiles`

Expected `conversation` fields:

```text
id
user1_id
user2_id
```

Expected `messages` fields:

```text
id
conversation_id
sender_id
content
created_at
read_at
```

Flow:

1. Product detail creates or finds a conversation between current user and seller.
2. `/chats` loads the current Supabase session.
3. Inbox fetches conversations where current user is `user1_id` or `user2_id`.
4. Inbox subscribes to Pusher channel `inbox-{currentUserId}` for refresh events.
5. Chat window fetches messages and the other user's profile.
6. Sending a message inserts into Supabase `messages`.
7. The route `/api/messages/send` triggers:
   - `chat-{conversationId}` event `new-message`
   - `inbox-{recipientId}` event `refresh-inbox`
   - `inbox-{senderId}` event `refresh-inbox`
8. `useRealtimeChat()` subscribes to `chat-{conversationId}` and appends incoming messages.

Important limitation:

- Pusher is used for notification/refresh only; Supabase is still the source of truth.
- There is no read-receipt update logic even though `read_at` is used for unread counts.

Naming drift:

- Some helper files reference `conversations`, but active buyer/seller chat code uses `conversation`.
- Keep one table name or clearly split old/new chat table names.

### AI Assistant Chat Frontend

Primary files:

- `web/app/chat/page.tsx`
- `web/app/components/ChatSideBar.tsx`
- `web/app/components/ChatWindow.tsx`
- `web/app/components/MessageInput.tsx`
- `web/app/api/chat/route.ts`

Flow:

1. Page loads Supabase user ID.
2. It calls `/api/chat?user_id={id}`.
3. Next.js proxies to FastAPI `/conversations/{user_id}`.
4. Sidebar displays MongoDB conversation IDs and titles.
5. Selecting a conversation calls `/api/chat?conversation_id={id}`.
6. Sending a message posts to `/api/chat`.
7. Next.js proxies to FastAPI `/chat-stream`.
8. Browser reads chunks and progressively updates the assistant bubble.

Important split:

- AI assistant chat history lives in MongoDB.
- Buyer/seller chat history lives in Supabase.
- Both use the word "conversation", but they are different systems.

### Recommendation Page

File:

- `web/app/recommendation/page.tsx`

Flow:

1. User chooses a budget range and a priority:
   - camera
   - battery
   - gaming
   - overall
2. The page sends the upper budget value as `max_price`.
3. It calls `/api/phones/recommend`.
4. The route streams FastAPI recommendation chunks.
5. The page appends chunks into one response.

Current rendering detail:

- The page strips `*` characters from streamed text with `chunk.replace(/\*/g, ' ')`.
- The backend prompt asks the LLM to use Markdown bold, but the frontend removes asterisks.
- Either render Markdown properly or remove Markdown instructions from the backend prompt.

### Auth and Profiles

Files:

- `web/app/(auth)/signin/page.tsx`
- `web/app/(auth)/signup/page.tsx`
- `web/app/(auth)/callback/page.tsx`
- `web/app/components/auth/GoogleButton.tsx`

Behavior:

- Email/password sign-up uses Supabase Auth and upserts a row into `profiles`.
- Google OAuth callback also upserts `profiles`.
- Sign-in redirects normal users to `/`.
- A client-side admin email/password shortcut redirects to `/admin`.
- `GoogleButton.tsx` calls Supabase Google OAuth.

Profile fields referenced:

```text
id
full_name
email
avatar_url
created_at
```

Issue:

- Sign-up link points to `/auth/signin` in one place, while the actual route is `(auth)/signin`, exposed as `/signin`.
- Make route links consistent.
- `GoogleButton.tsx` redirects OAuth users to `${window.location.origin}/`, so `web/app/(auth)/callback/page.tsx` may not run for Google sign-in unless the redirect URL is changed to `/callback` or Supabase OAuth settings route through it.
- The callback page can create/update `profiles` from OAuth metadata, so decide whether OAuth profile creation should happen there or through another post-login flow.

### Admin Dashboard

Files:

- `web/app/admin/page.tsx`
- `web/app/api/admin/users/route.ts`
- `web/app/api/admin/ads/route.ts`
- `web/app/api/admin/users/delete/[id]/route.ts`
- `web/app/api/admin/ads/delete/[id]/route.ts`

Behavior:

- Fetches users from `profiles`.
- Fetches ads from `mobile_phones`.
- Builds simple six-month bar charts in the browser.
- Allows deleting an ad and its storage files.
- Allows deleting a user, all their ads, all ad images/reports, the profile row, and then attempts to delete the Supabase auth user.

Important security note:

- The admin page and routes rely on frontend navigation/admin shortcut behavior.
- Server routes should verify the current session and admin role before returning or deleting data.

### Profile Page

File:

- `web/app/profile/page.tsx`

Behavior:

- Requires a Supabase user.
- Fetches profile from `/api/users/{id}`.
- Fetches all phone listings from `/api/phones/list`.
- Filters ads by current `user.id`.
- Allows the owner to delete an ad through `/api/phones/delete/{id}` with `{ userId }` in the request body.

Improvement:

- Add `GET /api/phones?user_id={id}` or a dedicated profile ads route so profile does not fetch every listing.
- Use server-side auth instead of trusting a `userId` body for ownership checks.

### Report Listing

File:

- `web/app/report/[id]/page.tsx`

Behavior:

- Fetches all phones and selects one by route ID.
- Lets a user choose a report reason and details.
- Sends an email through EmailJS using public EmailJS config.

Issues:

- Back/cancel links point to `/product/{id}`, but the actual route is `/phones/{id}`.
- The destination email is hardcoded in the client-side template data.
- Reports are not stored in Supabase, so there is no admin queue or audit trail.

Recommended improvement:

- Store listing reports in a Supabase table, then optionally send email from a server route or Supabase Edge Function.

### Static/Informational Pages

Pages present:

- `web/app/about/page.tsx`
- `web/app/contactus/page.tsx`
- `web/app/helpcenter/page.tsx`
- `web/app/privacypolicy/page.tsx`
- `web/app/termsofservice/page.tsx`

These are mostly content pages and do not drive the AI backend.

### UI Components

Reusable UI files include:

- `web/app/components/SearchBar.tsx`
- `web/app/components/card/ProductCard.tsx`
- `web/components/ui/input.tsx`
- `web/components/ui/select.tsx`
- `web/components/ui/slider.tsx`
- `web/components/ui/textarea.tsx`

`ProductCard` is the core listing card used by home, marketplace, and similar-phone sections. The shared UI controls support forms and filtering.

`ProductCard` behavior:

- Displays the first listing image or a fallback Unsplash phone image.
- Shows a "Verified" badge when `pta_status` is truthy.
- Shows price, RAM, storage, and condition score when present.
- Links to `/phones/{phone.id}`.

`SearchBar` behavior:

- Keeps local input state.
- Calls an `onSearch(query)` prop on form submit.
- The home page uses it to redirect to `/marketplace?search={query}`.

`MessageInput` behavior:

- Used by the AI assistant chat window.
- Sends on button click or Enter.
- Clears local input after sending.
- Disables input while a stream is in progress.

Styling files:

- `web/app/globals.css` imports Tailwind and defines global theme variables.
- `web/app/components/chat/chat.css` defines the buyer/seller chat visual system, including inbox layout, chat bubbles, unread badges, and message input styling.

### Supabase Storage Buckets

Buckets expected by the current code:

| Bucket | Used For |
| --- | --- |
| `phone-images` | Seller-uploaded listing images |
| `phone-reports` | AI-generated damage report PDFs |

Storage path handling:

- Delete routes parse public URLs and remove the bucket-relative path.
- Delete routes also accept values already formatted as `bucket/path`.

### Supabase Edge Function

File:

- `web/app/supabase/functions/send-contact-email/index.ts`

This appears to be a Supabase Edge Function placeholder/location for sending contact emails. The current report page uses EmailJS directly from the frontend instead. If email sending should be private and auditable, move report/contact email logic into this Edge Function or a Next.js server route.

## End-to-End Flows

### Seller Verification Flow

1. Seller uploads phone images.
2. Web app stores images in Supabase Storage.
3. Web API sends image URLs to FastAPI.
4. FastAPI downloads images.
5. YOLO detects damage.
6. Condition scoring computes `condition_score`.
7. Report generator creates and uploads PDF.
8. Marketplace listing is saved with verification results.

### Price Prediction Flow

1. User submits brand, model, RAM, storage, condition, and damage flags.
2. FastAPI creates a `UsedMobile`.
3. Price module fetches exact normalized model data from MongoDB.
4. Recent 60-day listings are used if at least 50 valid records exist.
5. Otherwise all historical matching records are used.
6. Random forest trains on current data.
7. Rule-based condition and damage adjustments are applied.
8. Min/max price range is returned.

### OLX Scraping Flow

1. Cron chooses brand/model batch.
2. Scraper searches OLX.
3. Search page listings are title pre-filtered.
4. Relevant detail pages are fetched.
5. DeepSeek validates exact model and extracts structured fields.
6. MongoDB stores unique listings by link.
7. Price prediction later consumes the data.

### YouTube Recommendation Flow

1. Watcher checks configured channels.
2. DeepSeek classifies videos as relevant or irrelevant.
3. Relevant videos are transcript-processed.
4. DeepSeek extracts phones, summaries, and price ranges.
5. MongoDB stores phone recommendation records with source channel and weight.
6. Runtime recommender fetches under-budget candidates.
7. DeepSeek ranks candidates by user priority.

### Marketplace Listing Flow

1. User signs in with Supabase Auth.
2. User uploads phone images to Supabase Storage bucket `phone-images`.
3. Frontend gets public image URLs.
4. Next.js API route calls FastAPI `/damage-detection/` with those URLs.
5. FastAPI downloads images, runs YOLO, generates a PDF, uploads report to `phone-reports`, and returns `pdf_url`, `condition_score`, and AI flags.
6. Next.js inserts the listing into Supabase `mobile_phones`.
7. Marketplace and home pages read listings from `/api/phones/list`.

Current break:

- The Next.js add-listing route expects `pdf_path`, but FastAPI returns `pdf_url`.
- Fix this before relying on the add-listing flow.

### Buyer/Seller Message Flow

1. Buyer clicks contact on a phone detail page.
2. Product page finds or creates a row in Supabase `conversation`.
3. Buyer is redirected to `/chats?conversation={id}`.
4. Chat window loads messages from Supabase `messages`.
5. New message is inserted into Supabase.
6. `/api/messages/send` triggers Pusher updates.
7. Recipient chat window receives `new-message`.
8. Inboxes refresh through `refresh-inbox`.

### AI Assistant Chat Flow

1. User opens `/chat`.
2. Frontend uses Supabase Auth only to identify the user.
3. Conversation list is loaded from MongoDB through FastAPI.
4. User message is streamed through Next.js -> FastAPI -> DeepSeek.
5. FastAPI saves both messages in MongoDB after the stream finishes.

Important difference:

- Supabase Auth identifies the user.
- MongoDB stores AI assistant conversations.
- Supabase stores buyer/seller conversations.

## Current Strengths

- Clear split between marketplace data and AI data.
- OLX scraper now preserves older pricing data for fallback.
- Price prediction avoids model-variant mixing through normalized exact matching.
- YouTube recommendation records now carry source metadata and channel trust weights.
- DeepSeek configuration is consistent across LLM modules.
- Damage detection, condition scoring, report generation, and price prediction form a coherent verification pipeline.
- Streaming exists for recommendations and AI assistant replies, improving perceived speed.
- Admin/profile delete routes clean up associated Supabase Storage objects instead of only deleting database rows.
- The buyer/seller chat treats Supabase as the source of truth and Pusher as the realtime notification layer.

## Current Limitations and Improvement Ideas

### OLX Scraping

- OLX CSS selectors use generated class names and may break.
- LLM extraction depends on seller text quality.
- Pakistani price phrases such as `1 lac` or `1.5 lakh` could be parsed more explicitly.
- The local test run at the bottom of `olx_scraper_service.py` is currently commented; keep it disabled for production imports.

### Price Prediction

- Random forest trains on every request.
- Add model caching per brand/model to reduce latency.
- Add outlier filtering for fake OLX prices.
- Return confidence metadata:
  - records used
  - recent-only or historical fallback
  - uncertainty

### YouTube Recommendations

- Existing old MongoDB `phones` records will not have `phone_name_normalized`, `source_channel`, or `source_weight` until reprocessed.
- Channel weights are manually configured, which is good for control but needs periodic review.
- Recommendation ranking relies on LLM interpretation of candidate descriptions.
- Price extraction from transcripts may still miss phrases like `lac`, `lakh`, or vague prices.

### Web App

- Buyer/seller chat and AI assistant chat are separate systems; this is intentional but should be documented in user-facing architecture.
- Some web chat schema names should be verified against Supabase tables.
- FastAPI URLs are hardcoded to `http://127.0.0.1:8000` in Next.js API routes.
- Add-listing route expects `pdf_path`, but FastAPI returns `pdf_url`.
- AI chat streaming does not return the newly created Mongo conversation ID to the frontend.
- Admin auth is currently a client-side shortcut and should become server-enforced role auth.
- Field naming is inconsistent across `brand` and `company`.
- Some links point to `/product/{id}` even though product details live at `/phones/{id}`.
- Product/profile/report pages often fetch all listings and filter client-side; add direct lookup routes later.

### Deployment/Operations

- Docker starts `main_api:app`, not `main:app`, so chatbot and recommendations are missing in that deployment mode.
- Dockerfile uses `requirements.txt`, while docs mention `requirements.ai-backend.txt`.
- The frontend should use an environment variable for AI backend base URL.
- Add startup or health diagnostics that confirm MongoDB, Supabase report bucket, and DeepSeek configuration separately from the lightweight `/health`.
- Add structured logging for scraper runs, LLM extraction failures, and price prediction record counts.

## Future Additions Roadmap

### Backend Additions

- Cache trained price models per normalized brand/model for a short TTL.
- Return prediction metadata: records used, recent-vs-historical mode, uncertainty, and data date range.
- Add outlier filtering for suspicious OLX prices.
- Add a direct `GET /phones/{id}` style API on the web side instead of fetching all listings.
- Add FastAPI response models for every endpoint for stronger contract validation.
- Add explicit Mongo indexes for chat collections:
  - `conversations.user_id`
  - `conversations.updated_at`
  - `messages.conversation_id`
  - `messages.created_at`
- Add retry/backoff around DeepSeek calls in scraping, recommendations, and chatbot.
- Add a background job queue for expensive damage detection and report generation if uploads become slow.
- Add per-request IDs across frontend, FastAPI, and cron logs for easier debugging.

### Frontend Additions

- Replace hardcoded FastAPI URLs with `AI_BACKEND_URL`.
- Fix the add-listing `pdf_url` contract.
- Merge `/add` and `/sell-phone` into one guided seller flow.
- Render AI recommendation/chat Markdown safely instead of stripping asterisks.
- Add user-owned saved phones in Supabase instead of `localStorage`.
- Add report-listing storage in Supabase with admin review.
- Add admin route protection on the server.
- Add route-level loading/error states for every API-backed page.
- Add direct detail API routes for phone and profile lookups.
- Normalize `brand`/`company` naming.

### Data/Schema Additions

- Create a documented Supabase schema file or migration set for:
  - `profiles`
  - `mobile_phones`
  - `conversation`
  - `messages`
  - future `saved_phones`
  - future `listing_reports`
- Document required Supabase Storage bucket policies.
- Add MongoDB seed script for `mobile_brands`.
- Add validation for listing price, RAM, storage, and PTA status before inserting into Supabase.

### Testing Additions

- Unit tests for:
  - model normalization
  - price data filtering
  - condition scoring
  - storage URL path extraction
  - recommendation candidate selection
- Integration tests for:
  - `/damage-detection/` with mocked YOLO and Supabase upload
  - `/price-prediction/` with test Mongo data
  - `/chat-stream` conversation creation and persistence
  - Next.js add-listing route contract with FastAPI response
- End-to-end tests for:
  - sign up/sign in
  - upload listing
  - marketplace filtering
  - buyer/seller chat
  - AI recommendation streaming

## Active Development Notes

- Use only files under `ai-backend/` for backend development.
- Root-level duplicate backend files are old untracked copies and do not contain the recent changes.
- The lean backend venv is under `ai-backend/.venv/`.
- `ai-backend/requirements.ai-backend.txt` contains the slimmer runtime dependency list for the backend.
- `ai-backend/Dockerfile` currently installs `requirements.txt` and starts `main_api:app` with Uvicorn.
- If production needs chatbot and recommendation routes, start `main:app` or merge those routes into `main_api.py`.
- Deploy cron jobs separately from the API container:
  - OLX cron command: `python DataCronJob/cron_scraper.py`
  - YouTube cron command: `python DataCronJob/youtube_watcher_service.py`
- Current production scheduling:
  - OLX scraper is deployed as a separate Render Cron Job and runs daily at 2:00 AM Pakistan time.
  - YouTube watcher is deployed as a separate Render Cron Job and runs every Saturday at 2:00 AM Pakistan time.
- Current API deployment split:
  - Chatbot and recommendation engine are deployed together on one EC2 server because they are linked.
  - The remaining FastAPI endpoints are deployed on a separate EC2 server.
- Current client deployment split:
  - The Next.js web frontend is deployed on Vercel.
  - A separate mobile app exists and is built in React Native.

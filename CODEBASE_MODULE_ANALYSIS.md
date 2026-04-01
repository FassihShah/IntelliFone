# IntelliFone Codebase Module Analysis

## Scope

This document is based on the current repository contents in `web/` and `ai-backend/`.
It explains:

- which packages are used by which module
- how each module is implemented
- how modules interact with each other
- notable implementation gaps or inconsistencies visible in code

The analysis is grounded in source files, not only the README. A PDF named `IntelliFone System Modules.pdf` was available locally, but text extraction was blocked in the sandbox, so this write-up is based on the repository itself.

## High-Level Architecture

IntelliFone is split into two main runtimes:

1. `web/`
   - Next.js frontend and API routes
   - handles UI, auth, Supabase CRUD, and proxy calls to the AI backend
2. `ai-backend/`
   - FastAPI service plus supporting ML/data collection scripts
   - handles damage detection, condition scoring, price prediction, recommendations, chatbot, and cron-style data ingestion

Persistent storage is split across:

- Supabase
  - auth
  - relational app data such as `mobile_phones`, `profiles`, chat-related tables, storage buckets
- MongoDB
  - AI/data-engineering collections such as `used_mobiles`, `phones`, `videos`, `conversations`, `messages`

## Package Usage by Area

### Frontend and Web API: `web/package.json`

Core packages:

| Package | Used For |
| --- | --- |
| `next`, `react`, `react-dom`, `typescript` | App runtime and page/component implementation |
| `@supabase/supabase-js` | Auth, database access, storage uploads, profile lookups |
| `@tanstack/react-query` | Marketplace data fetching and caching |
| `tailwindcss`, `@tailwindcss/postcss`, `tw-animate-css` | Styling |
| `@radix-ui/react-select`, `@radix-ui/react-slider` | Marketplace filters and recommendation inputs |
| `lucide-react`, `react-icons` | Icons |
| `emailjs-com` | Contact/report email sending in frontend pages |
| `pusher`, `pusher-js` | Real-time chat and inbox refresh |
| `clsx`, `class-variance-authority`, `tailwind-merge` | Utility styling composition |

### AI Backend: `ai-backend/requirements.txt`

Core API and validation:

| Package | Used For |
| --- | --- |
| `fastapi`, `uvicorn[standard]`, `python-multipart` | FastAPI server, form uploads, API endpoints |
| `pydantic`, `python-dotenv` | request/data models and environment loading |

ML and data processing:

| Package | Used For |
| --- | --- |
| `ultralytics` | YOLO-based damage detection |
| `scikit-learn`, `pandas`, `numpy` | price prediction training and inference |
| `shapely` | polygon area/bounds measurement from segmentation masks |
| `reportlab` | PDF damage report generation |

Data collection and databases:

| Package | Used For |
| --- | --- |
| `pymongo`, `motor`, `dnspython` | MongoDB access |
| `beautifulsoup4`, `requests` | OLX scraping |
| `google-api-python-client` | YouTube Data API integration |
| `youtube-transcript-api` | transcript extraction |
| `deep-translator` | transcript translation to English |

LLM and orchestration:

| Package | Used For |
| --- | --- |
| `openai` | list-video relevance check in YouTube watcher |
| `google-genai`, `langchain-google-genai` | Gemini-backed recommendations, transcript processing, chatbot |
| `langchain`, `langchain-core` | prompt/output pipeline for OLX extraction |

## Main Backend Entry Point

File: `ai-backend/main.py`

This file wires the AI modules into a FastAPI app titled `IntelliFone AI Backend`.

Endpoints implemented:

| Endpoint | Purpose | Main Module Calls |
| --- | --- | --- |
| `GET /` | health-style welcome route | none |
| `POST /damage-detection/` | download image URLs, run YOLO, generate PDF, compute score | `analyze_phone_images`, `generate_damage_report`, `compute_condition_score` |
| `POST /condition-scoring/` | calculate score from damage JSON | `compute_condition_score` |
| `POST /price-prediction/` | predict price band from phone attributes and AI flags | `run_pipeline` |
| `POST /full-verification/` | one-shot detection + scoring + prediction | damage, scoring, prediction pipeline |
| `GET /recommend/` | recommendation text for budget and priority | `get_recommendations` |
| `POST /chat` | chatbot reply plus conversation persistence | `generate_reply`, chat CRUD helpers |
| `GET /chat/{conversation_id}` | formatted chat history | chat CRUD helpers |

Operational flow:

1. frontend sends image URLs or form data to Next.js API routes
2. Next.js routes proxy requests to FastAPI
3. FastAPI calls the relevant AI/data module
4. output is returned to Next.js and, where needed, persisted in Supabase

## Shared Data Models

File: `ai-backend/models.py`

Important models:

- `UsedMobile`
  - shared schema for scraped phones and inference input
  - includes brand, model, RAM, storage, subjective condition, AI condition score, PTA flag, hardware flags, price, city, images, source metadata
- `NewMobile`
  - schema for richer phone specification data, currently not central in the observed flow
- `ChatRequest`, `ChatResponse`, `ChatMessage`, `ChatHistoryResponse`
  - request/response types for chatbot endpoints

This file is the main contract between scraping, prediction, and API layers on the Python side.

## Module-by-Module Analysis

### 1. Damage Detection Module

Primary file:

- `ai-backend/DamageDetection/Damage_Detection.py`

Packages used:

- `ultralytics`
- `cv2`
- `matplotlib`
- `shapely`
- `os`

Implementation:

- loads a YOLO model from a `.pt` file
- expects a mapping of phone sides to local image paths:
  - `front`
  - `back`
  - `left`
  - `right`
  - `top`
  - `bottom`
- runs `model.predict()` per valid image
- reads segmentation masks from YOLO output
- converts each mask into a `Polygon`
- measures damage size:
  - `dot` as polygon area
  - `crack` and `line` as max side of polygon bounds
- returns JSON in this shape:
  - `{"damages": {"front": {"crack": [...], "dot": [...]}, ...}}`

Working logic:

1. FastAPI saves or downloads images locally.
2. `analyze_phone_images()` loads the model.
3. Each valid image is inferred independently.
4. Annotated output images can optionally be saved in `outputs/`.
5. A normalized `damages` structure is returned to later modules.

Notes:

- class names are fixed as `["crack", "dot", "line"]`
- the `DAMAGE_MEASUREMENT` map uses `screen_line` but the actual class list uses `line`; because the code falls back to length measurement, it still works, but the mapping is inconsistent
- `cv2` and `matplotlib` are imported in code but not listed in `requirements.txt`

### 2. Damage Report Generator

Primary file:

- `ai-backend/report_generator.py`

Packages used:

- `reportlab`
- `os`

Implementation:

- creates an A4 PDF
- prints each phone side as a section
- embeds the saved annotated image if it exists
- lists detected damage metrics under that side

Working logic:

1. `/damage-detection/` saves annotated images to `outputs/`.
2. `generate_damage_report()` builds a PDF in `reports/`.
3. the Next.js add-phone route reads the generated PDF from disk and uploads it to Supabase Storage.

### 3. Condition Scoring Module

Primary file:

- `ai-backend/ConditionScoring/condition_scoring.py`

Packages used:

- `numpy`
- `json`

Implementation:

- takes either a damage JSON object or a file path to JSON
- applies side weights:
  - front `1.0`
  - back `0.6`
  - left/right/top/bottom `0.3`
- applies severity weights:
  - crack `8`
  - line `7`
  - dot `6`
- sums magnitudes per class and side
- computes penalty using `severity * side_weight * log1p(total_magnitude)`
- converts penalty into a `0-20` condition score with `20 - penalty / SCALE`
- also sets AI boolean flags:
  - `screen_crack`
  - `panel_dot`
  - `panel_line`

Working logic:

1. consumes the `damages` object from the damage detection module
2. compresses multiple detections into a single penalty score
3. returns both numeric score and feature flags for price prediction

Output shape:

- `condition_score`
- `penalty_total`
- `ai_detected`

### 4. Price Prediction Module

Primary file:

- `ai-backend/PricePrediction/predict_price_service.py`

Packages used:

- `pymongo`
- `scikit-learn`
- `pandas`
- `dotenv`
- `re`

Implementation:

- connects to MongoDB collection `MobileDB.used_mobiles`
- fetches training records using regex match on the requested model
- converts raw OLX records into `UsedMobile`
- derives condition score from subjective condition if missing
- preprocesses training and input rows:
  - extracts numeric values from `ram` and `storage`
  - converts booleans to `0/1`
  - removes non-feature fields
- trains a `RandomForestRegressor` on the fly for the requested model
- predicts a base price
- adjusts it using:
  - AI condition score
  - discrepancy between user-declared damage flags and AI-detected flags
  - PTA status
  - panel and hardware penalties
- calculates uncertainty from interquartile range of market prices
- returns `{min_price, max_price}`

Working logic:

1. `run_pipeline()` fetches same-model training data from MongoDB.
2. if fewer than 20 valid records exist, it raises an error.
3. a random forest is trained per request.
4. the raw prediction is post-adjusted by rules.
5. final price band is rounded to nearest `500`.

Important implementation detail:

- the model is not pre-trained and saved; training happens during inference for each request based on current MongoDB data

Strengths:

- uses fresh scraped market data
- adapts uncertainty to price dispersion for that model

Limitations visible in code:

- no caching of trained models
- same-model regex fetch may mix variants with similar names
- runtime depends on MongoDB data quality and availability

### 5. OLX Scraping and Market Data Module

Primary files:

- `ai-backend/DataCronJob/olx_scraper_service.py`
- `ai-backend/DataCronJob/cron_scraper.py`

Packages used:

- `beautifulsoup4`
- `requests`
- `pymongo`
- `python-dotenv`
- `langchain-core`
- `langchain-google-genai`
- `langchain_openai` import in source

Implementation in `olx_scraper_service.py`:

- scrapes OLX listing pages and detail pages
- extracts title, price, location, description, details, image URLs, link
- builds a strict prompt asking an LLM to:
  - verify listing matches expected brand/model
  - reject multi-model or mismatched listings with `"skip"`
  - return structured JSON for valid listings
- sanitizes LLM JSON
- validates it against `UsedMobile`
- stores results in MongoDB with:
  - unique index on `link`
  - TTL index on `extraction_date` for 60 days

Working logic:

1. `scrape_used_data(model, brand)` iterates OLX search pages.
2. each listing is fetched and parsed with BeautifulSoup.
3. listing text is sent through an LLM extraction chain.
4. validated records are inserted into `used_mobiles`.
5. process stops when no more listings are found or 150 records are saved.

Implementation in `cron_scraper.py`:

- reads brand/model lists from MongoDB collection `mobile_brands`
- processes brands in round-robin batches
- uses `model_index` to remember where the next batch should start
- default batch size is 10 models per run

Why it matters:

- this module is the training data source for the price prediction service

Important code observation:

- `olx_scraper_service.py` imports `ChatOpenAI` from `langchain_openai`, but `langchain-openai` is not listed in `requirements.txt`

### 6. Recommendation Data Ingestion Module

Primary file:

- `ai-backend/DataCronJob/recommender_data_service.py`

Packages used:

- `youtube-transcript-api`
- `deep-translator`
- `pymongo`
- `langchain-google-genai`
- `json`, `re`

Implementation:

- fetches a YouTube transcript by video ID
- tries English first
- falls back to other languages and translates to English
- chunks long transcripts
- sends transcript or chunks to Gemini via LangChain
- expects JSON array of:
  - `phone_name`
  - `description`
  - `price_range`
- consolidates duplicate or similar phone names
- stores:
  - video metadata in `videos`
  - extracted phones in `phones`
- creates a TTL index on `phones.created_at`

Working logic:

1. transcript is fetched from YouTube.
2. if needed, non-English transcript is translated.
3. LLM extracts phones and rounded price bands.
4. duplicate phone mentions are merged.
5. final phone recommendation records are upserted into MongoDB.

Role in system:

- this is the knowledge-ingestion pipeline for the recommendation engine

### 7. YouTube Watcher Module

Primary file:

- `ai-backend/DataCronJob/youtube_watcher_service.py`

Packages used:

- `google-api-python-client`
- `openai`
- `pymongo`
- `dotenv`

Implementation:

- monitors a fixed set of YouTube channels
- fetches videos published in the recent window
- uses OpenAI to classify whether a video is a list/recommendation style video
- skips single-phone reviews or irrelevant content
- stores new relevant videos in MongoDB
- triggers `process_video()` for transcript extraction and phone extraction

Working logic:

1. call YouTube Data API for each channel
2. use `gpt-4o` to check semantic relevance
3. skip duplicates already in MongoDB
4. process relevant new videos into structured phone records

Role in system:

- this module keeps the recommendation database fresh

### 8. Recommendation Engine Runtime Module

Primary file:

- `ai-backend/RecommendationEngine/recommendation_service.py`

Packages used:

- `pymongo`
- `langchain-google-genai`
- `pydantic`

Implementation:

- loads recommendation candidates from MongoDB collection `phones`
- tries to filter by approximate budget
- builds a prompt listing candidate phones and user priority
- asks Gemini to rank and explain matches
- returns formatted recommendation text

Working logic:

1. `/recommend/` receives `max_price` and `priority`
2. MongoDB is queried for candidate phones
3. prompt is sent to Gemini
4. formatted text is returned to frontend

Important code issue:

- the MongoDB query defines `price_range` twice in the same object:
  - once with `$lte`
  - once with `$gte`
- in Python dictionaries, the second key overwrites the first, so only one bound is actually applied
- result: recommendation filtering is looser than intended

### 9. Chatbot Module

Primary files:

- `ai-backend/ChatBot/chatbot.py`
- `ai-backend/ChatBot/crud.py`
- `ai-backend/ChatBot/db.py`

Packages used:

- `langchain-google-genai`
- `pymongo`
- `bson`
- `dotenv`
- `re`

Implementation:

- chatbot is restricted to smartphone-related topics via a system prompt
- detects recommendation-style user messages with keyword matching
- recommendation-like prompts are routed to `get_recommendations()`
- all other prompts go to Gemini chat
- conversations and messages are stored in MongoDB

Working logic:

1. frontend posts chat message to Next.js API.
2. Next.js proxies to FastAPI `/chat`.
3. FastAPI either creates a new conversation or loads history.
4. chatbot generates reply.
5. both user and assistant messages are saved.
6. frontend can fetch conversation history from `/chat/{conversation_id}`.

Strength:

- combines general phone Q&A with recommendation routing

Limitation:

- recommendation intent detection is keyword-based and may miss more complex phrasing

## Frontend Module Analysis

### 1. Supabase Access Layer

Primary files:

- `web/app/lib/supabaseClient.ts`
- `web/app/lib/supabaseAdmin.ts`

Packages used:

- `@supabase/supabase-js`

Implementation:

- `supabaseClient.ts` creates a browser-safe client using public URL and anon key
- `supabaseAdmin.ts` creates a server-side client with service role key and disabled session persistence

Role:

- all auth, DB, and storage interactions in the web app depend on these wrappers

### 2. Phone Listing Creation Flow

Primary files:

- `web/app/add/page.tsx`
- `web/app/api/phones/add/route.ts`

Packages used:

- frontend: React, Supabase client, Next navigation
- API route: Supabase client, Node `fs`

Implementation in page:

- ensures user is logged in
- uploads up to 6 images directly to Supabase Storage bucket `phone-images`
- collects form data for phone details
- submits JSON payload to `/api/phones/add`

Implementation in API route:

- forwards image URLs to FastAPI `/damage-detection/`
- receives:
  - `pdf_path`
  - `condition_score`
- reads the generated PDF from disk
- uploads PDF to Supabase Storage bucket `phone-reports`
- inserts final listing into `mobile_phones`

End-to-end working flow:

1. seller uploads images to Supabase
2. image public URLs are sent to FastAPI
3. FastAPI downloads those URLs, runs AI verification, generates report
4. Next.js uploads report to Supabase
5. listing is inserted into Supabase with report URL and condition score

### 3. Marketplace Module

Primary files:

- `web/app/marketplace/page.tsx`
- `web/app/api/phones/list/route.ts`
- `web/app/components/card/ProductCard.tsx`

Packages used:

- `@tanstack/react-query`
- Radix Select and Slider
- Supabase

Implementation:

- page fetches phone list via React Query from `/api/phones/list`
- API route selects from Supabase table `mobile_phones`
- client applies:
  - search by model/company
  - company filter
  - storage filter
  - price slider
  - pagination

Role:

- this is the main buyer browsing experience

### 4. Product Detail Module

Primary file:

- `web/app/phones/[id]/page.tsx`

Packages used:

- React
- Next navigation and linking
- Supabase client
- Lucide icons

Implementation:

- fetches all phones and selects one by route ID
- fetches seller info from `/api/users/[id]`
- renders image carousel, condition/verification state, report link, seller info
- allows:
  - save action
  - report action
  - start chat with seller

Important code observations:

- page imports `getOrCreateConversation()` but does not use it
- some chat table names are inconsistent:
  - `conversation`
  - `conversations`
- this can affect reliability depending on actual Supabase schema

### 5. Recommendation UI Module

Primary files:

- `web/app/recommendation/page.tsx`
- `web/app/api/phones/recommend/route.ts`

Packages used:

- React
- Radix slider
- Lucide icons

Implementation:

- user chooses budget and priority
- page calls `/api/phones/recommend`
- API route proxies to FastAPI `/recommend/`
- response text is shown in a formatted block

Role:

- this is the user-facing entry point to the recommendation engine

### 6. Auth Module

Representative files:

- `web/app/(auth)/signin/page.tsx`
- `web/app/(auth)/signup/page.tsx`
- `web/app/components/auth/GoogleButton.tsx`
- `web/app/(auth)/callback/page.tsx`

Packages used:

- Supabase
- React

Implementation:

- supports session-based auth checks
- supports Google OAuth via Supabase
- redirects users after login or callback completion

### 7. Real-Time Chat Module

Primary files:

- `web/app/chats/ChatClient.tsx`
- `web/app/components/chat/Inbox.tsx`
- `web/app/components/chat/ChatWindow.tsx`
- `web/hooks/useRealtimeChat.ts`
- `web/app/api/messages/send/route.ts`

Packages used:

- Supabase
- `pusher`
- `pusher-js`
- React

Implementation:

- `ChatClient.tsx` verifies authenticated session and loads inbox/chat panes
- `Inbox.tsx` reads conversations from Supabase and subscribes to Pusher inbox events
- `ChatWindow.tsx` loads messages from Supabase and listens for room events via `useRealtimeChat`
- sending a message:
  - inserts into Supabase `messages`
  - calls `/api/messages/send`
  - server route triggers Pusher events for:
    - active chat room
    - recipient inbox
    - sender inbox

Role:

- this is a separate real-time buyer/seller messaging system on the web side

Important architectural note:

- there are two chat systems in this repository:
  - MongoDB-backed AI assistant chat in `ai-backend/ChatBot`
  - Supabase + Pusher user-to-user chat in `web/app/components/chat`

These are independent modules serving different use cases.

### 8. Admin Module

Primary files:

- `web/app/api/admin/users/route.ts`
- `web/app/api/admin/ads/route.ts`
- `web/app/admin/page.tsx`

Packages used:

- Supabase admin client

Implementation:

- server routes query `profiles` and `mobile_phones`
- intended to support admin dashboard management of users and ads

## End-to-End Data Flows

### A. Seller Listing and AI Verification

1. user signs in via Supabase
2. user uploads up to 6 photos to Supabase Storage
3. frontend sends public image URLs to `/api/phones/add`
4. Next.js route calls FastAPI `/damage-detection/`
5. FastAPI downloads images, runs YOLO, creates PDF, computes score
6. Next.js uploads PDF report to Supabase Storage
7. listing is inserted into `mobile_phones`

### B. Price Prediction

1. user or frontend sends structured phone attributes to `/price-prediction/`
2. FastAPI builds a `UsedMobile` object
3. MongoDB training data is fetched from `used_mobiles`
4. random forest is trained on demand
5. price band is returned

### C. Recommendations

1. YouTube watcher finds relevant videos
2. transcript processor extracts phone summaries and price ranges
3. records are stored in MongoDB `phones`
4. frontend recommendation page calls FastAPI `/recommend/`
5. Gemini ranks candidate phones for the requested priority

### D. AI Assistant Chat

1. frontend sends prompt to `/api/chat`
2. Next.js proxies to FastAPI `/chat`
3. FastAPI reads or creates MongoDB conversation
4. chatbot returns recommendation text or Gemini-generated answer

### E. User-to-User Chat

1. buyer opens seller chat from product page
2. conversation and messages live in Supabase
3. Pusher distributes real-time updates to inbox and active chat window

## Notable Code-Level Findings

These are implementation findings from the current codebase, not assumptions:

1. `ai-backend/RecommendationEngine/recommendation_service.py`
   - budget filter query is incorrect because `price_range` is declared twice in one dictionary
   - only one bound survives

2. `ai-backend/DamageDetection/Damage_Detection.py`
   - `DAMAGE_MEASUREMENT` uses `screen_line` while classes use `line`
   - behavior still works due to fallback path, but naming is inconsistent

3. `ai-backend/main.py`
   - code references `best2.pt`
   - repository listing showed `best3.pt`
   - model filename alignment should be verified

4. `ai-backend/requirements.txt`
   - source imports `cv2`, `matplotlib`, and `langchain_openai`
   - these dependencies are not clearly present in requirements

5. `web` chat-related code
   - table names alternate between `conversation` and `conversations`
   - consistency depends on actual Supabase schema and may cause defects

6. `web/app/api/phones/add/route.ts`
   - FastAPI report path must be readable from the Next.js runtime host filesystem
   - this works for local co-hosted development but becomes fragile if services are separated

## Summary

IntelliFone is implemented as a hybrid marketplace plus AI platform with clear functional separation:

- `web/` handles user workflows, Supabase persistence, auth, and real-time user chat
- `ai-backend/` handles all ML, LLM, and ingestion pipelines
- MongoDB supports dynamic AI features:
  - scraped OLX market data for pricing
  - extracted YouTube phone opinions for recommendations
  - assistant chat history
- Supabase supports application-facing marketplace features:
  - listings
  - users
  - storage
  - buyer/seller messaging

From an implementation perspective, the strongest modules are:

- the listing verification flow
- the layered AI pipeline of damage detection -> scoring -> reporting
- the data-driven price prediction design
- the recommendation ingestion pipeline from YouTube

The main areas that need cleanup are:

- dependency declarations
- model file naming consistency
- Mongo query correctness in recommendations
- chat schema consistency between `conversation` and `conversations`

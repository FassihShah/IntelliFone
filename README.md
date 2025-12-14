# IntelliFone

> **Final Year Project**: A comprehensive web and mobile platform for buying and selling used smartphones with AI-driven verification, damage detection, price prediction, and intelligent recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [AI/ML Backend Details](#aiml-backend-details)
- [Progress Checklist](#progress-checklist)
- [Contributing](#contributing)
- [License](#license)
- [Team](#team)

---


## 🎯 Overview

**IntelliFone** is an intelligent marketplace platform designed specifically for the Pakistani smartphone resale market. It combines cutting-edge AI/ML technologies with a user-friendly interface to solve key problems in the used phone market:

- **Trust Issues**: AI-powered damage detection and condition verification
- **Pricing Uncertainty**: ML-based price prediction using real market data
- **Information Asymmetry**: YouTube review-powered recommendation engine
- **Market Transparency**: Real-time data collection from OLX Pakistan

### Key Innovations

1. **6-Angle Damage Detection**: YOLOv11-based computer vision analyzes phones from all angles
2. **Weighted Condition Scoring**: Intelligent algorithm prioritizes screen damage over cosmetic issues
3. **Pakistan-Specific Pricing**: Random Forest model trained on local market data
4. **LLM-Powered Recommendations**: LLM-powered phone suggestions based on budget and priorities
5. **Sensor Diagnostics Integration**: Hardware verification through mobile app testing

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│           Client Applications                   │
│   ┌──────────────┐      ┌──────────────┐       │
│   │  Next.js Web │      │ React Native │       │
│   │  Application │      │  Mobile App  │       │
│   └──────┬───────┘      └──────┬───────┘       │
└──────────┼─────────────────────┼────────────────┘
           │                     │
┌──────────▼─────────────────────▼────────────────┐
│            API Gateway Layer                    │
│   ┌──────────────┐      ┌──────────────┐       │
│   │  Next.js API │      │  FastAPI AI  │       │
│   │    Routes    │      │   Service    │       │
│   └──────┬───────┘      └──────┬───────┘       │
└──────────┼─────────────────────┼────────────────┘
           │                     │
    ┌──────┼──────┬──────────────┼──────────┐
    │      │      │              │          │
┌───▼──┐ ┌─▼───┐ ┌▼────────┐ ┌──▼──────┐ ┌▼────────┐
│Supa- │ │Fast │ │ MongoDB │ │ Render  │ │External │
│base  │ │ API │ │  Atlas  │ │  Crons  │ │Services │
│      │ │     │ │         │ │         │ │         │
│•Auth │ │•AI  │ │•Scraped │ │•OLX     │ │•EmailJS │
│•DB   │ │•ML  │ │ Data    │ │•YouTube │ │•YouTube │
│•Store│ │     │ │•Reviews │ │ Watcher │ │•OpenAI  │
└──────┘ └─────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## 💻 Technology Stack

### **Frontend (Web)**
| Technology | Purpose | Version |
|------------|---------|---------|
| Next.js | React framework with App Router | 14.x |
| TypeScript | Type-safe development | 5.x |
| Tailwind CSS | Utility-first styling | 3.x |
| React Query | Server state management | 5.x |
| Supabase Client | Authentication & data fetching | Latest |

### **Frontend (Mobile)**
| Technology | Purpose |
|------------|---------|
| React Native | Cross-platform mobile development |
| TypeScript | Type-safe mobile development |
| Expo | Development and build tooling |

### **Backend (Web Services)**
| Technology | Purpose |
|------------|---------|
| Supabase | Backend-as-a-Service (Auth, DB, Storage) |
| PostgreSQL | Primary relational database |
| Supabase Storage | Image and file storage |
| Next.js API Routes | Backend API endpoints |

### **Backend (AI/ML Service)**
| Technology | Purpose | Version |
|------------|---------|---------|
| FastAPI | Python web framework | Latest |
| Python | Primary language | 3.10+ |
| YOLOv11 | Damage detection (Ultralytics) | Latest |
| Scikit-learn | Random Forest price prediction | 1.3+ |
| OpenAI GPT-4o | LLM for data processing | Latest |
| Google Gemini | Alternative LLM | 2.5 Flash |
| MongoDB | NoSQL database for AI data | Latest |
| BeautifulSoup | Web scraping | 4.x |
| Uvicorn | ASGI server | Latest |

### **Third-Party Services**
- **EmailJS**: Contact form email delivery
- **YouTube Data API**: Video and transcript extraction
- **OpenAI API**: Natural language processing
- **Google Gemini API**: Alternative LLM processing

---

## ✨ Features

### **For Sellers**
-  **6-Image Upload System**: Front, back, left, right, top, bottom angles
-  **AI Damage Detection**: Automatic identification of cracks, dots, and lines
-  **Condition Scoring**: 0-20 scale based on weighted damage analysis
-  **Smart Price Suggestion**: ML-based price prediction using market data
-  **Easy Listing Creation**: Simple form with image upload to Supabase Storage
-  **Seller Dashboard**: Manage listings and view buyer inquiries

### **For Buyers**
-  **Advanced Search & Filters**: Search by brand, model, storage, price range
-  **AI Verification Badge**: See which phones have been AI-verified
-  **Detailed Product Pages**: High-quality images, specs, seller info
-  **Smart Recommendations**: Budget and priority-based suggestions
-  **Similar Phones**: AI-powered related product discovery
-  **Direct Communication**: Built-in chat with sellers

### **AI/ML Capabilities**
-  **Damage Detection Module**: YOLOv8 segmentation for 3 damage classes
-  **Condition Scoring Engine**: Weighted algorithm (screen: 1.0, back: 0.6, sides: 0.3)
-  **Price Prediction Model**: Random Forest trained on OLX Pakistan data
-  **Recommendation Engine**: LLM-powered analysis of YouTube tech reviews
-  **Market Data Collection**: Automated OLX scraping with LLM verification

### **Additional Features**
-  **Google OAuth Integration**: Seamless authentication
-  **Contact Form**: EmailJS-powered support system
-  **Responsive Design**: Mobile-first approach with Tailwind CSS
-  **Real-time Search**: Instant marketplace filtering
-  **Modern UI/UX**: Glass-morphism, neon effects, smooth animations

---

## 📁 Project Structure

```
intellifone/
│
├── web/                                    # Next.js Web Application
│   ├── app/
│   │   ├── home/
│   │   │   └── page.tsx                       # Landing page
│   │   ├── marketplace/
│   │   │   └── page.tsx                       # Product listings with filters
│   │   ├── add/
│   │   │   └── page.tsx                       # Sell phone (6-image upload)
│   │   ├── phones/
│   │   │   └── [id]/
│   │   │       └── page.tsx                   # Product detail page
│   │   ├── recommendation/
│   │   │   └── page.tsx                       # AI recommendations
│   │   ├── about/
│   │   │   └── page.tsx                       # About page
│   │   ├── helpcenter/
│   │   │   └── page.tsx                       # Help & FAQs
│   │   ├── termsofservice/
│   │   │   └── page.tsx                       # Terms of service
│   │   ├── privacypolicy/
│   │   │   └── page.tsx                       # Privacy policy
│   │   ├── components/
│   │   │   ├── SearchBar.tsx                  # Search component
│   │   │   ├── card/
│   │   │   │   └── ProductCard.tsx            # Phone listing card
│   │   │   └── auth/
│   │   │       └── GoogleButton.tsx           # Google OAuth button
│   │   └── api/
│   │       ├── phones/
│   │       │   ├── list/
│   │       │   │   └── route.ts               # GET all phones
│   │       │   ├── add/
│   │       │   │   └── route.ts               # POST new phone
│   │       │   └── recommend/
│   │       │       └── route.ts               # GET recommendations (proxy to FastAPI)
│   │       └── users/
│   │           └── list/
│   │               └── route.ts               # GET all users
│   ├── lib/
│   │   ├── supabaseClient.ts                  # Client-side Supabase setup
│   │   └── supabaseAdmin.ts                   # Server-side Supabase (service role)
│   ├── providers/
│   │   └── ClientProvider.tsx                 # React Query provider
│   ├── public/                                # Static assets
│   ├── styles/                                # Global styles
│   ├── .env.local                             # Environment variables
│   ├── next.config.js                         # Next.js configuration
│   ├── tailwind.config.js                     # Tailwind CSS configuration
│   ├── tsconfig.json                          # TypeScript configuration
│   └── package.json                           # Dependencies
│
├── ai-backend/                                # FastAPI AI/ML Service
│   ├── ConditionScoring/
│   │   └── condition_scoring.py               # Condition Scoring algorithm
│   ├── DamageDetection/
│   │   ├── Damage_Detection_Training.ipynb    # Model training notebook
│   │   ├── Damage_Detection.py                # Damage Detection Service
│   │   └── best.pt                            # model weights
│   ├── DataCronJob/
│   │   ├── cron_scraper.py                    # Scheduled OLX data collection
│   │   ├── olx_scraper_service.py             # OLX page scraping logic
│   │   ├── recommender_data_service.py        # Process YouTube review data
│   │   └── youtube_watcher_service.py         # Monitor YouTube channels
│   ├── PricePrediction/
│   │   └── predict_price_service.py           # Random Forest price prediction
│   ├── RecommendationEngine/
│   │   └── recommendation_service.py          # LLM-powered recommendations
│   ├── app.py                                 # Main FastAPI application
│   ├── models.py                              # Shared data models
│   ├── .env                                   # Environment variables (API keys, DB)
│   ├── .gitignore                             # Git ignore rules
│   ├── README.md                              # AI/ML backend documentation
│   └── requirements.txt                       # Python dependencies
│
└── README.md                                  # Main project documentation

```

---

## 📋 Prerequisites

### **System Requirements**
- **Node.js**: v18.x or higher
- **Python**: 3.10 or higher
- **npm** or **yarn**: Latest version
- **MongoDB**: v6.x or higher (for AI backend)
- **Git**: For version control

### **Required Accounts**
1. **Supabase Account**: [supabase.com](https://supabase.com)
   - Create a new project
   - Enable Authentication (Google OAuth)
   - Create storage bucket: `phone-images`
   
2. **OpenAI Account**: [platform.openai.com](https://platform.openai.com)
   - Generate API key for GPT-4o access
   
3. **Google Cloud Account**: [console.cloud.google.com](https://console.cloud.google.com)
   - Enable YouTube Data API v3
   - Enable Gemini API (optional)

4. **MongoDB Atlas**: [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
   - Create free cluster
   - Get connection string

---

## 🚀 Installation & Setup

### **1. Clone the Repository**

```bash
git clone https://github.com/FassihShah/IntelliFone.git
cd IntelliFone
```

### **2. Web Application Setup**

```bash
# Navigate to web directory
cd fyp-web

# Install dependencies
npm install
# or
yarn install

# Create environment file
cp .env.example .env.local

# Edit .env.local with your credentials (see Environment Configuration section)
```

### **3. AI/ML Backend Setup**

```bash
# Navigate to AI backend directory
cd ../ai-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (if not included)
# Place yolov8_damage.pt in models/ directory

# Create environment file
cp .env.example .env

# Edit .env with your credentials (see Environment Configuration section)
```

---

## 🔐 Environment Configuration

### Web Application (.env.local)
```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# EmailJS
NEXT_PUBLIC_EMAILJS_SERVICE_ID=your_service_id
NEXT_PUBLIC_EMAILJS_TEMPLATE_ID=your_template_id
NEXT_PUBLIC_EMAILJS_PUBLIC_KEY=your_public_key

# AI Backend
NEXT_PUBLIC_AI_BACKEND_URL=http://localhost:8000
```

### AI Backend (.env)
```env
# OpenAI
OPENAI_API_KEY=sk-your_openai_key

# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/intellifone

# YouTube
YOUTUBE_API_KEY=your_youtube_api_key

# Optional
GEMINI_API_KEY=your_gemini_key
```

---

## 🎮 Running the Application

### **Development Mode**

#### **Start Web Application**

```bash
cd fyp-web
npm run dev
# or
yarn dev

# Access at: http://localhost:3000
```

#### **Start AI/ML Backend**

```bash
cd ai-backend

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Run FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Access API docs at: http://localhost:8000/docs
```

#### **Start Scraping Services (Optional)**

```bash
cd ai-backend

# Run OLX scraper cron job
python cron_scraper.py

# Run YouTube watcher
python youtube_watcher_service.py
```

### **Production Mode**

#### **Web Application**

```bash
cd fyp-web

# Build for production
npm run build

# Start production server
npm run start
```

#### **AI/ML Backend**

```bash
cd ai-backend

# Run with production settings
gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📡 API Endpoints

### Web API (Next.js)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/phones/list` | GET | Get all phone listings |
| `/api/phones/add` | POST | Create new listing |
| `/api/phones/recommend` | GET | Get recommendations |
| `/api/users/list` | GET | Get all users |

### AI/ML API (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/damage-detection` | POST | Detect damages from images |
| `/condition-scoring` | POST | Calculate condition score |
| `/price-prediction` | POST | Predict price range |
| `/full-verification` | POST | Complete verification pipeline |
| `/recommend` | GET | Get phone recommendations |

---

## 🤖 AI/ML Backend Details

### **Module 1: Damage Detection**

**Technology:** YOLOv8 Segmentation Model

**Process:**
1. Receives 6 images (front, back, left, right, top, bottom)
2. Runs YOLOv8 inference on each image
3. Detects 3 damage classes:
   - **Crack**: Significant screen/body cracks
   - **Dot**: Small impact points
   - **Line**: Minor scratches or hairline cracks
4. Measures damage size:
   - Cracks/Lines: Length in pixels
   - Dots: Area in pixels²

**Output Format:**
```python
{
  "damages": {
    "front": {
      "crack": [{"length_px": 345.6, "bbox": [...]}],
      "dot": [{"area_px": 1158.8, "bbox": [...]}]
    },
    # ... other sides
  }
}
```

**Key Files:**
- `Damage_Detection.py`: Main detection logic
- `models/yolov8_damage.pt`: Trained model weights

---

### **Module 2: Condition Scoring**

**Algorithm:** Weighted Severity Scoring

**Side Weights:**
- **Front (Screen)**: 1.0 (highest priority)
- **Back**: 0.6
- **Sides (L/R/T/B)**: 0.3 (cosmetic)

**Severity Weights:**
- **Crack**: 3.0 (major)
- **Line**: 1.5 (medium)
- **Dot**: 0.8 (minor)

**Key Files:**
- `condition_scoring.py`: Scoring algorithm implementation

---

### **Module 3: Price Prediction**

**Model:** Random Forest Regressor

**Features (18 total):**
1. RAM (encoded)
2. Storage (encoded)
3. Condition Score (0-20)
4. PTA Approved (boolean)
5. Screen Crack (boolean)
6. Panel Dot (boolean)
7. Panel Line (boolean)
8-17. Sensor Status (camera, wifi, bluetooth, etc.)
18. Market Price (per model)

**Training Data:**
- Source: OLX Pakistan scraped listings
- Update Frequency: Weekly via cron job
- TTL: 60 days

**Price Band Calculation:**
```python
IQR = Q3 - Q1
uncertainty = IQR / median_price
price_min = predicted - (uncertainty × predicted)
price_max = predicted + (uncertainty × predicted)
```

**Key Files:**
- `predict_price_service.py`: Prediction service
- `cron_scraper.py`: Data collection

---

### **Module 4: Market Data Collection**

**Process Flow:**

1. **OLX Scraping**
   - Target: OLX Pakistan mobile listings
   - Batch size: 10 pages per run
   - Frequency: Daily via cron

2. **LLM Verification**
   - Uses GPT-4o or Gemini
   - Verifies brand/model accuracy
   - Extracts structured data

3. **Data Storage**
   - MongoDB with 60-day TTL
   - Indexed by brand/model and ads id
   - Used for model training

**Key Files:**
- `olx_scraper_service.py`: Page-level scraping
- `cron_scraper.py`: Batch scheduler

---

### **Module 5: Recommendation Engine**

**Pipeline:**

1. **YouTube Monitoring**
   - Tracks 4-5 Pakistani tech YouTubers
   - Detects new phone review videos
   - Checks for "top phones" / "best phones" format

2. **Transcript Extraction**
   - YouTube Data API for captions
   - Whisper fallback for non-captioned videos
   - Language: English & Urdu support

3. **LLM Processing**
   - GPT-4o analyzes transcript
   - Extracts:
     - Phone names
     - Price ranges (PKR)
     - Pros & Cons
     - Reviewer opinions
     - Video timestamps

4. **Data Storage**
   - MongoDB collection: `youtube_reviews`
   - Indexed by phone name
   - Includes video links and sources

5. **Query Matching**
   - User provides budget + priority
   - LLM ranks phones based on:
     - Price fit
     - Priority match (camera/gaming/battery)
     - Reviewer consensus
   - Returns formatted recommendations

**Key Files:**
- `youtube_watcher_service.py`: Video monitoring
- `recommender_data_service.py`: Transcript processing
- `recommendation_service.py`: Query matching

---

### **Cron Jobs & Automation**

**Scheduled Tasks:**

1. **OLX Data Collection**
   - Frequency: Daily at 2:00 AM
   - Batch size: 100 listings
   - Purpose: Keep price model trained on fresh data

2. **YouTube Monitoring**
   - Frequency: Every week
   - Checks: New videos from tracked channels
   - Purpose: Update recommendation database


---

## Progress Checklist


### 🖥️ Web Application (Frontend)

#### Completed
- [x] Landing page  
- [x] Marketplace listing page  
- [x] Search & filtering  
- [x] Phone detail pages  
- [x] Sell phone interface  
- [x] Image upload UI  
- [x] AI recommendation display  
- [x] Google OAuth authentication  
- [x] User profile management  
- [x] Contact Us page  
- [x] About, Terms & Privacy pages  
- [x] Fully responsive design (Tailwind CSS)  

#### Remaining
- [ ] Final backend integration testing  
- [ ] UI polishing & validation handling  

---

### 📱 Mobile Application (Frontend)

#### Completed
- [x] Complete mobile UI  
- [x] Sell phone flow UI  
- [x] Image upload screens  
- [x] Recommendation screens  
- [x] Authentication & profile UI  

#### Remaining
- [ ] Backend API integration  
- [ ] Sensor diagnostics integration  
- [ ] Final testing on real devices  

---

### ⚙️ Backend (Web & API)

#### Completed
- [x] Core backend architecture  
- [x] Supabase authentication  
- [x] Marketplace APIs  
- [x] User profile APIs  
- [x] Image storage (Supabase Storage)  
- [x] Frontend ↔ backend integration (mostly complete)  

#### Remaining
- [ ] Mobile app API integration  
- [ ] Final API testing & optimization  

---

### 🧠 AI / ML Backend (Core Project Contribution)

#### Completed
- [x] Custom dataset collection (no public dataset available)  
- [x] Manual data annotation (Roboflow)  
- [x] YOLO-based damage detection model  
- [x] Multi-side phone image analysis  
- [x] Weighted condition scoring algorithm (0–20)  
- [x] AI-based damage flags extraction  
- [x] Random Forest price prediction model  
- [x] Condition score integration into price prediction  
- [x] Market-aware dynamic price range (no hardcoding)  
- [x] LLM-based recommendation engine  
- [x] FastAPI-based AI services  
- [x] MongoDB integration  
- [x] Continuous improvement pipeline  

### Ongoing
- [ ] Dataset expansion  
- [ ] Annotation refinement  
- [ ] Damage detection accuracy improvement  

---

### 🕷️ Data Collection & Scraping

#### Completed
- [x] OLX scraper for used mobile listings  
- [x] Brand & model–based scraping  
- [x] LLM-based model verification  
- [x] Condition & price normalization  
- [x] MongoDB storage with TTL indexing  
- [x] Cron-based scraping (Render)  
- [x] YouTube review watcher  
- [x] Transcript extraction for recommendations  

#### Known Limitations
- [ ] OLX rate limiting optimization  

---

### 🗄️ Databases

#### Completed
- [x] PostgreSQL schema (Supabase)  
- [x] MongoDB collections for AI/ML data  
- [x] Indexes & TTL policies  
- [x] Secure storage configuration  

---

### ☁️ Deployment & Infrastructure

#### Completed
- [x] AI services deployed on Render  
- [x] Cron jobs running on Render  
- [x] Environment variable management  

#### Planned
- [ ] Full AWS deployment  
- [ ] Docker-based production setup  

---

### ⚠️ Known Technical Limitations

- [ ] Damage detection still improving with more data  
- [ ] Price prediction less accurate for rare models  
- [ ] OLX scraper needs rate-limit optimization  
- [ ] Mobile sensor diagnostics not fully integrated  


---

## 🤝 Contributing

1. **Fork the repository**
```bash
git clone https://github.com/FassihShah/IntelliFone.git
cd IntelliFone
git checkout -b feature/your-feature-name
```

2. **Make changes and test**
```bash
# Web app
cd fyp-web
npm run dev
npm run test

# AI backend
cd ai-backend
pytest tests/
```

3. **Commit with conventional commits**
```bash
git commit -m "feat: add new damage detection class"
```

4. **Push and create pull request**
```bash
git push origin feature/your-feature-name
```
---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**IntelliFone Development Team**
- Final Year Project - 2025
- Software Engineering Department
- PUCIT

---



---

**Built with ❤️ for the Pakistani smartphone market**

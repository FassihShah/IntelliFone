

# Intellifone — FYP Web App

Intellifone is a Next.js TypeScript web application developed as a final year project. It provides a marketplace for phones, AI verification, price prediction, and user-facing features like contact support and recommendations. This README documents how to set up, run, and extend the project.

---

## Quick Summary

- **Purpose:** Marketplace + AI verification and price prediction for phones.
- **Tech Stack:** Next.js (App Router), TypeScript, Tailwind CSS, Supabase, EmailJS, React.
- **Where to Look:** Core frontend lives in `fyp-web/app` and API routes in `fyp-web/app/api`.

---

## Prerequisites

- **Node.js:** v16+ recommended. Install from [nodejs.org](https://nodejs.org/).  
- **Package manager:** `npm` (commands below use `npm`)  
- **Supabase account:** For DB/auth/storage (optional but recommended)

---

## Installation

1. Open terminal and navigate to the web app folder:

```bash
cd fyp-web
````

2. Install dependencies:

```bash
npm install
```

---

## Environment Variables

Create a `.env.local` file in `fyp-web/` with the following keys:

```text
NEXT_PUBLIC_EMAILJS_SERVICE_ID=your_service_id
NEXT_PUBLIC_EMAILJS_TEMPLATE_ID=your_template_id
NEXT_PUBLIC_EMAILJS_PUBLIC_KEY=your_public_key
NEXT_PUBLIC_SUPABASE_URL=https://xyz.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=public-anon-key
SUPABASE_SERVICE_ROLE_KEY=service-role-secret
```

> **Security:** Do not commit `.env.local`. Keep `SUPABASE_SERVICE_ROLE_KEY` server-side only.

---

## Run (Development)

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

---

## Build & Start (Production)

```bash
npm run build
npm run start
```

---

## Project Structure

```
fyp-web/
├─ app/
│  ├─ home/page.tsx                # Home/landing page
│  ├─ marketplace/page.tsx         # Marketplace list & filters
│  ├─ add/page.tsx                 # Sell/add phone page
│  ├─ phones/[id]/page.tsx         # Product detail page
│  ├─ recommendation/page.tsx      # AI recommendations
│  ├─ contactus/page.tsx           # Contact form
│  ├─ helpcenter/page.tsx          # Help & FAQ
│  ├─ about/page.tsx               # About / mission
│  ├─ termsofservice/page.tsx      # Terms of service
│  ├─ privacypolicy/page.tsx       # Privacy policy
│  ├─ components/                  # Shared UI components
│     ├─ SearchBar.tsx
│     ├─ card/ProductCard.tsx
│     ├─ auth/GoogleButton.tsx
├─ app/api/
│  ├─ phones/list/route.ts         # GET all phones
│  ├─ phones/add/route.ts          # POST new phone
│  ├─ phones/recommend/route.ts    # GET AI recommendations
│  ├─ users/list/route.ts          # GET users
│  └─ supabase/functions/send-contact-email/index.ts
├─ app/lib/
│  ├─ supabaseClient.ts            # Client-side Supabase
│  ├─ supabaseAdmin.ts             # Server-side Supabase
├─ uploads/                        # Temporary uploaded images
├─ outputs/                        # YOLO annotated images
├─ reports/                        # Generated PDF reports
├─ best2.pt                         # YOLO model
├─ package.json                     # Dependencies & scripts
└─ .env.local                       # Environment variables (local only)
```

---

## Pages & Features

### Home & Landing Page

* Hero section, featured carousel, CTA, navigation links

### Marketplace

* Phone listing with search, company/storage filters, price slider, pagination
* Product grid using `ProductCard`

### Product Detail Page

* Loads phone by ID
* Image gallery, AI-verified badge, seller info, buy/chat/report actions
* Similar phones suggestions

### Sell / Add Phone Page

* Client-side auth check
* Uploads up to 6 images to Supabase Storage
* Form for phone details
* Submits data to `/api/phones/add`

### AI Recommendation Page

* Budget slider & priority selection
* Calls `/api/phones/recommend` (proxies FastAPI service)
* Displays AI-based phone suggestions

### Static / Informational Pages

* About / Mission, Help & FAQs, Terms of Service, Privacy Policy

### Contact Form

* Sends emails via EmailJS
* Shows success state

---

## API Routes & Integrations

| Route                                    | Description                               | Status        |
| ---------------------------------------- | ----------------------------------------- | ------------- |
| `/api/phones/list`                       | GET all phones from `mobile_phones` table | ✅ Implemented |
| `/api/phones/add`                        | POST phone listing with images            | ✅ Implemented |
| `/api/phones/recommend`                  | GET recommendations (calls FastAPI)       | ✅ Implemented |
| `/api/users/list`                        | GET users                                 | ✅ Implemented |
| `/supabase/functions/send-contact-email` | Sends email from contact form             | ✅ Implemented |

---

## Auth & Third-Party Integration

* Supabase authentication (sign in/out, auth state check)
* Google OAuth via `GoogleButton`
* Email delivery via EmailJS
* AI recommendations require a local FastAPI service

---

## UI Components

* **ProductCard**: Shows images, price, PTA/verified badges, links
* **SearchBar**: Navigates to marketplace with query
* **GoogleButton**: OAuth login
* Tailwind classes used extensively (`glass-panel`, `yellow-btn`, `neon-glow`)

---

## Testing & Linting

* Use `npm run lint` or scripts in `package.json`

---

## Deployment

* Recommended: Vercel
* Add environment variables in Vercel dashboard
* Keep service role key server-side only

---

## Development Tips

* Restart `npm run dev` after Tailwind changes
* Avoid OneDrive/long path issues on Windows
* Keep secrets in `.env.local` only

---

## Contributing / Next Steps

* Add unit and integration tests
* Harden server-side validation
* Add CI for lint/tests and secret scanning

---

```

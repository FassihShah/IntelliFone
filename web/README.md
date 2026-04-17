# IntelliFone Web App

This is the Next.js frontend for IntelliFone. It handles the marketplace UI, authentication-facing flows, seller listing forms, buyer browsing, recommendations UI, and calls into the FastAPI AI backend when AI features are needed.

## Tech Stack

- Next.js
- TypeScript
- Tailwind CSS
- Supabase
- React

## Local Setup

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

## Environment Variables

Create `web/.env.local` and populate the keys used by the frontend:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

NEXT_PUBLIC_AI_BACKEND_URL=http://localhost:8000

NEXT_PUBLIC_EMAILJS_SERVICE_ID=your_emailjs_service_id
NEXT_PUBLIC_EMAILJS_TEMPLATE_ID=your_emailjs_template_id
NEXT_PUBLIC_EMAILJS_PUBLIC_KEY=your_emailjs_public_key
```

Keep service-role secrets server-side only. Do not commit `.env.local`.

## Project Layout

```text
web/
  app/                  Next.js app routes and API routes
  components/           shared UI components
  hooks/                reusable frontend hooks
  lib/                  Supabase and utility modules
  public/               static assets
  package.json
```

## Main Responsibilities

- User-facing marketplace pages
- Add-phone/seller flow
- Recommendation UI
- Supabase auth and marketplace data access
- Frontend API routes that proxy or coordinate backend calls

The AI/ML logic does not live here. It lives in `../ai-backend/`.

## AI Backend Connection

The web app should call the FastAPI backend through the configured backend URL:

```env
NEXT_PUBLIC_AI_BACKEND_URL=http://localhost:8000
```

In deployment, set this to your deployed FastAPI URL.

The backend must also allow the frontend origin through `ALLOWED_ORIGINS` in `ai-backend/.env`, for example:

```env
ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
```

## Build

```bash
npm run build
npm run start
```

## Deployment

Vercel is the simplest deployment target for the frontend.

Deploy the FastAPI AI backend separately, then set `NEXT_PUBLIC_AI_BACKEND_URL` to that backend URL.

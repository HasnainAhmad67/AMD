# Deploy Telltale on Vercel

This repository is configured as one Vercel project: the Python FastAPI API runs as a Vercel Function, and the React/Vite dashboard is built and served as its frontend. The repository root must remain the Vercel **Root Directory**; do not set it to `apps/web`.

## Deploy from a new GitHub repository

1. Create a new GitHub repository under your own account, with a fresh name such as `telltale-amd-hackathon-2026`. Upload the contents of this project ZIP to the repository root. Do not upload a containing folder around the project files.
2. In Vercel, choose **Add New → Project**, import that new repository, and keep the Root Directory as `./`.
3. Let `vercel.json` supply the build command and keep the Output Directory unset. The build runs `npm ci` for `apps/web` and builds the React app; Vercel installs the lean Python inference dependencies from the root `requirements.txt`. `uv.lock` remains for reproducible local development.
4. Deploy. The frontend and API share one Vercel domain; React Router deep links use the Vercel FastAPI frontend fallback.

The ZIP contains no `.git` history or Git remote. It is ready to upload as a clean, new repository under your account.

## Demo mode and data

The Vercel entrypoint enables the project's built-in demo mode and offline sensor bootstrap so the demo can start without an AMD API key or a training-data download. `demo-token-123` is intentionally a public demo token embedded in the browser build; anyone who can access the deployment can call demo-only assessment/AI routes. This configuration is for a public hackathon demonstration only. Do not use it for private data or production authentication.

Vercel Function storage is ephemeral. The deployment writes its temporary application data under `/tmp`; changes and generated state are not a durable database and can be lost when an instance is recycled. The local Docker setup remains the supported choice for a persistent simulator/database or AMD GPU inference.

## Local checks

```bash
# Frontend
cd apps/web
npm ci
npm run build
cd ../..

# API tests (requires uv)
uv sync
uv run pytest apps/api/tests -q
```

The web build requires Node.js. For local full-stack development, continue to use the instructions in `README.md` and `apps/web/README.md`.

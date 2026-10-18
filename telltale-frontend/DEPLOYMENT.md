# Telltale split deployment

Deploy the API first, then build the frontend with that API URL. This split avoids putting the ML/audio runtime in a Vercel Function. The backend package keeps the sensor, fan acoustic, and bearing inference models and uses the existing Docker runtime dependencies.

## 1. Deploy the backend on Render

Create a new GitHub repository for the `backend/` folder only, or upload that folder's contents to the root of a new repository. In Render, create a **New → Blueprint** from the repository and let it read `render.yaml`; alternatively create a **Web Service → Docker** and set the Dockerfile and Docker context to `./Dockerfile` and `.`. The included `render.yaml` uses Docker and health-checks `/api/v1/health`.

After deployment, open `https://YOUR-SERVICE.onrender.com/api/v1/health`. Wait for JSON with HTTP 200 before continuing. On a free/idle service, the first request may take a little time while the container wakes up.

The backend is configured as a public demo: demo-mode auth is enabled, the template AI provider is selected, and CORS allows any browser origin. Do not send private data or use the demo token as production authentication. The service writes state under `/tmp`, which is not persistent storage.

## 2. Deploy the frontend on Vercel

Create a separate repository from the contents of the `frontend/` folder, import it into Vercel, and set **Root Directory** to `./`. Vercel should detect Vite. The included `vercel.json` builds with `npm ci && npm run build`, outputs `dist`, and rewrites React Router routes to `index.html`.

Before deploying, add these Vercel environment variables for Production and Preview:

- `VITE_API_BASE_URL` = the backend origin, e.g. `https://YOUR-SERVICE.onrender.com` (no trailing slash and no `/api/v1` suffix).
- `VITE_API_TOKEN` = `demo-token-123` (public demo token; it is embedded in client JavaScript).

Vite injects these variables at build time, so changing either variable requires a new deployment. Do not paste an AMD key into the frontend project. The included split backend defaults to template AI and does not need the AMD API key.

## API URL behavior

The frontend requests `VITE_API_BASE_URL + /api/v1/...`, including audio sample downloads. The backend's CORS setting permits the Vercel origin. The public API remains demo-only and should not be used with sensitive data.

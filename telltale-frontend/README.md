# Telltale frontend

This is the Vite/React dashboard package for a separate Vercel project. It calls the backend at build-time-configured `VITE_API_BASE_URL`.

## Vercel setup

Create a new GitHub repository with the contents of this folder at its root, import it into Vercel, and leave **Root Directory** as `./`. Vercel detects Vite. The included `vercel.json` handles the production build, `dist` output, and React Router fallback.

Before deploying, add these environment variables in Vercel for Production and Preview:

- `VITE_API_BASE_URL`: the backend origin, e.g. `https://YOUR-SERVICE.onrender.com`, with no trailing slash or `/api/v1`.
- `VITE_API_TOKEN`: `demo-token-123` for the public demo. It is embedded into client JavaScript, so it is not a secret.

Deploy after setting both variables. If you change either variable, redeploy because Vite injects them during build. The backend CORS must permit the Vercel domain.

For local development, `VITE_API_PROXY` remains available in `.env.example`.

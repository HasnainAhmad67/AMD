# Deploy Telltale backend on Vercel

This backend package has a root `index.py` FastAPI entrypoint. It includes the API source, ML inference source, and trained sensor, fan-acoustic, and bearing models. It is separate from the Vite frontend project.

## Import settings

Upload the contents of this ZIP's `telltale-vercel-backend/` directory to the root of your backend GitHub repository. In Vercel, import that repo as a new project and use:

- **Root Directory:** `./` if repository root contains `index.py`, `vercel.json`, `requirements.txt`, `ml/`, `apps/`, and `models/`.
- **Framework Preset:** Python or FastAPI, if shown.
- **Install Command:** `pip install -r requirements.txt`.
- **Build Command:** leave empty/default.
- **Output Directory:** leave empty.

If you keep the files in a subfolder instead of putting them at repository root, set Root Directory to that exact folder. Do not choose `apps/api` as the root; the root `index.py` needs sibling `ml/` and `models/`.

## Enable the large Python function

The Python ML runtime is larger than Vercel's standard 500 MB function limit. In this backend Vercel project open **Settings → Environment Variables** and add:

- Key: `VERCEL_SUPPORT_LARGE_FUNCTIONS`
- Value: `1`
- Environments: Production and Preview

Save, then redeploy. Vercel's current docs say the large-functions beta supports Python bundles up to 5 GB; it must be enabled for existing projects via this variable. The project must also have Fluid Compute enabled. If this option is unavailable on the account/project, do not keep retrying: this backend needs a host that supports the required bundle size.

## Backend environment variables

The root entrypoint defaults to public-demo settings: models under `./models`, temporary data under `/tmp/telltale-data`, offline sensor setup, demo-mode assessment auth, template AI, and permissive CORS (`["*"]`). These defaults are intentionally suitable only for a public demo. Do not use private/sensitive data. If `TELLTALE_AI_PROVIDER=amd` is already set, it still requires a valid AMD API key; the template provider is the no-key default when that variable is unset.

After deployment, verify `https://YOUR-BACKEND.vercel.app/api/v1/health` returns HTTP 200 JSON. Set the frontend's `VITE_API_BASE_URL` to `https://YOUR-BACKEND.vercel.app` without the `/api/v1` suffix, and redeploy the frontend.

Function storage is ephemeral. Persistent fleet/database state and long-running simulator jobs are not guaranteed on serverless functions.

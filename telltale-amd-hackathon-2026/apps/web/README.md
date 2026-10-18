# apps/web: the Telltale dashboard

React 18 + TypeScript + Vite, with TanStack Query for data fetching and Recharts for charts. No UI framework: all styling is in `src/styles.css` (colours are variables at the top).

## Run it against a backend

```
cd apps/web
npm install
cp .env.example .env      # then edit it (see below)
npm run dev               # http://localhost:5173
```

`.env` (restart `npm run dev` after editing):

| Variable | Meaning |
|---|---|
| `VITE_API_PROXY` | Where the dev server forwards `/api` (your backend), e.g. `http://127.0.0.1:8100`. Docker dev sets it to `http://api:8000`. |
| `VITE_API_TOKEN` | Bearer token for `/api/v1/assessments` and `/api/v1/ai/*`. With `TELLTALE_DEMO_MODE=1` on the backend it is `demo-token-123`. It ships inside the browser bundle, so it is a demo token only. |

The app always calls relative `/api/v1/...` URLs. In dev Vite proxies them (no CORS setup needed); in production nginx does (see `nginx.conf`).

## Pages and the endpoints they use

| Page | Route | Endpoints |
|---|---|---|
| Fleet overview | `/` | `GET /machines`, `GET /machines/overview` (polled every 3 s) |
| Machine detail | `/machines/:id` | `GET /machines/{id}`, `GET /sensors/schema`, `POST /machines/{id}/maintenance` |
| Alerts | `/alerts` | `GET /alerts`, `POST /alerts/{id}/acknowledge`, `POST /alerts/{id}/resolve` |
| New assessment | `/assess` | `GET /machines`, `GET /acoustic/samples`, `POST /assessments` (sensor only), `/assessments/acoustic`, `/assessments/bearing`, `/assessments/multimodal` (2+ inputs), `GET /assessments` |
| Assessment result + AI | `/assessments/:id` | `GET /assessments/{id}`, `POST /ai/explain`, `/ai/analyze`, `/ai/evidence-review`, `/ai/ask`, `/ai/report`, `GET /ai/outputs/{id}` |
| Settings | `/settings` | `GET /health` |

Flow: create an assessment, the backend returns its `assessment_id`, the page opens at `/assessments/<id>`, and every AI call uses that id. Each AI result shows its `provider`, `model` and `validation_status` (`validated`, `fallback` or `refused`).

## Where things are

```
src/
├── api/            client.ts (every backend call), types.ts (mirrors the backend schemas), hooks.ts (react-query)
├── components/     Shell, AiPanel, AssessmentView, DataViz (waveform, spectrogram), RiskRing, ...
├── pages/          one file per page
├── lib/            format helpers, count-up and typewriter hooks, toast
└── styles.css      all styles
```

When the backend schemas change, update `src/api/types.ts` and `src/api/client.ts` first, then the components that use them.

## Notes

- Waveform, spectrogram and anomaly-score charts are drawn from the arrays in the stored acoustic output. The backend keeps only a hash of uploaded audio, never the audio itself.
- The ring on each machine shows the model's failure probability. The backend does not produce a health score, so none is invented.
- Settings has a Calm mode switch that turns off animations and 3D effects (useful on slow projectors).

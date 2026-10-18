# AGENTS.md

Guide for AI coding agents (Codex, Claude Code, Cursor, Qwen Code, Kimi, DeepSeek, Grok, Gemini, Copilot, …) and humans working on **Telltale**.

> **Who uses this repo:** university students, many with no experience of Docker, terminals or monorepos. Their entire workflow is the `telltale` helper (`.\telltale <cmd>` on Windows, `bash telltale.sh <cmd>` on Mac/Linux) and `START_HERE.md`. When you help one of them, give commands in that form, not raw `docker compose`/`uv`/`make`.

Read this file fully before changing code. It is the source of truth for structure, conventions and the "where do I change X" recipes. If the code and this file disagree, fix whichever is wrong in the same change.

---

## 1. What this project is

Telltale is a predictive-maintenance app for an AMD hackathon. It has two halves that share one dashboard:

1. **Telemetry → failure risk.** LightGBM models trained on the AI4I 2020 schema (temperatures, speed, torque, tool wear) predict P(failure) and four failure modes (TWF, HDF, PWF, OSF). Each prediction is explained with exact SHAP values and comes with a what-if analysis.
2. **Sound → anomaly and fault.** Anomaly detector v2 ("ensemble"): a small CNN trained with an ArcFace loss to tell recording conditions apart, plus classical average spectra. Both are scored by nearest-neighbour distance to banks of normal clips across all sections, and conformal p-values say how unusual a clip is. Trained in PyTorch (ROCm or CPU), served via ONNX. The legacy autoencoder stays available (`method: autoencoder`). A LightGBM classifier with envelope-spectrum features, microphone augmentation and conformal prediction sets labels bearing faults (Mendeley bearing dataset).

A FastAPI service loads the trained artifacts and runs a simulated fleet that streams telemetry. It raises alerts and maps outputs to inspection steps through a YAML playbook. A React dashboard shows all of it.

## 2. Golden rules (do not break these)

1. **One feature implementation.** Feature engineering lives only in `ml/src/telltale_ml/sensor/features.py` and `ml/src/telltale_ml/acoustic/features.py` (plus `acoustic/fault.py` for bearing features). Training and serving both import these functions. Never re-implement features in the API or the frontend.
2. **The API never imports torch.** Inference code (`sensor/predictor.py`, `acoustic/detector.py`, `acoustic/fault.py`) must stay torch-free. Torch is used only in `acoustic/model.py` and `acoustic/train_anomaly.py`. Sound models are served through ONNX Runtime.
3. **Missing models never crash the API.** `ModelRegistry` loads what exists. Routes call `registry.require_*()`, which returns HTTP 503 with a hint saying how to train the model. Keep this behaviour for any new model.
4. **No pickles.** LightGBM boosters are saved as `.txt` and the autoencoder as `.onnx`. Every model directory has a `metadata.json` with its features/config, thresholds, test metrics and `caveats`.
5. **Never tune on test data.** Thresholds, conformal calibration and the failure-mode reliability gate come from held-out training data: max-F1 on validation for sensor models, a calibration hold-out of normal clips for sound, out-of-fold predictions for bearings. Test splits are only for reported metrics. Ensemble weights are fixed at 0.5/0.5 up front for the same reason. Sound metrics are measured **without the section label** (`section_known: false`), because that's how uploads are scored. Don't report section-known numbers as deployment performance.
6. **Keep contracts in sync.** If you change a response shape, update all three in the same change:
   `apps/api/telltale_api/schemas/*.py` ⇄ `apps/web/src/api/types.ts` ⇄ the dict built in `telltale_ml` / services.
7. **The LLM only rephrases.** `services/narrator.py` may turn facts into text. It must never compute or change risk, status or recommendations, and the template fallback must keep working with no LLM.
8. **Be honest in the UI.** Keep the caveats (synthetic AI4I data, "anomaly ≠ specific fault", split strategy) visible on the Models page and in `metadata.json`.
9. **No secrets, data or model files in git.** `data/` and `models/` are git-ignored, apart from `.gitkeep` and `README.md` files.
10. **Beginners first.** `.\telltale start` on a fresh clone must work with only Docker Desktop installed. That means no manual training step (the API auto-trains the sensor model, see `services/bootstrap.py`), automatic free-port selection, and every error message saying *what to do next* in plain words. See section 6.12 before touching the helper scripts.

## 3. Repository map

```
apps/api/telltale_api/
  main.py                 create_app(settings) + lifespan: builds AppState, starts the simulator task
  core/config.py          Settings (env prefix TELLTALE_), get_settings()
  api/deps.py             AppState dataclass + get_state() dependency
  api/router.py           mounts all routers under /api/v1
  api/routes/
    health.py             GET /health
    models.py             GET /models, POST /models/reload
    sensors.py            GET /sensors/schema, POST /sensors/predict, /predict/batch, /what-if
    acoustic.py           GET /acoustic/models, /samples, /samples/{name}, /enrolled; POST /acoustic/analyze, /diagnose, /enroll;
                          DELETE /acoustic/enrolled/{type}/{name}
    machines.py           GET /machines/overview, /machines, /machines/{id}; POST /{id}/maintenance, /{id}/summary
    alerts.py             GET /alerts; POST /alerts/{id}/acknowledge, /resolve
    assessments.py        POST /assessments (sensor-only, trusted server-side inference), GET /assessments,
                          GET /assessments/{id}, POST /assessments/acoustic (fan-only multipart audio upload,
                          hash-only recording retention), POST /assessments/bearing (bearing-only multipart audio
                          upload, .mat accepted, hash-only recording retention), POST /assessments/multimodal
                          (>=2 modalities in one strict multipart: machine_id + sensor_json + acoustic_file
                          + bearing_file, hash-only recording retention); owner-scoped via demo bearer auth
                          (TELLTALE_API_TOKENS, default-deny). New routes only - legacy routes stay outside this auth boundary
    ai.py                   POST /ai/explain (one immutable assessment snapshot -> grounded, validated template
                          explanation; owner-scoped; the generator is config-selectable - see provider selection
                          below), GET /ai/outputs/{ai_output_id}; append-only AI output store
                          (versioned migration v3, new table only). No list/search endpoints in this slice.
                          POST /ai/evidence-review (same snapshot -> structured evidence/uncertainty review,
                          kind="evidence_review", same store and GET route; no new migration).
                          POST /ai/analyze (same snapshot -> cross-modality analysis of >=2 available modalities,
                          single-modality refusal, fixed pairwise templates, agreement value; never a fused
                          score; kind="analyze", same store and GET route; no new migration)
                          POST /ai/report (owner-scoped maintenance report over one machine's immutable
                          assessment history: newest-first retrieval capped at 50 with truncation disclosure,
                          created_at-anchored lookback window, stored status/modality/flag summaries and
                          non-binding review suggestions; kind="report", same store and GET route; no new migration)
                          POST /ai/ask (Ask Telltale: one bounded question + exactly one scope - assessment_id
                          XOR machine_id, lookback_days only with machine_id; the question is untrusted inert
                          data matched against a fixed server-owned intent dictionary, never an instruction;
                          deterministic offline template answers built from owner-visible immutable snapshots,
                          or deterministic persisted refusals (unsupported_question / no_evidence /
                          evidence_insufficient / scope_mismatch); machine scope reuses the newest-first,
                          created_at-anchored, 50-capped window; stores only a bounded sanitized question_echo
                          plus question_sha256; kind="ask", same store and GET route; no new migration)
  schemas/                common.py, sensor.py, acoustic.py, fleet.py, assessment.py, ai.py (Pydantic v2)
  services/
    registry.py           ModelRegistry: loads models/sensor, models/acoustic/<type>, models/bearing (+ HINTS shown to users)
    bootstrap.py          first start: trains the sensor model in the background if missing (status in /health)
    fleet.py              FleetSimulator: per-machine state, scenarios, tick(), maintenance actions, alerting; tool_life()
    alerts.py             AlertStore (in-memory) + SqliteAlertStore (default; <data_dir>/state/telltale.db)
    playbook.py           Playbook: model output -> Recommendation[] (reads data/playbook.yaml)
    narrator.py           Narrator: template or OpenAI-compatible LLM briefing
    assessment_store.py   AssessmentStore: append-only SQLite store for immutable assessments (versioned
                          migrations scoped to new tables; same telltale.db file, alerts untouched;
                          v2 adds the immutable owner column - authenticated routes are owner-scoped)
    assess.py             pure assess/v1 quality/status rules + create_assessment (server-owned fields;
                          the sensor-only slice is served by api/routes/assessments.py)
    ai_template.py        Central AI agent slice A: facts-only snapshot projection + deterministic offline
                          template explanation + refusal/fallback builders (no network, no ML calls)
    ai_review_template.py Central AI agent slice B: deterministic offline evidence/uncertainty review
                          (fixed conservative status/flag meanings; no network, no ML calls)
    ai_validate.py        evidence-grounding validation for AI outputs (ids/statuses/numbers must exist
                          in the snapshot; invented facts never persist as validated)
    ai_review_validate.py evidence-grounding validation for AI evidence reviews (present/missing/modality
                          rows must equal the stored snapshot; required non-claim phrases enforced)
    ai_analyze_template.py Central AI agent slice C: deterministic offline cross-modality analysis
                          (stored indication tokens, fixed pairwise templates, agreement summary;
                          never a fused probability/health/risk score)
    ai_analyze_validate.py evidence-grounding validation for AI analyses (>=2 available modalities,
                          canonical pairwise relations, agreement vocabulary, required non-claims)
    ai_report_template.py Central AI agent slice D: deterministic offline maintenance report over one
                          machine's owner-visible assessment history (created_at-anchored lookback window,
                          status distribution, modality availability, recurring flags, non-binding review
                          suggestions; never a fused probability/health/risk score)
    ai_report_validate.py evidence-grounding validation for maintenance reports (owner-visible history and
                          window, recomputed counts/flags, grounded numbers/quotes, required non-claims,
                          forbidden claim wording rejected)
    ai_ask_template.py    Central AI agent slice E: deterministic offline Ask Telltale provider - fixed
                          server-owned intent dictionary + priority order, question sanitization/digest,
                          assessment- and machine-scoped facts-only answers, deterministic refusal builders
                          (the question is inert data; no network/LLM, no model calls, no fused score)
    ai_ask_validate.py    evidence-grounding validation for Ask answers (scope identity, exact requested
                          assessment / retrieved window, reused structural count grounding, required
                          non-claims, forbidden wording rejected; the echoed question is excluded from the
                          grounding scan)
    ai_output_store.py    AiOutputStore: append-only SQLite store for validated AI outputs (versioned
                          migration v3 = new ai_outputs table only; assessment rows never touched)
    ai_provider.py        Central AI agent provider boundary: AIProvider protocol, ProviderRequest/Result/Usage,
                          ProviderError + ProviderFailureClass, TemplateAIProvider (delegates byte-identically
                          to the template providers), resolve_provider_name + get_ai_provider factory. No transport
                          import: it routes to a provider by name only (provider selection is config-driven and
                          fails closed; an explicitly selected but unavailable provider is never downgraded).
    ai_amd_provider.py    AMD Token Factory inference provider (static Bearer key) for /ai/explain ONLY: minimized
                          facts-only projection sent to the documented {base}/chat/completions endpoint over httpx
                          (non-streaming, json_object, temperature 0), strict response parsing, bounded
                          timeout/retry (retry only 429/502/503, short Retry-After only), and an UNTRUSTED prose
                          candidate (explanation_text + optional uncertainty_notes). Never receives raw audio,
                          paths, fingerprints, ids or secrets; the route owns every structural field and provenance
                          and runs the unchanged grounding validator. No OAuth, no account/key mutation.
  data/playbook.yaml      inspection steps per failure mode / driver / machine type / bearing class
  data/fleet.yaml         demo machines (id, line, product_quality, acoustic_profile, scenario)
apps/api/tests/           pytest; conftest trains tiny models on synthetic data, so tests need no downloads

apps/web/src/
  main.tsx, App.tsx       React Query + Router setup; routes: / , /machines/:id , /sound , /alerts , /models
  api/types.ts            TS mirror of the API schemas
  api/client.ts           typed fetch wrapper (all calls use relative /api/v1)
  api/hooks.ts            React Query hooks (poll every 3 s for live data)
  components/             Layout, ModelBanner, Badges, StatTile, Sparkline, ContributionChart, FailureModes,
                          Recommendations, WhatIfPanel, HistoryChart, Spectrogram (canvas), Waveform (svg),
                          Dropzone, SoundAnalyzer, BearingDiagnoser, Cmd (OS-aware helper command + copy button),
                          RecordButton (mic -> WAV in the browser), EnrollPanel (teach a machine's normal sound)
  lib/wav.ts              encodeWav()/toMono() for browser recordings
  pages/                  FleetPage, MachinePage, SoundLabPage, AlertsPage, ModelsPage
  lib/format.ts           formatting helpers + sensor labels/units
  styles.css              all styling; design tokens are CSS variables in :root

telltale.cmd              Windows entry point (CRLF!): no args = menu (double-click), else forwards to scripts/telltale.ps1
scripts/telltale.ps1      Windows helper logic (PowerShell 5.1, ASCII only)
telltale.sh               Mac/Linux helper (bash 3.2 compatible) - same commands as the .ps1
START_HERE.md             beginner guide (install Docker, open a terminal, cheat sheet, glossary, troubleshooting)
docs/demo-script.md       3-minute pitch path through the app (keep it in sync with UI changes)
docs/model-improvements.md research notes + every experiment and result (add new ones here)
apps/*/README.md, ml/README.md, data/README.md, models/README.md   short per-folder guides for beginners
.vscode/tasks.json        Terminal -> Run Task buttons for the helper commands

ml/
  configs/                sensor.yaml, acoustic.yaml, bearing.yaml (hyper-parameters, paths, thresholds)
  src/telltale_ml/
    paths.py              data/models dirs (env TELLTALE_DATA_DIR / TELLTALE_MODELS_DIR)
    utils.py              config loading, seeding, JSON writing (numpy-safe)
    metrics.py            best_f1_threshold, binary_report, anomaly_report
    sensor/schema.py      SENSOR_FIELDS, FEATURES (order matters), FAILURE_MODES, AI4I column map
    sensor/features.py    engineer(): raw readings -> feature matrix
    sensor/data.py        load_ai4i()
    sensor/synthetic.py   AI4I-rule data generator (tests, offline runs)
    sensor/train.py       train_sensor_models() + CLI telltale-train-sensor
    sensor/predictor.py   SensorPredictor.assess() / what_if() / default_sweep()
    acoustic/features.py  LogMelConfig, load_audio, log_mel, stack_frames, pool
    acoustic/data.py      Clip, discover_mimii_due(), discover_mimii()
    acoustic/model.py     DenseAutoencoder, ScoringModel (legacy autoencoder), pick_device  [torch]
    acoustic/embedding.py SpecEmbedder CNN + ArcFace loss, train_embedder(), export_embedder() -> ONNX  [torch]
    acoustic/train_ensemble.py  v2 trainer: spectrograms, calibration hold-out, embedder, banks, section-unknown eval  [torch]
    acoustic/knn_scorer.py      EnsembleScorer (domain-wise kNN, LOO z-norm, min over sections), conformal p-values
    acoustic/train_anomaly.py  train_anomaly_model() + ONNX export/verify + CLI   [torch]
    acoustic/detector.py  AcousticDetector.analyze() (ensemble or legacy), enroll()/clear_enrollment(): add a machine's
                          normal clips as an extra bank + Mondrian conformal calibration (>= 19 clips)
    acoustic/fault.py     bearing features, label_from_path, FaultClassifier.diagnose()
    acoustic/train_fault.py    train_fault_model(): leave-one-condition-out CV, demo hold-out clips + CLI telltale-train-bearing
    datasets/download.py  CLI telltale-download {ai4i|mimii-due|mimii|bearing|samples|list}
  tests/                  pytest for features, training, ONNX round-trip (skips torch parts if missing)
```

Artifacts produced by training:

```
models/sensor/{failure.txt, mode_TWF.txt, mode_HDF.txt, mode_PWF.txt, mode_OSF.txt, metadata.json}
models/acoustic/<machine_type>/{embedder.onnx, bank.npz, metadata.json}   (legacy autoencoder: model.onnx)
models/bearing/{model.txt, metadata.json}
data/processed/sensor_test_split.csv
```

## 4. Setup and commands

Prerequisites: **beginners need only Docker Desktop.** Developers who want to run things outside Docker also need Python 3.10+ (Docker uses 3.11), [uv](https://docs.astral.sh/uv/) and Node 20+.

**Beginner / default workflow (only Docker Desktop needed):**

| Task | Windows | Mac / Linux |
|---|---|---|
| Menu | `.\telltale` (or double-click `telltale.cmd`) | `bash telltale.sh` |
| Start app (+ auto-train sensor model on first start) | `.\telltale start` | `bash telltale.sh start` |
| Edit mode: live reload for web + API, in Docker | `.\telltale dev` | `bash telltale.sh dev` |
| Stop / logs / diagnose | `.\telltale stop` / `logs` / `doctor` | `bash telltale.sh stop` / `logs` / `doctor` |
| Train sound model | `.\telltale train-sound fan [--quick]` | `bash telltale.sh train-sound fan [--quick]` |
| Train bearing model / retrain sensor | `.\telltale train-bearing` / `train-sensor` | `bash telltale.sh train-bearing` / `train-sensor` |

Set `TELLTALE_NO_BROWSER=1` to stop the helper opening a browser (CI, testing).

**Developer workflow (Python + Node installed locally):**

| Task | Command (from the repo root) |
|---|---|
| Install everything | `uv sync --extra torch` and `cd apps/web && npm install` (or `make setup`) |
| Install with plain pip | `pip install -e "./ml[train,torch,dev]" -e "./apps/api[dev]"` |
| Get sensor data + train | `uv run telltale-download ai4i && uv run telltale-train-sensor` (`make demo`) |
| Train without internet | `uv run telltale-train-sensor --synthetic` |
| Sound data + train | `uv run telltale-download mimii-due --machine fan && uv run telltale-train-acoustic --machine-type fan` |
| Quick sound model | `uv run telltale-train-acoustic --machine-type fan --epochs 5 --max-train-clips 300` |
| Run API | `uv run uvicorn telltale_api.main:app --reload --port 8000` → http://localhost:8000/docs |
| Run web | `cd apps/web && npm run dev` → http://localhost:5173 (proxies /api to :8000) |
| Python tests | `uv run pytest ml/tests -q` and `uv run pytest apps/api/tests -q` (`make test`) |
| Web typecheck + build | `cd apps/web && npm run build` (`make test-web`) |
| Lint / format | `uv run ruff check ml apps/api` / `make fmt` |
| Docker stack | `docker compose up --build` → web :8080, API :8000 |
| Docker hot reload | `docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build` |
| AMD GPU shell | `make rocm-shell` (Linux + ROCm only) |

Run each test suite separately. They are two separate pytest roots.

## 5. Definition of done (for any change)

- [ ] `uv run pytest ml/tests -q` and `uv run pytest apps/api/tests -q` pass.
- [ ] `cd apps/web && npm run build` passes (strict TypeScript, no unused variables).
- [ ] `uv run ruff check ml apps/api` is clean.
- [ ] New behaviour has a test. API tests use the `client` fixture; ML tests use synthetic data. **Tests must never need downloaded datasets or a GPU.**
- [ ] Contracts are in sync (rule 6), and this file is updated if structure or commands changed.
- [ ] If you changed anything user-facing (commands, ports, setup), `START_HERE.md` and the per-folder READMEs still tell the truth.

## 6. Recipes

### 6.1 Add a sensor input (e.g. `vibration_mm_s`)
1. `ml/.../sensor/schema.py`: add it to `SENSOR_FIELDS`, `FEATURE_LABELS`, and the source column to `AI4I_COLUMN_MAP` (or a new loader).
2. `sensor/features.py`: nothing to do unless it needs derived features. `FEATURES` order defines the model input.
3. `sensor/synthetic.py`: generate it so tests and the simulator still work.
4. `apps/api/.../schemas/sensor.py`: add it to `SensorReading` (with bounds) and to the `SensorParameter` literal if it can be swept.
5. `services/fleet.py`: produce it in `_next_reading`.
6. `apps/web/src/api/types.ts` (`SENSOR_FIELDS`, `SensorReading`) and `lib/format.ts` (`SENSOR_LABELS`).
7. Optionally add a driver hint in `data/playbook.yaml` → `drivers`.
8. Retrain (`telltale-train-sensor`). Old artifacts will fail on a feature mismatch.

### 6.2 Add a sound machine type
Nothing in the code needs to change. Run `telltale-download mimii-due --machine gearbox`, then `telltale-train-acoustic --machine-type gearbox`, then reload models. Add a `gearbox` entry under `acoustic:` in `playbook.yaml` if it has none. To make a fleet machine use it, set `acoustic_profile: gearbox` in `data/fleet.yaml`.

### 6.3 Support a new audio dataset layout
Add `discover_<name>(root, machine_type) -> list[Clip]` in `acoustic/data.py`, register it in `discover()`, add the name to the `--dataset` choices in `train_anomaly.py`, and add a test in `ml/tests/test_acoustic.py` that builds a fake directory tree (see `test_discover_layouts`).

### 6.4 Add an API endpoint
1. Pydantic models go in `schemas/<area>.py`.
2. Add the route in `api/routes/<area>.py`, get state with `state: AppState = Depends(get_state)`, and guard model access with `state.registry.require_*()`.
3. CPU-heavy work (audio decoding, big batches) goes in an `async def` route that calls `await run_in_threadpool(fn, ...)`. Mutate fleet or alert state only on the event loop.
4. For a new router, register it in `api/router.py`.
5. Add the client method to `web/src/api/client.ts`, its types to `types.ts`, and optionally a hook to `hooks.ts`.
6. Add a test in `apps/api/tests/`.

### 6.5 Add a dashboard page
Create `pages/XPage.tsx`, add a `<Route>` in `App.tsx` and a `NAV` entry in `components/Layout.tsx`. Use the existing classes (`card`, `grid grid-2`, `tile`, `badge`, `btn`, `tabs`). Do not add a CSS framework. Charts use **recharts**; very dense visuals (spectrograms) use canvas.

### 6.6 Replace the simulator with real telemetry
`FleetSimulator._next_reading(state)` is the only place readings are created. Swap it for a source (MQTT subscriber, OPC UA poller, CSV replay) that returns the same dict shape. `tick()`, alerting, history and the UI stay unchanged. Set `TELLTALE_SIMULATOR_ENABLED=false` to stop the random generator and keep `tick()` driven by your source.

### 6.7 Persist alerts or history
Alerts already persist: `SqliteAlertStore` (same methods as `AlertStore`) writes `<data_dir>/state/telltale.db`, and compose mounts `./data/state`. Turn it off with `TELLTALE_PERSIST_ALERTS=false`. To persist more (telemetry history, acoustic snapshots), follow the same pattern: in-memory class plus a subclass that also writes, constructed in `main.build_state()`.

### 6.8 Change thresholds or risk bands
- Sensor: `ml/configs/sensor.yaml` → `risk.medium_fraction`. The high band is the validation max-F1 threshold. Retrain to apply.
- Failure-mode flags: `min_mode_val_f1` (default 0.5). A mode model whose **validation** F1 is below it is marked `reliable: false`. It is still shown, but it never raises a flag and is never picked as the "most likely mechanism". On real AI4I data this applies to TWF (tool-wear failures happen at a random point between 200 and 240 min).
- First-start auto-training: `TELLTALE_SENSOR_AUTOTRAIN` (default true) and `TELLTALE_SENSOR_AUTOTRAIN_OFFLINE` (skip the download, use synthetic data). If `models/` is not writable the model goes to `TELLTALE_FALLBACK_MODELS_DIR` (a temp dir).
- Sound: `ml/configs/acoustic.yaml` → `threshold_percentile`, `watch_fraction`. Retrain, or edit `metadata.json` for a quick demo tweak and call `POST /api/v1/models/reload`.

### 6.9 Edit inspection advice
Only `apps/api/telltale_api/data/playbook.yaml` needs to change. Keys must match failure-mode codes, feature names, machine types or bearing classes.

### 6.10 Enable LLM briefings
Run any OpenAI-compatible server. On AMD use `make llm-up` (ROCm vLLM, port 8001). Then set `TELLTALE_LLM_BASE_URL=http://llm:8001/v1` (Docker) or `http://localhost:8001/v1` (local) and optionally `TELLTALE_LLM_MODEL`. The prompt is in `services/narrator.py`.

### 6.11 Train on an AMD GPU
`make rocm-shell`, then any `telltale-train-*` command. `pick_device("auto")` picks the GPU because ROCm PyTorch reports AMD GPUs through `torch.cuda`. **Never `pip install torch` inside the ROCm image.** It would replace the ROCm build. Change the base image with `ROCM_PYTORCH_IMAGE`.

### 6.12 Change the helper scripts (`telltale.cmd`, `scripts/telltale.ps1`, `telltale.sh`)
- **Keep the two implementations in sync.** Every command must exist in both `scripts/telltale.ps1` and `telltale.sh` with the same name and behaviour, and appear in both `help` texts, both menus, the START_HERE cheat sheet and `.vscode/tasks.json` (if it is a common action).
- `scripts/telltale.ps1` must run on **Windows PowerShell 5.1**: no `&&`/`||`, no ternary or `??`, and **ASCII only** (5.1 reads BOM-less files as ANSI). Write `.env` without a BOM (`[System.IO.File]::WriteAllLines`).
- `telltale.cmd` must keep **CRLF** line endings (`.gitattributes` enforces it). Keep it a thin shim.
- `telltale.sh` must run on **bash 3.2** (macOS): no associative arrays, no `${var,,}`, no `mapfile`. Inside `$(...)`, `fail` only exits the subshell, so add `|| exit 1`.
- Error messages: say what went wrong and the exact next step (`fail "..." "tip 1" "tip 2"`).
- Commands shown in the web UI go through `components/Cmd.tsx` / `lib/format.ts: cli()`, which produce the OS-correct form. Server-side hints are in `services/registry.py: HINTS`.
- Test: `.\telltale doctor`, `start`, `start` again ("already running"), `stop`, `dev`, and the same with `bash telltale.sh` (Git Bash works on Windows).

### 6.13 Select the AI provider for the Central AI Agent (`template` vs `amd`)
Env-only, no code change. `TELLTALE_AI_PROVIDER` defaults to `template` and accepts `template` or `amd`; anything else fails closed at startup (no silent fallback). **`amd` changes only `POST /ai/explain` and `POST /ai/analyze`** — evidence-review, report and ask always keep the template provider and report `provider=template`, so a template result is never mislabelled AMD-generated. To use AMD, set `TELLTALE_AMD_API_KEY` (static Bearer key, `SecretStr` — never logged/echoed/serialized) and `TELLTALE_AMD_MODEL` (no default; there is no hard-coded model). The base URL is pinned to `https://developer.amd.com.cn/radeon/api/v1`. Transport bounds: `TELLTALE_AI_TIMEOUT_S` (<=120), `TELLTALE_AI_MAX_RETRIES` (<=3) and `TELLTALE_AI_RETRY_MAX_S`. `TELLTALE_AI_MAX_OUTPUT_TOKENS` is optional (1..4000): unset/empty/`auto`/`omit`/`none` OMITS `max_tokens` from the request body (some AMD models reject an explicit value with HTTP 400 — DeepSeek-V4-Flash reports its allowed maximum as `null`); a positive integer sends it. Only `explain` and `analyze` are wired; a request that fails (transport, timeout, 429/502/503 after bounded retries, malformed/empty JSON, or a grounding failure) persists the deterministic template fallback with `provider=template`, `model=null` — a failed AMD candidate is never stored as `validated`. The service is trial/development only (no SLA) and retains prompts/outputs ~2 days. See `docs/model-improvements.md` for the design notes.

## 7. Conventions

- **Python:** 3.10-compatible syntax (the ROCm image ships 3.10). Type hints everywhere, `X | None` style. Ruff with line length 110. Use the module-level `log = logging.getLogger("telltale.<area>")` for logging, and use `print` only for CLI summaries. Keep console output ASCII, because Windows consoles choke on emoji.
- **Naming:** sensor fields are snake_case with units (`torque_nm`, `tool_wear_min`). Failure modes use the AI4I codes (`TWF`…). Machine types are lowercase (`fan`).
- **Config over constants:** hyper-parameters and thresholds go in `ml/configs/*.yaml`. Runtime knobs go in `core/config.py` (env `TELLTALE_*`) and are documented in `.env.example`.
- **TypeScript:** strict mode. Server state goes through React Query hooks, not `useEffect` fetches. Components are small and presentational, and pages compose them. Colors come only from the CSS variables in `styles.css`.
- **Commits:** imperative mood ("Add vibration input to sensor model"), one logical change per commit.

## 8. Gotchas

- **Windows + Docker line endings:** `.gitattributes` forces LF (CRLF for `*.cmd`). If a container script fails with `\r` errors, re-checkout.
- **Live reload inside Docker** on Windows/macOS needs polling: `WATCHFILES_FORCE_POLLING` (API) and `VITE_USE_POLLING` (web) are set in `docker-compose.dev.yml`. Don't remove them.
- **Microphone recording** (`RecordButton`) only works on `http://localhost` or https. Browsers block the mic on plain-http LAN addresses. It disables echo cancellation, noise suppression and auto-gain on purpose, because those would remove the machine noise.
- **API image is a two-stage build:** packages are installed and stripped (test folders, debug symbols) in a builder stage, saving about 330 MB. If a new dependency breaks at import time inside Docker but works locally, suspect the stripping step in `apps/api/Dockerfile`.
- **Ports:** the helper writes the free ports it picked into `.env` (`API_PORT`, `WEB_PORT`, `DEV_WEB_PORT`). Don't hard-code 8000/8080/5173 in docs or code meant for users. Say "the address the helper prints".
- **The API container writes to `models/`** (auto-training and sound enrollment in `models/acoustic/<type>/enrolled/`), so `./models` is mounted read-write in compose. Retraining a sound model invalidates enrolled clips (different embeddings), so delete `enrolled/` after retraining.
- **Downloads are resumable:** `datasets/download._download` checks Content-Length, resumes with HTTP Range, and only gives up after 10 attempts in a row without progress. Partial files are `<name>.part`; re-running resumes. `_extract` refuses damaged zips (deletes them) and writes an `.extracted-<zip>` marker so it never re-extracts. Never replace this with a plain `urlopen(...).read()` loop: on flaky Wi-Fi the server drops the connection without an error and you get a silently truncated file.
- **Bearing order features need the shaft speed:** `FaultClassifier.diagnose(source, rpm)`; the API takes `rpm` or a `machine_id` (uses its live telemetry rpm) and otherwise assumes `default_rpm` and reports `rpm_assumed: true`.
- **Conformal p-values are only valid when new clips resemble the calibration clips.** Under a new operating condition, false alarms were ~57% until that condition was enrolled. Don't promise "5% false alarms" for unenrolled new conditions.
- **MIMII clips are 8-channel;** we take `channel: 0`. MIMII DUE is mono.
- **BatchNorm needs batches larger than 1** in training. The DataLoader uses `drop_last=True` and `batch_size = min(batch_size, n)`.
- **ONNX export:** `train_anomaly.export_onnx` forces the TorchScript exporter (`dynamo=False`, with a fallback for old torch) and checks the ONNX output against torch. Keep that check.
- **LightGBM needs `libgomp1`** and soundfile needs `libsndfile1` in slim Linux images (already in the Dockerfiles).
- **Models are loaded once** at startup or on `POST /models/reload`. After retraining, reload; don't restart everything.
- **`HIP_VISIBLE_DEVICES=""` hides all GPUs.** Leave it unset unless you are pinning.
- **MIMII DUE is CC BY-NC-SA** (non-commercial). Fine for the hackathon; flag it for anything else.
- **AI4I is synthetic.** Near-perfect scores are expected, so don't over-claim them in the UI or the pitch.
- **Sound CNN variance:** one ArcFace CNN's quality varies a lot between seeds (DCASE score 0.59–0.65 on fan). Compare methods over several seeds, never one run. `ensemble.n_models` averages several CNNs (steadier, slower). The experiment log is in `docs/model-improvements.md`. Add new results there.
- **Sound metrics are section-unknown** (min over all section banks), because the API scores uploads without metadata. Section-known numbers are about 0.03 higher and must not be presented as app performance.
- **Bearing dataset:** MATLAB `.mat` files (variable `x`, 44.1 kHz, no sample rate stored). `load_audio` reads `.mat` by extension or `MATLAB` header bytes. Labels come from folder names via keywords in `ml/configs/bearing.yaml`; "Case 6 - Combined faults (Inner, outer, ball)" matches three keywords, so the `combined` class has `match_first: true`. Bearing features must stay **loudness-independent**: the two recording setups differ ~5x in level, and absolute-level features made the model learn the setup instead of the fault (33% vs 81% accuracy).

## 9. Don'ts

- Don't add torch, tensorflow or other heavy ML dependencies to `telltale-ml`'s base dependencies or to the API image.
- Don't put inference logic in route handlers. Routes orchestrate; `telltale_ml` and `services/` do the work.
- Don't hard-code paths; use `telltale_ml.paths` or `Settings`.
- Don't commit datasets, model artifacts, `.env` or `node_modules`.
- Don't swallow exceptions silently. Log them (`log.exception`) and return a clear HTTP error.
- Don't let the frontend call any URL other than relative `/api/v1/...`.

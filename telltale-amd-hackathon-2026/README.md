<div align="center">

# Telltale

**Catch the telltale signs before the line stops.**

Predictive maintenance that watches machine telemetry *and* listens to machine sound, then tells the maintenance team what to inspect first.

`FastAPI` · `React + TypeScript` · `LightGBM` · `PyTorch → ONNX` · `AMD ROCm`

</div>

> **New here? Never used Docker or a terminal?** Read **[START_HERE.md](START_HERE.md)** first. It takes about 20 minutes and needs no experience.

> **Deploy the complete frontend and API on Vercel?** Follow **[VERCEL_DEPLOY.md](VERCEL_DEPLOY.md)**.

---

## The problem

Industrial machines usually warn you before they fail. Tool wear builds up, torque creeps up, cooling headroom shrinks, or a fan starts to sound slightly off. These signs are spread across different signals and across dozens of machines, so they get missed. The result is unplanned downtime, scrapped parts and emergency repairs.

## What Telltale does

Telltale combines two complementary signals in one dashboard:

| | **Telemetry: failure risk** | **Sound: anomaly and fault detection** |
|---|---|---|
| Input | Air/process temperature, speed, torque, tool wear | WAV/FLAC/OGG recordings of the machine |
| Model | LightGBM: P(failure) plus one model per failure mode | ArcFace CNN embeddings + nearest-neighbour scoring on *normal* sound (anomaly), and a LightGBM bearing-fault classifier with conformal prediction sets |
| Output | Risk band, likely failure mode (tool wear, heat dissipation, power, overstrain), **per-input explanation (SHAP)** | Anomaly score vs threshold, **when** it sounded odd, **which frequencies** differ from normal, bearing fault class |
| Interaction | **What-if analysis**: sweep any input and see how the risk curve moves | Upload a clip or pick a sample and compare it with the machine's normal reference |

For a machine the model hasn't heard before, upload about 20 healthy recordings (**"teach this machine's normal sound"**). They become its normal reference with no retraining, and its alarm calibration is personalised.

Both signals feed the same pipeline. Each machine gets a combined **health** status. Crossing a threshold raises an **alert**. A YAML **inspection playbook** turns model outputs into concrete steps, and operators can **acknowledge, resolve and log maintenance**. An optional **LLM briefing** runs on an AMD GPU via vLLM and turns the facts into a short summary for the technician. It only rephrases what the models computed.

### Screens

- **Fleet:** every machine ranked by health, with live risk sparklines, the latest sound check and a tool-change countdown
- **Machine:** risk, failure-mode bars, SHAP "why" chart, inspection steps, trend, what-if, sound check, maintenance actions, briefing
- **Sound lab:** anomaly check (waveform, spectrogram with an error strip, per-band deviation) and bearing diagnosis. Upload a file or **record live from the microphone**
- **Alerts:** triage queue (kept across restarts)
- **Models:** what each model was trained on, its held-out metrics and its caveats

## Built on AMD

- **Training on AMD Instinct / Radeon GPUs** with ROCm PyTorch (`ml/Dockerfile.rocm`, `docker-compose.rocm.yml`). ROCm exposes AMD GPUs through the `torch.cuda` API, so the same training code runs unchanged on an MI300X, a Radeon card or a CPU.
- **Lean serving:** the sound models are exported to **ONNX** and served with ONNX Runtime. ROCm/MIGraphX execution providers can be enabled with `TELLTALE_ORT_PROVIDERS`.
- **On-prem LLM briefings:** an optional ROCm **vLLM** service (`make llm-up`) serves an open model such as Qwen2.5-7B-Instruct on the same GPU. No factory data leaves the site.

---

## Quick start

The only thing you need is **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**, open and running. Then, in a terminal in this folder:

| | Windows | Mac / Linux |
|---|---|---|
| Start (first time: 5–15 min) | `.\telltale start` | `bash telltale.sh start` |
| Edit mode (live reload) | `.\telltale dev` | `bash telltale.sh dev` |
| Stop | `.\telltale stop` | `bash telltale.sh stop` |
| Something's wrong | `.\telltale doctor` | `bash telltale.sh doctor` |
| Everything else (menu) | `.\telltale` or double-click `telltale.cmd` | `bash telltale.sh` |

The helper checks Docker (and starts it if needed), picks free ports, builds and starts everything, waits until the app is healthy, then opens your browser. On the **first start** the API trains the sensor model by itself (about a minute). If there's no internet, it uses simulated data and the Models page says so.

The fleet starts moving immediately. A built-in simulator streams AI4I-style telemetry with injected faults (tool wear, cooling fault, overload, power anomaly), so machines drift into *watch* and *critical* and raise alerts. Use **Replace tool** or **Service cooling** on a machine page and watch the risk fall.

### Add the sound models

```bash
.\telltale train-sound fan          # Mac/Linux: bash telltale.sh train-sound fan
                                    # downloads MIMII DUE (~1.1 GB), trains, exports ONNX, copies demo clips
                                    # also: pump | gearbox | slider | valve   (add --quick for a fast rough model)
.\telltale train-bearing            # downloads the Mendeley bearing sounds (75 MB), trains, adds demo clips
```

The running app reloads the new models automatically.

### Advanced: without Docker

You need Python 3.10+, [uv](https://docs.astral.sh/uv/), Node 20+ and `make` (on Windows, copy the commands from the `Makefile`).

```bash
make setup            # uv sync --extra torch  +  npm install
make demo             # download AI4I + train sensor model
make api              # terminal 1 → http://localhost:8000/docs
make web              # terminal 2 → http://localhost:5173
make test             # Python tests;  make test-web for the web build
```

### Train on an AMD GPU

On a Linux host with ROCm drivers, for example an AMD Developer Cloud MI300X instance:

```bash
make rocm-shell                          # ROCm PyTorch container with the repo mounted
python -c "import torch; print(torch.cuda.get_device_name(0))"
telltale-download mimii-due --machine fan --machine pump --machine gearbox
telltale-train-acoustic --machine-type fan
```

Pick the `rocm/pytorch` tag that matches your driver and set `ROCM_PYTORCH_IMAGE` in `.env`.

---

## Repository layout

```
.
├── apps/
│   ├── api/                    FastAPI service (telltale_api)
│   │   ├── telltale_api/
│   │   │   ├── api/routes/     health · models · sensors · acoustic · machines · alerts
│   │   │   ├── schemas/        Pydantic request/response models
│   │   │   ├── services/       registry · fleet simulator · alerts · playbook · narrator (LLM)
│   │   │   └── data/           playbook.yaml (inspection steps) · fleet.yaml (demo machines)
│   │   └── tests/
│   └── web/                    React + TypeScript + Vite dashboard
│       └── src/  api/ · components/ · pages/ · lib/
├── ml/                         telltale_ml: shared training + inference package
│   ├── configs/                sensor.yaml · acoustic.yaml · bearing.yaml
│   ├── src/telltale_ml/
│   │   ├── sensor/             schema · features · train · predictor · synthetic
│   │   ├── acoustic/           features · data · model (torch) · train_anomaly · detector (ONNX) · fault · train_fault
│   │   └── datasets/           download CLI
│   ├── Dockerfile              CPU trainer
│   └── Dockerfile.rocm         AMD GPU trainer
├── data/                       raw/ · processed/ · samples/   (git-ignored)
├── models/                     trained artifacts               (git-ignored)
├── docs/                       architecture.md · datasets.md · model-improvements.md · demo-script.md
├── docker-compose.yml          api + web (+ trainer profile)
├── docker-compose.dev.yml      hot-reload overlay
├── docker-compose.rocm.yml     AMD GPU trainer + optional vLLM
├── telltale.cmd / telltale.sh one-command helper (start, dev, stop, doctor, train-*)
├── scripts/telltale.ps1       Windows logic behind telltale.cmd
├── Makefile                   advanced, non-Docker workflow
├── START_HERE.md              beginner guide
└── AGENTS.md                  guide for AI coding agents (and humans)
```

## API at a glance

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness and which models are loaded |
| GET | `/api/v1/models` · POST `/models/reload` | Model cards (metrics, caveats); hot reload |
| POST | `/api/v1/sensors/predict` | Risk, failure modes, SHAP contributions, recommendations |
| POST | `/api/v1/sensors/what-if` | Risk curve while sweeping one input |
| POST | `/api/v1/acoustic/analyze` | Sound anomaly analysis (multipart upload) |
| POST | `/api/v1/acoustic/diagnose` | Bearing fault classification |
| GET | `/api/v1/machines` · `/machines/{id}` | Fleet and machine detail with history |
| POST | `/api/v1/machines/{id}/maintenance` · `/summary` | Log an action · plain-language briefing |
| GET/POST | `/api/v1/alerts` · `/{id}/acknowledge` · `/{id}/resolve` | Alert triage |

The full interactive docs are at `/docs`.

## Evaluation and honesty

We report what the models can and cannot do:

- **Sensor:** PR-AUC, ROC-AUC, precision, recall and F1 on a stratified held-out test split, with prevalence shown next to them (failures are about 3.4% of rows). Thresholds are tuned on a separate validation split. Failure-mode models that are weak on validation data (on AI4I that is TWF, because tool-wear failures happen at a random point) are shown as *low-confidence* and never raise alerts. AI4I 2020 is **synthetic** and its labels follow rules, so the scores overstate real-world performance. The Models page says this.
- **Sound:** AUC and pAUC (FPR ≤ 0.1) on the official MIMII DUE test clips, reported **per section and per domain**, and measured **without telling the model which section a clip came from** (like real uploads). Scores are turned into conformal p-values using held-out *normal* clips, so "p = 0.05" really means 5% of normal recordings look that unusual. The research and every experiment are in [`docs/model-improvements.md`](docs/model-improvements.md).
- **Bearings:** leave-one-condition-out cross-validation. Every speed + recording-setup combination is held out in turn, so a recording is never split between training and test. The much harder "different microphone setup" score is reported next to it (about 45%, up from 32%, thanks to microphone augmentation and envelope features). When the shaft speed is known, order analysis lifts accuracy on unseen speeds to about 82% of recordings. Conformal prediction sets make the model say "inner race or ball" when it can't tell. The demo clips come from a condition the final model never saw.
- Every prediction includes **out-of-range warnings** when inputs leave the training distribution.

## Datasets

| Dataset | Licence | Use |
|---|---|---|
| [AI4I 2020 Predictive Maintenance (UCI #601)](https://archive.ics.uci.edu/dataset/601) | CC BY 4.0 | Sensor failure risk |
| [MIMII DUE (Zenodo 4740355)](https://zenodo.org/records/4740355) | CC BY-NC-SA 4.0 | Sound anomaly (default) |
| [MIMII (Zenodo 3384388)](https://zenodo.org/records/3384388) | CC BY-SA 4.0 | Sound anomaly (more data) |
| [Rolling-element bearing sounds (Mendeley n9y9c7xrz3)](https://data.mendeley.com/datasets/n9y9c7xrz3/1) | CC BY 4.0 | Bearing fault classes |

Details, folder layouts and caveats are in [`docs/datasets.md`](docs/datasets.md).

<details>
<summary>Citations</summary>

- S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," AI4I 2020.
- H. Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," DCASE 2019 Workshop, arXiv:1909.09347.
- R. Tanabe et al., "MIMII DUE: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection with Domain Shifts due to Changes in Operational and Environmental Conditions," arXiv:2105.02702.
- A. Ait Ben Ahmed, "Sound Datasets of a Rolling Element Bearing under Various Operating Conditions," Mendeley Data V1, doi:10.17632/n9y9c7xrz3.1.

</details>

## Roadmap ideas

- Real data ingestion (MQTT / OPC UA) behind the `FleetSimulator` interface
- Persist alerts and history (SQLite or Postgres)
- Remaining-useful-life estimate for tool wear
- Fuse sound and telemetry into one model once paired data exists
- Domain-adaptive sound models (MIMII DUE target domains)
- Mobile "record and check" page for technicians on the floor

## Contributing

Read **[AGENTS.md](AGENTS.md)**. It is written for AI coding agents, but it is also the fastest way for a human to understand the codebase.

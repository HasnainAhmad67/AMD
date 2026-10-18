# Telltale task runner. `make help` lists everything.
# Local targets use uv (https://docs.astral.sh/uv/). No make on Windows? Copy the command after the target.

MACHINE ?= fan
PY      ?= uv run
COMPOSE ?= docker compose
ROCM    := $(COMPOSE) -f docker-compose.yml -f docker-compose.rocm.yml
DEV     := $(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml

.DEFAULT_GOAL := help
.PHONY: help setup data-sensor data-sound data-bearing samples train-sensor train-sensor-synthetic \
        train-sound train-sound-quick train-bearing api web test test-ml test-api test-web lint fmt \
        up dev down logs docker-train-sensor docker-train-sound rocm-shell rocm-train-sound llm-up demo

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[33m%-24s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
setup: ## Install Python (uv, incl. torch) and web dependencies
	uv sync --extra torch
	cd apps/web && npm install

# ---------------------------------------------------------------- data
data-sensor: ## Download AI4I 2020 (0.5 MB)
	$(PY) telltale-download ai4i

data-sound: ## Download MIMII DUE for MACHINE (~1.1 GB), e.g. make data-sound MACHINE=pump
	$(PY) telltale-download mimii-due --machine $(MACHINE)

data-bearing: ## Print download steps for the Mendeley bearing-sound dataset
	$(PY) telltale-download bearing

samples: ## Copy a few test clips for MACHINE into data/samples (for the UI's sample picker)
	$(PY) telltale-download samples --machine $(MACHINE)

# ---------------------------------------------------------------- train
train-sensor: ## Train sensor failure-risk models (needs data-sensor)
	$(PY) telltale-train-sensor

train-sensor-synthetic: ## Train sensor models on AI4I-style synthetic data (offline smoke run)
	$(PY) telltale-train-sensor --synthetic

train-sound: ## Train the sound anomaly detector for MACHINE (GPU if available)
	$(PY) telltale-train-acoustic --machine-type $(MACHINE)

train-sound-quick: ## Fast low-quality sound model for MACHINE (1 CNN, 5 epochs, 300 clips)
	$(PY) telltale-train-acoustic --machine-type $(MACHINE) --epochs 5 --max-train-clips 300 --n-models 1

train-bearing: ## Train the bearing fault classifier (needs the Mendeley data in data/raw/bearing_sound)
	$(PY) telltale-train-bearing

# ---------------------------------------------------------------- run locally
api: ## Run the API with auto-reload on :8000
	$(PY) uvicorn telltale_api.main:app --reload --port 8000 --reload-dir apps/api/telltale_api --reload-dir ml/src

web: ## Run the web app (Vite) on :5173
	cd apps/web && npm run dev

demo: data-sensor train-sensor ## Minimum to see the dashboard: data + sensor model (then `make api` + `make web`)

# ---------------------------------------------------------------- quality
test: test-ml test-api ## Run all Python tests

test-ml:
	$(PY) pytest ml/tests -q

test-api:
	$(PY) pytest apps/api/tests -q

test-web: ## Typecheck + production build of the web app
	cd apps/web && npm run build

lint: ## Ruff lint
	$(PY) ruff check ml apps/api

fmt: ## Ruff format + autofix
	$(PY) ruff format ml apps/api
	$(PY) ruff check --fix ml apps/api

# ---------------------------------------------------------------- docker
up: ## Production-like stack: API :8000, web :8080
	$(COMPOSE) up --build -d

dev: ## Hot-reload stack: API :8000, web :5173
	$(DEV) up --build

down: ## Stop all containers
	$(COMPOSE) down

logs: ## Tail container logs
	$(COMPOSE) logs -f --tail=100

docker-train-sensor: ## Train the sensor model inside the CPU trainer container
	$(COMPOSE) --profile train run --rm trainer sh -c "telltale-download ai4i && telltale-train-sensor"

docker-train-sound: ## Train the sound model for MACHINE inside the CPU trainer container
	$(COMPOSE) --profile train run --rm trainer telltale-train-acoustic --machine-type $(MACHINE)

# ---------------------------------------------------------------- AMD GPU (Linux + ROCm)
rocm-shell: ## Shell in the ROCm training container (AMD GPU)
	$(ROCM) --profile train run --rm trainer bash

rocm-train-sound: ## Train the sound model for MACHINE on the AMD GPU
	$(ROCM) --profile train run --rm trainer telltale-train-acoustic --machine-type $(MACHINE) --device cuda

llm-up: ## Start the optional vLLM server on the AMD GPU (port 8001)
	$(ROCM) --profile llm up -d llm

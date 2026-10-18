#!/usr/bin/env bash
# Telltale helper for Mac and Linux (also works in Git Bash on Windows).
#   bash telltale.sh            -> menu
#   bash telltale.sh start      -> build + start the app, open the browser
#   bash telltale.sh help       -> all commands
# Kept compatible with the old bash 3.2 that ships with macOS.

set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
SOUND_TYPES="fan pump gearbox slider valve"
DEV_FILES="-f docker-compose.yml -f docker-compose.dev.yml"
CMD_NAME="bash telltale.sh"

if [ -t 1 ]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; CYAN=$'\033[36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  RED=""; GREEN=""; YELLOW=""; CYAN=""; BOLD=""; RESET=""
fi

say()  { printf '%s\n' "${1:-}"; }
ok()   { printf '  %s[ok]%s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '  %s[!] %s %s\n' "$YELLOW" "$RESET" "$1"; }
info() { printf '  %s...%s  %s\n' "$CYAN" "$RESET" "$1"; }
title() { printf '\n  %s%s%s\n' "$BOLD" "$1" "$RESET"; }
fail() {
  printf '\n  %s[x]  %s%s\n' "$RED" "$1" "$RESET"
  shift
  for tip in "$@"; do printf '       %s- %s%s\n' "$YELLOW" "$tip" "$RESET"; done
  printf '\n  Still stuck? Run  %s doctor  and show the output to a mentor.\n\n' "$CMD_NAME"
  exit 1
}

open_url() {
  [ -n "${TELLTALE_NO_BROWSER:-}" ] && return 0
  if command -v open >/dev/null 2>&1; then open "$1" >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 &
  elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$1" >/dev/null 2>&1
  fi
}

# ------------------------------------------------------------------ docker checks
docker_installed() { command -v docker >/dev/null 2>&1; }
docker_running() { docker info >/dev/null 2>&1; }

compose_version_ok() {
  local v major minor
  v="$(docker compose version --short 2>/dev/null | sed 's/^v//')"
  major="$(printf '%s' "$v" | cut -d. -f1)"; minor="$(printf '%s' "$v" | cut -d. -f2)"
  case "$major" in ''|*[!0-9]*) return 1 ;; esac
  case "$minor" in ''|*[!0-9]*) minor=0 ;; esac
  COMPOSE_VERSION="$v"
  [ "$major" -gt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -ge 24 ]; }
}

ensure_docker() {
  docker_installed || fail "Docker is not installed." \
    "Install Docker Desktop: https://www.docker.com/products/docker-desktop/" \
    "Open it once after installing, then run this again."
  if ! docker_running; then
    if docker info 2>&1 | grep -qi "permission denied"; then
      fail "Docker is running but your user may not use it." \
        "Run:  sudo usermod -aG docker \$USER   then log out and back in."
    fi
    if [ "$(uname)" = "Darwin" ] && [ -d "/Applications/Docker.app" ]; then
      info "Docker Desktop is not running - starting it for you (this can take a minute)..."
      open -a Docker
      local waited=0
      while ! docker_running; do
        [ $waited -ge 180 ] && fail "Docker Desktop did not finish starting." "Open Docker Desktop yourself and wait until it says 'Engine running'."
        printf '.'; sleep 3; waited=$((waited + 3))
      done
      printf '\n'
    else
      fail "Docker is installed but not running." \
        "Mac/Windows: open Docker Desktop and wait until it says 'Engine running'." \
        "Linux: sudo systemctl start docker"
    fi
  fi
  compose_version_ok || fail "Your Docker is too old (need Docker Compose 2.24 or newer)." "Update Docker Desktop (Settings -> Software updates)."
  ok "Docker is running (Compose $COMPOSE_VERSION)"
}

# ------------------------------------------------------------------ .env + ports
ensure_env() {
  if [ ! -f .env ]; then cp .env.example .env; ok "Created .env from .env.example (your local settings file)"; fi
}

get_env() { # key default
  local v=""
  [ -f .env ] && v="$(grep -E "^[[:space:]]*$1[[:space:]]*=" .env | tail -n 1 | cut -d= -f2- | tr -d '\r' | tr -d ' ')"
  if [ -n "$v" ]; then printf '%s' "$v"; else printf '%s' "$2"; fi
}

set_env() { # key value
  if grep -qE "^[[:space:]]*$1[[:space:]]*=" .env; then
    awk -v k="$1" -v v="$2" '{ line=$0; sub(/\r$/, "", line); split(line, a, "="); key=a[1]; gsub(/[[:space:]]/, "", key);
      if (key == k) print k "=" v; else print line }' .env > .env.tmp && mv .env.tmp .env
  else
    printf '%s=%s\n' "$1" "$2" >> .env
  fi
}

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

select_port() { # key preferred -> echoes chosen port
  local start p
  start="$(get_env "$1" "$2")"; p=$start
  while [ "$p" -lt $((start + 50)) ]; do
    if ! port_busy "$p"; then
      [ "$p" != "$start" ] && warn "Port $start is busy (another program uses it) - using $p instead." >&2
      set_env "$1" "$p"; printf '%s' "$p"; return 0
    fi
    p=$((p + 1))
  done
  fail "Could not find a free port near $start." "Close other development servers and try again."
}

# ------------------------------------------------------------------ app state
healthy() { curl -fsS --max-time 3 "http://localhost:$1/api/v1/health" 2>/dev/null | grep -q '"status":"ok"'; }

running_port() {
  local p
  for p in "$(get_env WEB_PORT 8080)" "$(get_env DEV_WEB_PORT 5173)"; do
    if healthy "$p"; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

wait_healthy() { # port
  info "Waiting for the app to come up..."
  local waited=0
  while ! healthy "$1"; do
    [ $waited -ge 300 ] && fail "The app did not start in time." "Look at the logs:  $CMD_NAME logs" "Then try:  $CMD_NAME stop  and  $CMD_NAME start"
    printf '.'; sleep 3; waited=$((waited + 3))
  done
  printf '\n'
}

show_running() { # port dev(0/1)
  local url="http://localhost:$1"
  say ""
  say "  ${GREEN}==============================================================${RESET}"
  say "  ${GREEN} Telltale is running:  $url${RESET}"
  [ "$2" = "1" ] && say "  ${GREEN} EDIT mode: change files in apps/web/src or apps/api and the app updates by itself.${RESET}"
  say "  ${GREEN} API docs:             http://localhost:$(get_env API_PORT 8000)/docs${RESET}"
  say "  ${GREEN} Stop it later with:   $CMD_NAME stop${RESET}"
  say "  ${GREEN}==============================================================${RESET}"
  say ""
  open_url "$url"
}

reload_models() {
  local p
  if p="$(running_port)"; then
    if curl -fsS -X POST --max-time 30 "http://localhost:$p/api/v1/models/reload" >/dev/null 2>&1; then ok "The running app picked up the new model."
    else warn "Could not tell the app to reload. Click 'Reload models' on the Models page."; fi
  fi
}

confirm() {
  case " ${ARGS:-} " in *" --yes "*|*" -y "*) return 0 ;; esac
  printf '  %s [Y/n] ' "$1"; read -r answer
  case "$answer" in ""|[Yy]*) return 0 ;; *) return 1 ;; esac
}

# ------------------------------------------------------------------ commands
cmd_start() { # dev(0/1)
  local dev="$1" web p
  if [ "$dev" = "1" ]; then title "Starting Telltale in EDIT mode"; else title "Starting Telltale"; fi
  ensure_docker; ensure_env
  if p="$(running_port)"; then ok "Already running."; show_running "$p" "$dev"; return; fi
  select_port API_PORT 8000 >/dev/null || exit 1
  if [ "$dev" = "1" ]; then web="$(select_port DEV_WEB_PORT 5173)" || exit 1
  else web="$(select_port WEB_PORT 8080)" || exit 1; fi
  info "Building and starting containers. FIRST TIME: 5-15 minutes (downloads ~1.5 GB). Later: seconds."
  if [ "$dev" = "1" ]; then
    # shellcheck disable=SC2086
    docker compose $DEV_FILES up --build -d --remove-orphans
  else
    docker compose up --build -d --remove-orphans
  fi || fail "Docker could not build or start the app." \
    "Check your internet connection and run this again - it continues where it stopped." \
    "Make sure Docker Desktop has at least 4 GB of memory (Settings -> Resources)." \
    "See the details:  $CMD_NAME logs"
  wait_healthy "$web"
  show_running "$web" "$dev"
  say "  Note: on the very first start the risk model trains itself (about a minute)."
  say "        The dashboard shows a blue 'Getting ready' banner until it is done."
}

cmd_stop() {
  title "Stopping Telltale"
  docker_running || { ok "Docker is not running, so Telltale is not running either."; return; }
  # shellcheck disable=SC2086
  docker compose $DEV_FILES down --remove-orphans
  ok "Stopped. Your trained models and data are kept."
}

cmd_logs() {
  title "Live logs (press Ctrl+C to stop watching - the app keeps running)"
  docker compose logs -f --tail 100
}

cmd_doctor() {
  title "Checking your setup"
  local problems=0 p free_kb h
  if docker_installed; then ok "Docker is installed"; else warn "Docker is NOT installed -> https://www.docker.com/products/docker-desktop/"; problems=$((problems + 1)); fi
  if docker_installed && docker_running; then
    ok "Docker is running"
    if compose_version_ok; then ok "Docker Compose $COMPOSE_VERSION"; else warn "Docker Compose is too old (need 2.24+) -> update Docker Desktop"; problems=$((problems + 1)); fi
  elif docker_installed; then
    warn "Docker is installed but NOT running -> open Docker Desktop"; problems=$((problems + 1))
  fi
  if [ -f .env ]; then ok ".env settings file exists"; else info ".env will be created on first start"; fi
  free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
  if [ "${free_kb:-0}" -ge 10485760 ]; then ok "Free disk space: $((free_kb / 1048576)) GB"
  else warn "Only $((free_kb / 1048576)) GB free disk space - the app needs ~5 GB, sound models ~5 GB more"; problems=$((problems + 1)); fi
  if p="$(running_port)"; then
    ok "App is running at http://localhost:$p"
    h="$(curl -fsS --max-time 3 "http://localhost:$p/api/v1/health" 2>/dev/null)"
    say "       health: $h"
  else
    info "App is not running (start it with  $CMD_NAME start)"
    for p in "$(get_env API_PORT 8000)" "$(get_env WEB_PORT 8080)"; do
      if port_busy "$p"; then info "Port $p is busy - start will pick another one automatically"; else ok "Port $p is free"; fi
    done
  fi
  say ""
  if [ $problems -eq 0 ]; then ok "Everything looks good."; else warn "$problems thing(s) to fix - see the [!] lines above."; fi
}

cmd_train_sensor() {
  title "Retraining the sensor risk model"
  ensure_docker; ensure_env
  if running_port >/dev/null; then docker compose exec api telltale-train-sensor
  else docker compose run --rm --no-deps api telltale-train-sensor; fi || fail "Training failed." "See the messages above."
  reload_models
}

cmd_train_sound() {
  local type="fan" quick=0 a train
  for a in ${ARGS:-}; do
    case "$a" in --quick) quick=1 ;; -*) ;; *) type="$(printf '%s' "$a" | tr '[:upper:]' '[:lower:]')" ;; esac
  done
  case " $SOUND_TYPES " in *" $type "*) ;; *) fail "Unknown machine type '$type'." "Choose one of: $SOUND_TYPES" ;; esac
  title "Training the sound model for: $type"
  say "  This downloads about 1.1 GB of machine recordings (only the first time)"
  say "  and trains a model. Expect 30-90 minutes on a laptop (quick mode: ~10 minutes)."
  [ $quick = 1 ] && say "  Quick mode: faster, less accurate."
  confirm "Continue?" || { say "  Cancelled."; return; }
  ensure_docker; ensure_env
  train="telltale-train-acoustic --machine-type $type"
  [ $quick = 1 ] && train="$train --epochs 5 --max-train-clips 300 --n-models 1"
  docker compose --profile train run --rm trainer sh -c \
    "telltale-download mimii-due --machine $type && $train && telltale-download samples --machine $type" \
    || fail "Sound training failed." "If the download broke, run the same command again - finished downloads are kept." "Make sure you have ~5 GB free disk space."
  ok "Sound model for '$type' is ready."
  reload_models
}

cmd_train_bearing() {
  title "Training the bearing fault model"
  say "  Downloads the bearing recordings (75 MB, only the first time) and trains."
  say "  Takes a few minutes. Afterwards the Sound lab offers demo clips to try."
  ensure_docker; ensure_env
  docker compose --profile train run --rm trainer telltale-train-bearing \
    || fail "Bearing training failed." "If the download broke, run the same command again." "See the messages above for details."
  ok "Bearing model is ready."
  reload_models
}

cmd_open() {
  local p
  if p="$(running_port)"; then show_running "$p" 0; else warn "The app is not running. Start it with  $CMD_NAME start"; fi
}

cmd_help() {
  say ""
  say "  ${BOLD}Telltale helper - run these in a terminal opened in the project folder${RESET}"
  say ""
  say "    $CMD_NAME                    menu with all options"
  say "    $CMD_NAME start              start the app and open it in your browser"
  say "    $CMD_NAME dev                start in EDIT mode (code changes show up live)"
  say "    $CMD_NAME stop               stop the app"
  say "    $CMD_NAME open               open the running app in your browser"
  say "    $CMD_NAME logs               watch what the app is doing (Ctrl+C to quit)"
  say "    $CMD_NAME doctor             check your setup and explain any problem"
  say "    $CMD_NAME train-sound fan    train a sound model (fan, pump, gearbox, slider, valve)"
  say "                                       add --quick for a fast, rough model"
  say "    $CMD_NAME train-bearing      train the bearing fault model"
  say "    $CMD_NAME train-sensor       retrain the sensor risk model"
  say ""
}

cmd_menu() {
  local choice t
  while true; do
    say ""
    say "  ${BOLD}Telltale - what do you want to do?${RESET}"
    say "    1) Start the app"
    say "    2) Start in EDIT mode (your code changes show up live)"
    say "    3) Stop the app"
    say "    4) Check my setup"
    say "    5) Show logs"
    say "    6) Train a sound model"
    say "    7) Train the bearing model"
    say "    8) Retrain the sensor model"
    say "    0) Exit"
    printf '  Type a number and press Enter: '; read -r choice
    case "$choice" in
      1) cmd_start 0; return ;;
      2) cmd_start 1; return ;;
      3) cmd_stop; return ;;
      4) cmd_doctor; return ;;
      5) cmd_logs; return ;;
      6) printf '  Which machine? (%s) [fan]: ' "$SOUND_TYPES"; read -r t; ARGS="${t:-fan}"; cmd_train_sound; return ;;
      7) cmd_train_bearing; return ;;
      8) cmd_train_sensor; return ;;
      0) return ;;
      *) warn "Please type one of the numbers shown." ;;
    esac
  done
}

COMMAND="${1:-menu}"
[ $# -gt 0 ] && shift
ARGS="$*"

case "$COMMAND" in
  menu) cmd_menu ;;
  start) cmd_start 0 ;;
  dev) cmd_start 1 ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start 0 ;;
  open) cmd_open ;;
  logs) cmd_logs ;;
  doctor|status) cmd_doctor ;;
  train-sensor) cmd_train_sensor ;;
  train-sound) cmd_train_sound ;;
  train-bearing) cmd_train_bearing ;;
  help|-h|--help) cmd_help ;;
  *) warn "Unknown command '$COMMAND'."; cmd_help; exit 1 ;;
esac

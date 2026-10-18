# Start here

Welcome! This guide gets Telltale running on your laptop **without any prior experience**. You don't need to know what Docker, a monorepo or an API is. Each term is explained when it first comes up, and there's a [glossary](#9-words-youll-hear) at the end.

It takes about **20 minutes**, and most of that is waiting for downloads.

---

## 1. What is Telltale?

It's a website that watches (pretend) factory machines and warns you before they break. It uses two kinds of clues:

- **Sensor readings:** temperature, speed, torque and tool wear. A model estimates the chance of failure and explains why.
- **Machine sounds:** upload a recording and a model tells you if it sounds abnormal.

When it's running you open it in your browser like any other website. The address is `http://localhost:8080` ("localhost" just means "this computer").

---

## 2. Install one thing: Docker Desktop

**What is Docker?** The app needs Python, Node.js, machine-learning libraries, a web server and more. Instead of installing all of that by hand (and fighting version problems), Docker downloads everything into sealed boxes called **containers** that run the same way on every laptop. You install Docker, and Docker installs the rest.

1. Download **Docker Desktop**: <https://www.docker.com/products/docker-desktop/>
   - **Windows:** run the installer and keep the default options. If it asks to install or update **WSL**, say yes and restart your computer.
   - **Mac:** pick the right download. "Apple chip" is for M1/M2/M3/M4 Macs; "Intel chip" is for older Macs. Not sure? Apple menu → *About This Mac*.
   - **Linux:** follow the Docker Desktop (or Docker Engine) instructions for your distribution.
2. **Open Docker Desktop** and wait until the bottom-left corner says **"Engine running"**. Skip any sign-in prompt; no account is needed.
3. Recommended: in Docker Desktop → **Settings → Resources**, give it at least **4 GB of memory**.

> You do **not** need to install Python, Node.js or anything else. Docker handles it.

Optional but recommended for editing code: **VS Code** (<https://code.visualstudio.com/>).

---

## 3. Get the code

- **Got a zip file?** Unzip it somewhere easy to find, like your Desktop or Documents folder.
- **On GitHub?** Click the green **Code** button → **Download ZIP**, then unzip. If you know git: `git clone <url>`.

You should now have a folder that contains files like `START_HERE.md`, `telltale.cmd` and `telltale.sh`. That folder is "**the project folder**".

---

## 4. Open a terminal in the project folder

A **terminal** is a window where you type commands instead of clicking.

**Windows:**
1. Open the project folder in File Explorer.
2. Click the **address bar** at the top (where the folder path is shown), type `powershell` and press **Enter**.
3. A blue or black window opens, already "inside" the project folder.

**Mac:**
1. Open the **Terminal** app (press Cmd+Space and type "Terminal").
2. Type `cd ` (with a space after it), **drag the project folder into the Terminal window**, then press **Enter**.

**VS Code (any computer):** *File → Open Folder…* → pick the project folder → *Terminal → New Terminal*.

To check you're in the right place, type `dir` (Windows) or `ls` (Mac) and press Enter. You should see `telltale.cmd` and `telltale.sh` in the list.

---

## 5. Start the app

| Windows | Mac / Linux |
|---|---|
| `.\telltale start` | `bash telltale.sh start` |

Type it and press **Enter**.

> **Windows shortcut:** you can also **double-click `telltale.cmd`** in File Explorer. A menu appears; type `1` and press Enter.

**What happens next:**
1. It checks Docker is running, and tries to start it if it isn't.
2. **First time only:** it downloads and builds everything, which takes **5–15 minutes** depending on your internet. You'll see a lot of text scroll by. That's normal.
3. Your browser opens **Telltale**.
4. **First time only:** a blue banner says *"Getting the risk model ready…"* for about a minute while the app trains its model. The page updates by itself.

After that, starting takes a few seconds.

**When you're done for the day:** `.\telltale stop` (Windows) or `bash telltale.sh stop` (Mac/Linux). Your data and trained models are kept.

---

## 6. Cheat sheet

Run these in a terminal opened in the project folder. On Mac/Linux, replace `.\telltale` with `bash telltale.sh`.

| I want to… | Command |
|---|---|
| See a menu of everything | `.\telltale` |
| Start the app | `.\telltale start` |
| Start it so my code changes show up live | `.\telltale dev` |
| Stop it | `.\telltale stop` |
| Open it in the browser again | `.\telltale open` |
| See what it's doing / error messages | `.\telltale logs` (press Ctrl+C to stop watching) |
| Check my setup when something's wrong | `.\telltale doctor` |
| Train a sound model | `.\telltale train-sound fan` (or `pump`, `gearbox`, `slider`, `valve`) |
| Train the bearing model | `.\telltale train-bearing` |
| Practise the pitch | read [`docs/demo-script.md`](docs/demo-script.md) |
| Retrain the sensor model | `.\telltale train-sensor` |

> Training a sound model downloads about **1.1 GB** and can take **30–90 minutes** on a laptop (a few minutes on the AMD GPU). Add `--quick` for a rough model in about 10 minutes: `.\telltale train-sound fan --quick`.

---

## 7. Changing the code

1. Start in **edit mode**: `.\telltale dev`. The browser opens on a different address, usually `http://localhost:5173`.
2. Open the project folder in VS Code.
3. Change a file and **save** it. The browser updates by itself within a second or two.

Normal mode (`.\telltale start`) does **not** pick up changes until you run `.\telltale stop` and then `.\telltale start` again.

---

## 8. Which folder do I work in?

This project is a **monorepo**: one folder that holds several smaller projects that work together. Here are the parts:

```
amd_hackathon/
├── apps/web/      ← the WEBSITE you see (React + TypeScript)
├── apps/api/      ← the SERVER the website talks to (Python + FastAPI)
├── ml/            ← the MACHINE-LEARNING code that trains the models (Python)
├── data/          ← downloaded datasets (created for you, not in git)
├── models/        ← trained models (created for you, not in git)
├── docs/          ← extra documentation
├── telltale.cmd   ← the helper you run on Windows
└── telltale.sh    ← the helper you run on Mac/Linux
```

Pick your lane:

| If you're working on… | Go to | Start with |
|---|---|---|
| **How the website looks** (pages, buttons, charts, colours) | `apps/web/src/` | `pages/` (one file per page), `components/` (reusable pieces), `styles.css` (all colours and styles) |
| **What the server does** (data, rules, alerts) | `apps/api/telltale_api/` | `api/routes/` (one file per group of URLs), `services/` (the logic) |
| **The inspection advice shown to users** (no coding!) | `apps/api/telltale_api/data/playbook.yaml` | Plain text lists; edit the sentences |
| **The pretend machines on the dashboard** (no coding!) | `apps/api/telltale_api/data/fleet.yaml` | One line per machine |
| **The machine-learning models** | `ml/` | `configs/` (settings), `src/telltale_ml/sensor/` and `acoustic/` (code) |

Each of these folders has its own short `README.md` with more detail.

**Leave these alone unless you know what you're doing:** `Dockerfile`s, `docker-compose*.yml`, `pyproject.toml`, `package.json`, and the `models/` and `data/` folders. The models and data are generated automatically, so if you break them you can just retrain.

**Using an AI coding assistant** (Copilot, Cursor, Claude, ChatGPT…)? Tell it: *"Read AGENTS.md first."* That file explains the whole project to it.

---

## 9. Words you'll hear

| Word | What it means here |
|---|---|
| **Terminal** | A window where you type commands. On Windows it's PowerShell, on Mac it's Terminal. |
| **Project folder** | The folder containing `START_HERE.md`. Run all commands from here. |
| **Monorepo** | One folder (repository) that holds several projects: website, server and ML. |
| **Repository / repo** | A project folder tracked by **git**, a tool that saves the history of changes. |
| **Docker** | A tool that runs apps inside sealed **containers**, so you don't install their dependencies yourself. |
| **Container** | A running, sealed box with one part of the app inside (we have one for the website and one for the server). |
| **Image** | The recipe-made "template" a container starts from. Building images is the slow first step. |
| **Dockerfile** | The recipe for building an image. |
| **Docker Compose** | Starts several containers together. Configured in `docker-compose.yml`. |
| **Frontend** | The part you see in the browser (`apps/web`). |
| **Backend / server / API** | The part that does the work and sends data to the frontend (`apps/api`). |
| **API endpoint** | One URL the server answers, e.g. `/api/v1/machines`. Browse them all at the "API docs" link shown when you start. |
| **localhost** | "This computer." `http://localhost:8080` is a website running on your own laptop. |
| **Port** | The number after the colon (`8080`). Several programs can run at once on different ports. The helper picks free ones automatically. |
| **Model** | A file that has *learned* patterns from data and can make predictions. |
| **Training** | The process of creating a model from a dataset. |
| **Dataset** | A collection of example data (sensor readings or sound recordings) used for training. |
| **Inference / prediction** | Using a trained model on new data. |
| **`.env` file** | Your personal settings (like which ports to use). Created automatically; not shared in git. |
| **ONNX** | A file format for models that lets the server run the sound model quickly without PyTorch. |
| **AMD GPU / ROCm** | The graphics card used to train models faster. ROCm is AMD's software for this. Optional: everything also works on a normal laptop CPU. |
| **LLM** | A large language model (like ChatGPT). Optional here: it can write short summaries for technicians. |

---

## 10. Something went wrong?

**First, run** `.\telltale doctor` (Mac/Linux: `bash telltale.sh doctor`). It checks everything and tells you what to fix. Still stuck? Show its output to a mentor or teammate.

<details>
<summary><b>"'.\telltale' is not recognized…" or "No such file"</b></summary>

Your terminal isn't in the project folder. Go back to [step 4](#4-open-a-terminal-in-the-project-folder). `dir` (Windows) or `ls` (Mac) must show `telltale.cmd`.
</details>

<details>
<summary><b>"running scripts is disabled on this system" (Windows)</b></summary>

Use `.\telltale …` (which runs `telltale.cmd`) rather than running `scripts\telltale.ps1` directly. The `.cmd` file takes care of this for you.
</details>

<details>
<summary><b>"Docker is not installed" / "Docker is not running"</b></summary>

Install Docker Desktop ([step 2](#2-install-one-thing-docker-desktop)), open it, and wait for **"Engine running"**. On Windows, if Docker complains about **WSL**, open PowerShell and run `wsl --update`, then restart.
</details>

<details>
<summary><b>The first start failed halfway with a network or "timed out" error</b></summary>

The first start downloads a lot. Run the same command again; it continues where it stopped. A stable Wi-Fi connection helps.
</details>

<details>
<summary><b>"port is already allocated" / "address already in use"</b></summary>

Another program uses that port. The helper normally picks a free port for you. Run `.\telltale stop`, then `.\telltale start` again. The ports it chose are saved in `.env`.
</details>

<details>
<summary><b>The page says "The app's server is not running"</b></summary>

Run `.\telltale start`. If it's already running, wait 30 seconds and refresh. Still there? Check `.\telltale logs` for red error lines.
</details>

<details>
<summary><b>The blue "Getting the risk model ready" banner never goes away</b></summary>

Check `.\telltale logs`. Without internet, the app trains on simulated data instead (the Models page says so), so it should still finish. If it shows a red error banner instead, run `.\telltale train-sensor`.
</details>

<details>
<summary><b>"No sound model is trained yet"</b></summary>

That's expected. Sound models are big, so they aren't trained automatically. Run `.\telltale train-sound fan` (add `--quick` to go faster).
</details>

<details>
<summary><b>My code change doesn't show up</b></summary>

Are you in edit mode (`.\telltale dev`)? In normal mode you must `stop` and `start` again. In edit mode, make sure you **saved** the file. Then look at `.\telltale logs`: a typo in Python code shows up there as an error.
</details>

<details>
<summary><b>"permission denied" on Mac/Linux</b></summary>

Run it with `bash` in front: `bash telltale.sh start`, not `./telltale.sh start`. On Linux, if Docker itself says permission denied, run `sudo usermod -aG docker $USER`, then log out and back in.
</details>

<details>
<summary><b>My laptop is slow or the disk is full</b></summary>

Stop the app when you're not using it (`.\telltale stop`). To free disk space, go to Docker Desktop → **Troubleshoot** → **Clean / Purge data**, or run `docker system prune`. That deletes downloaded images, so the next start takes the long way again.
</details>

---

## 11. For the curious: what's actually running?

```
  Your browser  ──►  web container  ──►  api container  ──►  models/ folder
  (localhost:8080)   (serves the        (Python server:      (trained models,
                      website files)     predictions,         saved on your disk)
                                         alerts, fake
                                         machines)
```

- The **website** (`apps/web`) asks the **server** (`apps/api`) for data every few seconds.
- The **server** loads the **models** trained by the code in `ml/` and uses them to score each machine.
- **Training** a model reads a **dataset** from `data/` and writes a model into `models/`.

Want the full technical picture? See [`README.md`](README.md), [`docs/architecture.md`](docs/architecture.md) and [`AGENTS.md`](AGENTS.md).

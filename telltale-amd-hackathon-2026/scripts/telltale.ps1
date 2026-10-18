# Telltale helper for Windows. Use it through ..\telltale.cmd:
#   .\telltale              -> menu (also what double-clicking telltale.cmd does)
#   .\telltale start        -> build + start the app, open the browser
#   .\telltale help         -> all commands
# Written for Windows PowerShell 5.1 (the version every Windows 10/11 machine has).

param(
    [Parameter(Position = 0)][string]$Command = "menu",
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest = @()
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$SoundTypes = @("fan", "pump", "gearbox", "slider", "valve")
$MinCompose = [version]"2.24"
$DevFiles = @("-f", "docker-compose.yml", "-f", "docker-compose.dev.yml")

# ------------------------------------------------------------------ output helpers
function Say([string]$msg = "", [string]$color = "Gray") { Write-Host $msg -ForegroundColor $color }
function Ok([string]$msg) { Say "  [ok] $msg" "Green" }
function Warn([string]$msg) { Say "  [!]  $msg" "Yellow" }
function Info([string]$msg) { Say "  ...  $msg" "Cyan" }
function Fail([string]$msg, [string[]]$tips = @()) {
    Say ""
    Say "  [x]  $msg" "Red"
    foreach ($t in $tips) { Say "       - $t" "Yellow" }
    Say ""
    Say "  Still stuck? Run  .\telltale doctor  and show the output to a mentor." "Gray"
    exit 1
}
function Title([string]$msg) { Say ""; Say "  $msg" "White"; Say ("  " + ("-" * $msg.Length)) "DarkGray" }

# ------------------------------------------------------------------ docker checks
function Test-DockerInstalled { return $null -ne (Get-Command docker -ErrorAction SilentlyContinue) }

function Test-DockerRunning {
    $old = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
    & docker info *> $null
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    return ($code -eq 0)
}

function Get-ComposeVersion {
    $old = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
    $raw = (& docker compose version --short 2>$null | Out-String).Trim()
    $ErrorActionPreference = $old
    if ($raw -match "(\d+)\.(\d+)") { return [version]"$($Matches[1]).$($Matches[2])" }
    return $null
}

function Ensure-Docker {
    if (-not (Test-DockerInstalled)) {
        Fail "Docker is not installed." @(
            "Install Docker Desktop: https://www.docker.com/products/docker-desktop/",
            "Restart your computer after installing, open Docker Desktop once, then run this again."
        )
    }
    if (-not (Test-DockerRunning)) {
        $exe = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
        if (Test-Path $exe) {
            Info "Docker Desktop is not running - starting it for you (this can take a minute)..."
            Start-Process $exe | Out-Null
            $deadline = (Get-Date).AddSeconds(180)
            while (-not (Test-DockerRunning)) {
                if ((Get-Date) -gt $deadline) {
                    Fail "Docker Desktop did not finish starting." @(
                        "Open Docker Desktop yourself and wait until it says 'Engine running'.",
                        "If it asks you to install or update WSL, accept and restart the computer."
                    )
                }
                Write-Host "." -NoNewline; Start-Sleep -Seconds 3
            }
            Write-Host ""
        }
        else {
            Fail "Docker is installed but not running." @("Open Docker Desktop and wait until it says 'Engine running', then run this again.")
        }
    }
    $v = Get-ComposeVersion
    if ($null -eq $v -or $v -lt $MinCompose) {
        Fail "Your Docker is too old (Compose $v, need $MinCompose or newer)." @("Update Docker Desktop: open it -> Settings -> Software updates.")
    }
    Ok "Docker is running (Compose $v)"
}

# ------------------------------------------------------------------ .env + ports
function Ensure-EnvFile {
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Ok "Created .env from .env.example (your local settings file)"
    }
}

function Get-EnvValue([string]$key, [string]$default) {
    if (-not (Test-Path ".env")) { return $default }
    foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $Root ".env"))) {
        if ($line -match "^\s*$key\s*=\s*(.*)$") { return $Matches[1].Trim() }
    }
    return $default
}

function Set-EnvValue([string]$key, [string]$value) {
    $path = Join-Path $Root ".env"
    $lines = New-Object System.Collections.Generic.List[string]
    $found = $false
    foreach ($line in [System.IO.File]::ReadAllLines($path)) {
        if ($line -match "^\s*$key\s*=") { $lines.Add("$key=$value"); $found = $true } else { $lines.Add($line) }
    }
    if (-not $found) { $lines.Add("$key=$value") }
    [System.IO.File]::WriteAllLines($path, $lines)  # UTF-8 without BOM (Docker Compose dislikes BOMs)
}

function Test-PortFree([int]$port) {
    # Busy if something answers on it...
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect("127.0.0.1", $port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(250) -and $client.Connected) { return $false }
    }
    catch { } finally { $client.Close() }
    # ...or if we cannot bind it ourselves.
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Any, $port)
        $listener.Start()
        return $true
    }
    catch { return $false }
    finally { if ($listener) { $listener.Stop() } }
}

function Select-Port([string]$key, [int]$preferred) {
    $start = [int](Get-EnvValue $key "$preferred")
    for ($p = $start; $p -lt $start + 50; $p++) {
        if (Test-PortFree $p) {
            if ($p -ne $start) { Warn "Port $start is busy (another program uses it) - using $p instead." }
            Set-EnvValue $key "$p"
            return $p
        }
    }
    Fail "Could not find a free port near $start." @("Close other development servers and try again.")
}

# ------------------------------------------------------------------ app state
function Test-Healthy([int]$port) {
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:$port/api/v1/health" -TimeoutSec 3
        return ($r.status -eq "ok")
    }
    catch { return $false }
}

function Get-RunningPort {
    foreach ($key in @("WEB_PORT", "DEV_WEB_PORT")) {
        $default = "8080"; if ($key -eq "DEV_WEB_PORT") { $default = "5173" }
        $p = [int](Get-EnvValue $key $default)
        if (Test-Healthy $p) { return $p }
    }
    return $null
}

function Wait-Healthy([int]$port, [int]$seconds = 300) {
    Info "Waiting for the app to come up..."
    $deadline = (Get-Date).AddSeconds($seconds)
    while (-not (Test-Healthy $port)) {
        if ((Get-Date) -gt $deadline) {
            Fail "The app did not start in time." @("Look at the logs:  .\telltale logs", "Then try:  .\telltale stop  and  .\telltale start")
        }
        Write-Host "." -NoNewline; Start-Sleep -Seconds 3
    }
    Write-Host ""
}

function Show-Running([int]$port, [bool]$dev) {
    $url = "http://localhost:$port"
    Say ""
    Say "  ==============================================================" "Green"
    Say "   Telltale is running:  $url" "Green"
    if ($dev) { Say "   EDIT mode: change files in apps\web\src or apps\api and the app updates by itself." "Green" }
    Say "   API docs:             http://localhost:$(Get-EnvValue 'API_PORT' '8000')/docs" "Green"
    Say "   Stop it later with:   .\telltale stop" "Green"
    Say "  ==============================================================" "Green"
    Say ""
    if (-not $env:TELLTALE_NO_BROWSER) { Start-Process $url | Out-Null }
}

function Invoke-Reload {
    $p = Get-RunningPort
    if ($p) {
        try { Invoke-RestMethod -Method Post -Uri "http://localhost:$p/api/v1/models/reload" -TimeoutSec 30 | Out-Null; Ok "The running app picked up the new model." }
        catch { Warn "Could not tell the app to reload. Click 'Reload models' on the Models page." }
    }
}

# ------------------------------------------------------------------ commands
function Cmd-Start([bool]$dev) {
    Title $(if ($dev) { "Starting Telltale in EDIT mode" } else { "Starting Telltale" })
    Ensure-Docker
    Ensure-EnvFile
    $running = Get-RunningPort
    if ($running) { Ok "Already running."; Show-Running $running $dev; return }

    Select-Port "API_PORT" 8000 | Out-Null
    $webKey = "WEB_PORT"; $webDefault = 8080
    if ($dev) { $webKey = "DEV_WEB_PORT"; $webDefault = 5173 }
    $web = Select-Port $webKey $webDefault

    Info "Building and starting containers. FIRST TIME: 5-15 minutes (downloads ~1.5 GB). Later: seconds."
    $composeArgs = @()
    if ($dev) { $composeArgs += $DevFiles }
    $composeArgs += @("up", "--build", "-d", "--remove-orphans")
    & docker compose @composeArgs
    if ($LASTEXITCODE -ne 0) {
        Fail "Docker could not build or start the app." @(
            "Check your internet connection (the first build downloads a lot) and run this again - it continues where it stopped.",
            "Make sure Docker Desktop has at least 4 GB of memory (Settings -> Resources).",
            "See the details:  .\telltale logs"
        )
    }
    Wait-Healthy $web
    Show-Running $web $dev
    Say "  Note: on the very first start the risk model trains itself (about a minute)." "Gray"
    Say "        The dashboard shows a blue 'Getting ready' banner until it is done." "Gray"
}

function Cmd-Stop {
    Title "Stopping Telltale"
    if (-not (Test-DockerRunning)) { Ok "Docker is not running, so Telltale is not running either."; return }
    & docker compose @DevFiles down --remove-orphans
    Ok "Stopped. Your trained models and data are kept."
}

function Cmd-Logs {
    Title "Live logs (press Ctrl+C to stop watching - the app keeps running)"
    & docker compose logs -f --tail 100
}

function Cmd-Doctor {
    Title "Checking your setup"
    $problems = 0
    if (Test-DockerInstalled) { Ok "Docker is installed" } else { Warn "Docker is NOT installed -> https://www.docker.com/products/docker-desktop/"; $problems++ }
    if ((Test-DockerInstalled) -and (Test-DockerRunning)) {
        Ok "Docker is running"
        $v = Get-ComposeVersion
        if ($v -and $v -ge $MinCompose) { Ok "Docker Compose $v" } else { Warn "Docker Compose $v is too old (need $MinCompose+) -> update Docker Desktop"; $problems++ }
    }
    elseif (Test-DockerInstalled) { Warn "Docker is installed but NOT running -> open Docker Desktop"; $problems++ }

    if (Test-Path ".env") { Ok ".env settings file exists" } else { Info ".env will be created on first start" }

    $drive = (Get-Item $Root).PSDrive
    $freeGb = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGb -ge 10) { Ok "Free disk space: $freeGb GB" } else { Warn "Only $freeGb GB free disk space - the app needs ~5 GB, sound models ~5 GB more"; $problems++ }

    $p = Get-RunningPort
    if ($p) {
        Ok "App is running at http://localhost:$p"
        try {
            $h = Invoke-RestMethod -Uri "http://localhost:$p/api/v1/health" -TimeoutSec 3
            $sensorState = "missing"; if ($h.models.sensor) { $sensorState = "loaded" }
            Ok "Sensor model: $sensorState ($($h.sensor_bootstrap.status))"
            $types = ($h.acoustic_machine_types -join ", "); if (-not $types) { $types = "none yet (.\telltale train-sound fan)" }
            Ok "Sound models: $types"
            $bearing = "not trained (.\telltale train-bearing)"; if ($h.models.bearing) { $bearing = "loaded" }
            Ok "Bearing model: $bearing"
        }
        catch { }
    }
    else {
        Info "App is not running (start it with  .\telltale start)"
        foreach ($key in @("API_PORT", "WEB_PORT")) {
            $default = "8000"; if ($key -eq "WEB_PORT") { $default = "8080" }
            $port = [int](Get-EnvValue $key $default)
            if (Test-PortFree $port) { Ok "Port $port ($key) is free" } else { Info "Port $port ($key) is busy - start will pick another one automatically" }
        }
    }
    Say ""
    if ($problems -eq 0) { Ok "Everything looks good." } else { Warn "$problems thing(s) to fix - see the [!] lines above." }
}

function Confirm-Step([string]$question) {
    if ($Rest -contains "--yes" -or $Rest -contains "-y") { return $true }
    $answer = Read-Host "  $question [Y/n]"
    return ($answer -eq "" -or $answer -match "^[Yy]")
}

function Cmd-TrainSensor {
    Title "Retraining the sensor risk model"
    Ensure-Docker; Ensure-EnvFile
    if (Get-RunningPort) { & docker compose exec api telltale-train-sensor }
    else { & docker compose run --rm --no-deps api telltale-train-sensor }
    if ($LASTEXITCODE -ne 0) { Fail "Training failed." @("See the messages above.") }
    Invoke-Reload
}

function Cmd-TrainSound {
    $type = "fan"
    $positional = @($Rest | Where-Object { $_ -notmatch "^-" })
    if ($positional.Count -gt 0) { $type = $positional[0].ToLower() }
    if ($SoundTypes -notcontains $type) { Fail "Unknown machine type '$type'." @("Choose one of: $($SoundTypes -join ', ')") }
    $quick = $Rest -contains "--quick"

    Title "Training the sound model for: $type"
    Say "  This downloads about 1.1 GB of machine recordings (only the first time)" "Gray"
    Say "  and trains a model. Expect 30-90 minutes on a laptop$(if ($quick) { ' (quick mode: ~10 minutes, less accurate)' })." "Gray"
    if (-not (Confirm-Step "Continue?")) { Say "  Cancelled."; return }
    Ensure-Docker; Ensure-EnvFile
    $train = "telltale-train-acoustic --machine-type $type"
    if ($quick) { $train += " --epochs 5 --max-train-clips 300 --n-models 1" }
    $script = "telltale-download mimii-due --machine $type && $train && telltale-download samples --machine $type"
    & docker compose --profile train run --rm trainer sh -c $script
    if ($LASTEXITCODE -ne 0) {
        Fail "Sound training failed." @("If the download broke, just run the same command again - finished downloads are kept.", "Make sure you have ~5 GB free disk space.")
    }
    Ok "Sound model for '$type' is ready."
    Invoke-Reload
}

function Cmd-TrainBearing {
    Title "Training the bearing fault model"
    Say "  Downloads the bearing recordings (75 MB, only the first time) and trains." "Gray"
    Say "  Takes a few minutes. Afterwards the Sound lab offers demo clips to try." "Gray"
    Ensure-Docker; Ensure-EnvFile
    & docker compose --profile train run --rm trainer telltale-train-bearing
    if ($LASTEXITCODE -ne 0) {
        Fail "Bearing training failed." @("If the download broke, run the same command again.", "See the messages above for details.")
    }
    Ok "Bearing model is ready."
    Invoke-Reload
}

function Cmd-Open {
    $p = Get-RunningPort
    if ($p) { Show-Running $p $false } else { Warn "The app is not running. Start it with  .\telltale start" }
}

function Cmd-Help {
    Say ""
    Say "  Telltale helper - run these in a terminal opened in the project folder" "White"
    Say ""
    Say "    .\telltale                    menu with all options"
    Say "    .\telltale start              start the app and open it in your browser"
    Say "    .\telltale dev                start in EDIT mode (code changes show up live)"
    Say "    .\telltale stop               stop the app"
    Say "    .\telltale open               open the running app in your browser"
    Say "    .\telltale logs               watch what the app is doing (Ctrl+C to quit)"
    Say "    .\telltale doctor             check your setup and explain any problem"
    Say "    .\telltale train-sound fan    train a sound model (fan, pump, gearbox, slider, valve)"
    Say "                                  add --quick for a fast, rough model"
    Say "    .\telltale train-bearing      train the bearing fault model"
    Say "    .\telltale train-sensor       retrain the sensor risk model"
    Say ""
}

function Cmd-Menu {
    while ($true) {
        Say ""
        Say "  Telltale - what do you want to do?" "White"
        Say "    1) Start the app"
        Say "    2) Start in EDIT mode (your code changes show up live)"
        Say "    3) Stop the app"
        Say "    4) Check my setup"
        Say "    5) Show logs"
        Say "    6) Train a sound model"
        Say "    7) Train the bearing model"
        Say "    8) Retrain the sensor model"
        Say "    0) Exit"
        $choice = Read-Host "  Type a number and press Enter"
        switch ($choice) {
            "1" { Cmd-Start $false; return }
            "2" { Cmd-Start $true; return }
            "3" { Cmd-Stop; return }
            "4" { Cmd-Doctor; return }
            "5" { Cmd-Logs; return }
            "6" {
                $t = Read-Host "  Which machine? ($($SoundTypes -join ', ')) [fan]"
                if ($t) { $script:Rest = @($t) } else { $script:Rest = @("fan") }
                Cmd-TrainSound; return
            }
            "7" { Cmd-TrainBearing; return }
            "8" { Cmd-TrainSensor; return }
            "0" { return }
            default { Warn "Please type one of the numbers shown." }
        }
    }
}

switch ($Command.ToLower()) {
    "menu" { Cmd-Menu }
    "start" { Cmd-Start $false }
    "dev" { Cmd-Start $true }
    "stop" { Cmd-Stop }
    "restart" { Cmd-Stop; Cmd-Start $false }
    "open" { Cmd-Open }
    "logs" { Cmd-Logs }
    "doctor" { Cmd-Doctor }
    "status" { Cmd-Doctor }
    "train-sensor" { Cmd-TrainSensor }
    "train-sound" { Cmd-TrainSound }
    "train-bearing" { Cmd-TrainBearing }
    "help" { Cmd-Help }
    default { Warn "Unknown command '$Command'."; Cmd-Help; exit 1 }
}

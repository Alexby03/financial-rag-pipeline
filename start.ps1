#Requires -Version 5.1
<#
.SYNOPSIS
  Start Docker Compose (Qdrant + MinIO), the FastAPI server, then Streamlit.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
Set-Location -LiteralPath $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ApiUrl = "http://127.0.0.1:8000"
$UiUrl = "http://127.0.0.1:8501"

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-UrlOk([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch {
        return $false
    }
}

function Wait-UrlOk {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSec = 60,
        [string]$Name = $Url
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-UrlOk $Url) {
            return
        }
        Start-Sleep -Milliseconds 400
    }
    throw "Timed out waiting for $Name ($Url)"
}

function Test-PortOpen([int]$Port) {
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $client.Connect("127.0.0.1", $Port)
        $client.Close()
        return $true
    }
    catch {
        return $false
    }
}

function Start-DevWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string[]]$PythonArgs
    )
    $argLine = ($PythonArgs | ForEach-Object {
            if ($_ -match '[ "]') { '"' + ($_ -replace '"', '""') + '"' } else { $_ }
        }) -join " "
    # cmd start: first quoted arg is the window title. /k keeps the window if the process exits.
    $line = 'start "{0}" /D "{1}" cmd /k "{2} {3}"' -f $Title, $Root, $Python, $argLine
    & cmd.exe /c $line
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker is not on PATH. Install Docker Desktop and retry."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. From the repo root run: python -m venv .venv  then  .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}
if (-not (Test-Path -LiteralPath (Join-Path $Root ".env"))) {
    throw "Missing .env. Copy .env.example and set OPENAI_API_KEY plus MISTRAL_OCR_ENDPOINT / MISTRAL_OCR_API_KEY."
}

Write-Step "Starting Docker Compose (qdrant, minio, minio-init, redis)"
& docker compose up -d qdrant minio minio-init redis
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed (exit $LASTEXITCODE)."
}

Write-Step "Waiting for Qdrant, MinIO, and Redis"
Wait-UrlOk -Url "http://127.0.0.1:6333/readyz" -Name "Qdrant" -TimeoutSec 90
Wait-UrlOk -Url "http://127.0.0.1:9000/minio/health/live" -Name "MinIO" -TimeoutSec 90
while (-not (Test-PortOpen 6379)) {
    Start-Sleep -Milliseconds 400
}

Write-Step "Ensuring Qdrant collection exists"
& $Python -m app.scripts.init_qdrant
if ($LASTEXITCODE -ne 0) {
    throw "Qdrant init failed (exit $LASTEXITCODE)."
}

$apiAlreadyUp = Test-UrlOk "$ApiUrl/health"
if ($apiAlreadyUp) {
    Write-Host "API already healthy at $ApiUrl - not starting another uvicorn."
}
elseif (Test-PortOpen 8000) {
    throw "Port 8000 is in use but /health is not answering. Stop the leftover uvicorn and retry."
}
else {
    Write-Step "Starting uvicorn in a new window"
    Start-DevWindow -Title "RAG-API-8000" -PythonArgs @(
        "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"
    )
    Wait-UrlOk -Url "$ApiUrl/health" -Name "uvicorn /health" -TimeoutSec 60
}

Write-Step "API health"
$health = Invoke-WebRequest -Uri "$ApiUrl/health" -UseBasicParsing -TimeoutSec 10
Write-Host $health.Content.Trim()

$uiAlreadyUp = Test-UrlOk "$UiUrl/_stcore/health"
if ($uiAlreadyUp) {
    Write-Host "Streamlit already running at $UiUrl - not starting another instance."
}
elseif (Test-PortOpen 8501) {
    throw "Port 8501 is in use but Streamlit health is not answering. Stop the leftover process and retry."
}
else {
    Write-Step "Starting Streamlit in a new window"
    Start-DevWindow -Title "RAG-UI-8501" -PythonArgs @(
        "-m", "streamlit", "run", "streamlit_app.py", "--server.headless", "true", "--server.port", "8501"
    )
    Wait-UrlOk -Url "$UiUrl/_stcore/health" -Name "Streamlit" -TimeoutSec 60
}

Write-Host ""
Write-Host "API  $ApiUrl/docs"
Write-Host "UI   $UiUrl"
Write-Host "Done."

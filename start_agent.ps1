<#
.SYNOPSIS
    Automated startup script for local Qwen3.6-35B-A3B TurboQuant server and Hermes Agent.

.DESCRIPTION
    Launches llama-server with custom TurboQuant CUDA settings (-ctk turbo4 -ctv turbo3),
    waits for the server health check endpoint to return HTTP 200 OK, and then launches
    the local desktop coding agent.

.EXAMPLE
    .\start_agent.ps1 -ModelPath "C:\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
#>

param (
    [string]$ServerExe = "C:\llama-turboquant\build\bin\Release\llama-server.exe",
    [string]$ModelPath = "C:\models\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
    [string]$AgentExe  = "$env:LOCALAPPDATA\Programs\Hermes\Hermes.exe",
    [int]$ContextSize = 131072,
    [int]$Port        = 8080,
    [int]$FitTarget   = 400,
    [int]$UBatch      = 2048   # prefill batch: 2048 ~doubles agent prompt-processing (482->878 t/s) vs default 512, decode unchanged, fits 128K KV in 6GB. Drop to 1024 if OOM.
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Qwen3.6-35B-A3B Local Agent Launcher (128K TurboQuant)   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check if llama-server is already running
$existing = Get-Process llama-server -ErrorAction SilentlyContinue

if (-not $existing) {
    if (-not (Test-Path $ServerExe)) {
        Write-Error "Server executable not found at: $ServerExe`nPlease update -ServerExe parameter or verify build path."
        exit 1
    }

    if (-not (Test-Path $ModelPath)) {
        Write-Warning "Model file not found at default path: $ModelPath"
        Write-Warning "Please specify your model path: .\start_agent.ps1 -ModelPath 'C:\path\to\model.gguf'"
        exit 1
    }

    Write-Host "[+] Starting llama-server with TurboQuant (KV: turbo4/turbo3, Context: $ContextSize, uBatch: $UBatch)..." -ForegroundColor Yellow

    Start-Process -FilePath $ServerExe -ArgumentList @(
        "-m", "`"$ModelPath`"",
        "-c", "$ContextSize",
        "--parallel", "1",
        "--port", "$Port",
        "-fitt", "$FitTarget",
        "-ub", "$UBatch",
        "-ctk", "turbo4",
        "-ctv", "turbo3"
    ) -WindowStyle Minimized

    Write-Host "[+] Waiting for server to initialize..." -NoNewline

    $ready = $false
    for ($i = 0; $i -lt 36; $i++) {
        Start-Sleep -Seconds 5
        try {
            $res = Invoke-RestMethod -Uri "http://localhost:$Port/health" -TimeoutSec 2 -ErrorAction Stop
            if ($res.status -eq "ok" -or $res -match "ok") {
                $ready = $true
                break
            }
        } catch {
            Write-Host "." -NoNewline
        }
    }

    Write-Host ""
    if ($ready) {
        Write-Host "[✔] llama-server is UP and READY on http://localhost:$Port" -ForegroundColor Green
    } else {
        Write-Warning "[!] Server startup took longer than expected. Continuing launcher..."
    }
} else {
    Write-Host "[✔] llama-server is already running (PID: $($existing.Id))." -ForegroundColor Green
}

# 2. Launch Agent Application
if (Test-Path $AgentExe) {
    Write-Host "[+] Launching Agent application..." -ForegroundColor Yellow
    Start-Process -FilePath $AgentExe
    Write-Host "[✔] Agent launched successfully!" -ForegroundColor Green
} else {
    Write-Host "[i] Agent app not found at default location ($AgentExe)." -ForegroundColor Gray
    Write-Host "[i] Connect your agent (Hermes, OpenHands, AutoGen, etc.) to OpenAI endpoint: http://localhost:$Port/v1" -ForegroundColor Cyan
}

Write-Host "============================================================" -ForegroundColor Cyan

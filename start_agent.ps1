<#
.SYNOPSIS
    Automated startup script for local Qwen3.6-35B-A3B TurboQuant server and coding agent.

.DESCRIPTION
    Launches llama-server with TurboQuant CUDA KV-cache (-ctk turbo4 -ctv turbo3) and the
    tuned prefill batch (-ub 2048), waits for the /health endpoint, then launches the agent app.

    NOTE: ASCII-only on purpose. PowerShell 5.1 (the default "Run with PowerShell" right-click
    handler) misreads UTF-8-without-BOM files that contain non-ASCII characters, which causes
    parser errors. Keep this file ASCII.

.EXAMPLE
    .\start_agent.ps1
    .\start_agent.ps1 -ModelPath "C:\path\to\model.gguf" -AgentExe "C:\path\to\Agent.exe"
#>

param (
    [string]$ServerExe = "C:\llama-turboquant\build\bin\Release\llama-server.exe",
    [string]$ModelPath = "",    # blank -> auto-detect Q3_K_XL GGUF in the HF cache / C:\models (or pass explicitly)
    [string]$AgentExe  = "",    # blank -> auto-detect Hermes.exe under %LOCALAPPDATA%\hermes (or pass explicitly)
    [int]$ContextSize  = 131072,
    [int]$Port         = 8080,
    [int]$FitTarget    = 400,
    [int]$UBatch       = 2048   # prefill batch: 2048 ~doubles prompt-processing (482->878 t/s) vs default 512; decode unchanged; fits 128K KV in 6GB. Drop to 1024 if a config OOMs.
)

# --- Auto-detect model + agent when not passed explicitly ---
if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    $searchDirs = @("$env:USERPROFILE\.cache\huggingface\hub", "C:\models") | Where-Object { Test-Path $_ }
    $hit = Get-ChildItem $searchDirs -Recurse -Filter "*A3B*Q3_K_XL*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $ModelPath = $hit.FullName }
}
if (-not $AgentExe -or -not (Test-Path $AgentExe)) {
    $hermesRoot = "$env:LOCALAPPDATA\hermes"
    if (Test-Path $hermesRoot) {
        $hit = Get-ChildItem $hermesRoot -Recurse -Filter "Hermes.exe" -ErrorAction SilentlyContinue |
               Where-Object { $_.FullName -notmatch '\.bak' } | Select-Object -First 1
        if ($hit) { $AgentExe = $hit.FullName }
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Qwen3.6-35B-A3B Local Agent Launcher (128K TurboQuant)   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Start llama-server if not already running
$existing = Get-Process llama-server -ErrorAction SilentlyContinue

if (-not $existing) {
    if (-not (Test-Path $ServerExe)) {
        Write-Host "[ERROR] Server executable not found at: $ServerExe" -ForegroundColor Red
        Write-Host "        Fix -ServerExe or verify your build path." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    if (-not (Test-Path $ModelPath)) {
        Write-Host "[ERROR] Model file not found at: $ModelPath" -ForegroundColor Red
        Write-Host "        Pass the correct path: .\start_agent.ps1 -ModelPath 'C:\path\to\model.gguf'" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "[+] Starting llama-server (KV turbo4/turbo3, ctx $ContextSize, uBatch $UBatch)..." -ForegroundColor Yellow

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

    # Cold 16GB load + warmup (experts streamed over PCIe on 6GB) can exceed 3 min. Wait up to ~5.
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 5
        try {
            $res = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction Stop
            if ($res.status -eq "ok" -or "$res" -match "ok") { $ready = $true; break }
        } catch {
            Write-Host "." -NoNewline
        }
    }

    Write-Host ""
    if ($ready) {
        Write-Host "[OK] llama-server is UP on http://localhost:$Port" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Server startup took longer than expected. Continuing..." -ForegroundColor Yellow
    }
} else {
    Write-Host "[OK] llama-server already running (PID $($existing.Id))." -ForegroundColor Green
}

# 2. Launch the agent application (optional)
if (Test-Path $AgentExe) {
    Write-Host "[+] Launching agent application..." -ForegroundColor Yellow
    Start-Process -FilePath $AgentExe
    Write-Host "[OK] Agent launched." -ForegroundColor Green
} else {
    Write-Host "[i] Agent app not found at: $AgentExe" -ForegroundColor Gray
    Write-Host "[i] Server is ready. Point your agent (Hermes / OpenHands / etc.) at:" -ForegroundColor Cyan
    Write-Host "    OpenAI endpoint  ->  http://localhost:$Port/v1" -ForegroundColor Cyan
    Write-Host "[i] Or re-run with:  .\start_agent.ps1 -AgentExe 'C:\path\to\YourAgent.exe'" -ForegroundColor Gray
}

Write-Host "============================================================" -ForegroundColor Cyan
Read-Host "Press Enter to close this window"

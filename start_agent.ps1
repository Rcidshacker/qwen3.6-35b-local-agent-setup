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
    [string]$ModelPath = "",    # blank -> auto-detect: prefer faster REAP-28B, else the 35B (or pass explicitly)
    [string]$AgentExe  = "",    # blank -> auto-detect Hermes.exe under %LOCALAPPDATA%\hermes (or pass explicitly)
    [int]$ContextSize  = 131072,
    [int]$Port         = 8080,
    [int]$FitTarget    = 400,
    [int]$UBatch       = 0      # 0 = auto (4096 for REAP-28B, its optimum; 2048 for the 35B). Override with any explicit value.
)

# --- Auto-detect model + agent when not passed explicitly ---
if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    $searchDirs = @("$env:USERPROFILE\llama_models", "$env:USERPROFILE\.cache\huggingface\hub", "C:\models") | Where-Object { Test-Path $_ }
    # Prefer the faster REAP-28B (matches/beats 35B quality at +40-70% speed on this 6GB rig); fall back to the 35B.
    $hit = Get-ChildItem $searchDirs -Recurse -Filter "*28B*REAP*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $hit) { $hit = Get-ChildItem $searchDirs -Recurse -Filter "*A3B*Q3_K_*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1 }
    if ($hit) { $ModelPath = $hit.FullName }
}
# Auto ubatch: REAP-28B fits -ub 4096 (its optimum); the larger 35B tops out at 2048.
if ($UBatch -le 0) {
    if ($ModelPath -match '28B|REAP') { $UBatch = 4096 } else { $UBatch = 2048 }
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

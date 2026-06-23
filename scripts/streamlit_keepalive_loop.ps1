$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectRoot "logs"
$LoopLogPath = Join-Path $LogDir "streamlit-keepalive-loop.log"
$PidPath = Join-Path $LogDir "streamlit-keepalive-loop.pid"
$EnsureScript = Join-Path $PSScriptRoot "ensure_streamlit.ps1"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Write-LoopLog {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LoopLogPath -Value "$Timestamp $Message"
}

if (Test-Path $PidPath) {
    $ExistingPid = Get-Content $PidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ExistingPid -and (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue)) {
        Write-LoopLog "Keep-alive loop already running with PID $ExistingPid."
        exit 0
    }
}

Set-Content -Path $PidPath -Value $PID
Write-LoopLog "Starting Streamlit keep-alive loop with PID $PID."

while ($true) {
    try {
        & $EnsureScript
    } catch {
        Write-LoopLog "Keep-alive check failed: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds 300
}

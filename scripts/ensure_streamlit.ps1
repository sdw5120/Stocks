$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogDir "streamlit-keepalive.log"
$Port = 8501

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Write-KeepAliveLog {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "$Timestamp $Message"
}

$Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -gt 0 } |
    Select-Object -First 1

if ($Listener) {
    Write-KeepAliveLog "Streamlit already listening on port $Port with PID $($Listener.OwningProcess)."
    exit 0
}

Write-KeepAliveLog "Streamlit is not listening on port $Port. Starting dashboard."
Start-Process `
    -FilePath "py" `
    -ArgumentList "-3", "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1", "--server.port", "$Port", "--server.headless", "true" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden

Start-Sleep -Seconds 8
$Started = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -gt 0 } |
    Select-Object -First 1

if ($Started) {
    Write-KeepAliveLog "Streamlit started on port $Port with PID $($Started.OwningProcess)."
    exit 0
}

Write-KeepAliveLog "Streamlit did not start successfully."
exit 1

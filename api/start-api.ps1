# start-api.ps1 — start the Breast API EC2 instance and wait until it answers.
#   Usage:  ./start-api.ps1
$ErrorActionPreference = "Stop"

$INSTANCE = "i-015b6f5692790d7f3"
$REGION   = "ap-southeast-1"
$HEALTH   = "http://122.248.246.94:8000/health"

# Locate the AWS CLI (PATH first, then the default install location).
$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }
if (-not (Test-Path $aws)) { Write-Error "AWS CLI not found. Install it or edit `$aws in this script."; exit 1 }

Write-Host "Starting $INSTANCE ..." -ForegroundColor Cyan
& $aws ec2 start-instances --instance-ids $INSTANCE --region $REGION | Out-Null
& $aws ec2 wait instance-running --instance-ids $INSTANCE --region $REGION
Write-Host "Instance is running. Waiting for the container/API to come up..." -ForegroundColor Cyan

$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
  try {
    $r = Invoke-WebRequest -Uri $HEALTH -TimeoutSec 5 -UseBasicParsing
    if ($r.StatusCode -eq 200) {
      Write-Host "API is LIVE: $($r.Content)" -ForegroundColor Green
      Write-Host ("URL: {0}" -f ($HEALTH -replace '/health$',''))
      Write-Host ("Docs: {0}/docs" -f ($HEALTH -replace '/health$',''))
      exit 0
    }
  } catch { }
  Start-Sleep 10
}
Write-Warning "API did not respond within 5 min. It may still be starting — try:  curl $HEALTH"

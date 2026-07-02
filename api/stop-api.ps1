# stop-api.ps1 — stop the Breast API EC2 instance to pause compute billing.
#   Usage:  ./stop-api.ps1
$ErrorActionPreference = "Stop"

$INSTANCE = "i-015b6f5692790d7f3"
$REGION   = "ap-southeast-1"

$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }
if (-not (Test-Path $aws)) { Write-Error "AWS CLI not found. Install it or edit `$aws in this script."; exit 1 }

Write-Host "Stopping $INSTANCE ..." -ForegroundColor Cyan
& $aws ec2 stop-instances --instance-ids $INSTANCE --region $REGION | Out-Null
& $aws ec2 wait instance-stopped --instance-ids $INSTANCE --region $REGION
Write-Host "Instance stopped. Compute billing paused (EBS storage ~`$2.88/mo remains)." -ForegroundColor Green
Write-Host "The Elastic IP is unchanged — run start-api.ps1 to bring it back on the same URL."

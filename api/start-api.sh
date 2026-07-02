#!/usr/bin/env bash
# start-api.sh — start the Breast API EC2 instance and wait until it answers.
#   Usage:  chmod +x start-api.sh && ./start-api.sh
set -euo pipefail

INSTANCE="i-015b6f5692790d7f3"
REGION="ap-southeast-1"
HEALTH="http://122.248.246.94:8000/health"
URL="${HEALTH%/health}"

command -v aws >/dev/null 2>&1 || { echo "AWS CLI not found. Install it: brew install awscli"; exit 1; }

echo "Starting $INSTANCE ..."
aws ec2 start-instances --instance-ids "$INSTANCE" --region "$REGION" >/dev/null
aws ec2 wait instance-running --instance-ids "$INSTANCE" --region "$REGION"
echo "Instance is running. Waiting for the container/API to come up..."

for _ in $(seq 1 30); do
  if curl -sf --max-time 5 "$HEALTH" >/dev/null 2>&1; then
    echo "API is LIVE: $(curl -s "$HEALTH")"
    echo "URL:  $URL"
    echo "Docs: $URL/docs"
    exit 0
  fi
  sleep 10
done
echo "API did not respond within ~5 min. It may still be starting — try:  curl $HEALTH"

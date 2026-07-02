#!/usr/bin/env bash
# stop-api.sh — stop the Breast API EC2 instance to pause compute billing.
#   Usage:  chmod +x stop-api.sh && ./stop-api.sh
set -euo pipefail

INSTANCE="i-015b6f5692790d7f3"
REGION="ap-southeast-1"

command -v aws >/dev/null 2>&1 || { echo "AWS CLI not found. Install it: brew install awscli"; exit 1; }

echo "Stopping $INSTANCE ..."
aws ec2 stop-instances --instance-ids "$INSTANCE" --region "$REGION" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$INSTANCE" --region "$REGION"
echo "Instance stopped. Compute billing paused (EBS storage ~\$2.88/mo remains)."
echo "The Elastic IP is unchanged — run ./start-api.sh to bring it back on the same URL."

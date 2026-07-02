#!/bin/bash
# EC2 user-data — first-boot bootstrap for the Breast Lesion Segmentation API.
#
# Paste this into the "User data" field when launching an Amazon Linux 2023
# instance (Advanced details → User data), OR pass it via:
#     aws ec2 run-instances ... --user-data file://api/ec2-userdata.sh
#
# Requirements on the instance:
#   * AMI: Amazon Linux 2023 (has aws CLI v2 preinstalled)
#   * IAM instance profile with AmazonEC2ContainerRegistryReadOnly (your Step 2 role)
#   * Security group inbound: TCP 8000 (and 22 for SSH)
#
# Logs: everything below is written to /var/log/user-data.log on the instance.
set -euxo pipefail
exec > >(tee -a /var/log/user-data.log) 2>&1

REGION="ap-southeast-1"
ACCOUNT="177989594048"
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ECR}/breast-api:latest"

echo "[user-data] installing Docker..."
dnf install -y docker
systemctl enable --now docker
# let ec2-user run docker without sudo after the next login
usermod -aG docker ec2-user || true

echo "[user-data] logging in to ECR..."
# retry: the network / IAM credentials can take a few seconds to be ready
for i in 1 2 3 4 5; do
  if aws ecr get-login-password --region "$REGION" \
       | docker login --username AWS --password-stdin "$ECR"; then
    break
  fi
  echo "[user-data] ECR login attempt $i failed; retrying in 10s..."
  sleep 10
done

echo "[user-data] pulling image $IMAGE ..."
docker pull "$IMAGE"

echo "[user-data] starting container..."
docker rm -f breast-api 2>/dev/null || true
docker run -d --name breast-api \
  --restart unless-stopped \
  -p 8000:8000 \
  "$IMAGE"

echo "[user-data] waiting for health..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[user-data] API is healthy: $(curl -s http://localhost:8000/health)"
    break
  fi
  sleep 3
done

echo "[user-data] done. API on port 8000 — http://<PUBLIC_IP>:8000/docs"

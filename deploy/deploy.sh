#!/usr/bin/env bash
# Deploy to remote GPU machine via SSH, build, and run tests
# Usage: bash deploy/deploy.sh

set -euo pipefail

REMOTE="shuyua01@10.190.0.91"
REMOTE_PATH="/home/shuyua01/Development/cuda-rmsnorm"

echo "=== Deploying cuda-rmsnorm to $REMOTE:$REMOTE_PATH ==="

# rsync everything to remote machine
rsync -avz \
    --exclude='__pycache__' \
    --exclude='build/' \
    --exclude='.git/' \
    --exclude='*.pyc' \
    --exclude='ncu-reports/' \
    ./ "$REMOTE:$REMOTE_PATH"

echo "=== Files synced ==="

# Build on remote machine
echo "=== Building on remote ==="
ssh "$REMOTE" "cd $REMOTE_PATH && bash deploy/remote_build.sh"

# Run tests
echo "=== Running tests on remote ==="
ssh "$REMOTE" "cd $REMOTE_PATH && bash deploy/run_tests.sh"

echo "=== Deploy complete ==="

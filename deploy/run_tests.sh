#!/usr/bin/env bash
# Remote test runner - runs on GPU machine
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH="${PROJECT_DIR}/python:${PYTHONPATH}"

echo "=== Running unit tests ==="
python3 -m pytest tests/ -v --tb=short

echo "=== Running quick benchmark ==="
python3 benchmarks/benchmark.py --quick

echo "=== Tests and benchmarks complete ==="

#!/usr/bin/env bash
# Remote build script - runs on GPU machine
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Set CUDA path
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Check CUDA installation."
    exit 1
fi

echo "CUDA version: $(nvcc --version | grep release)"
echo "Python version: $(python3 --version)"

# Create build directory
mkdir -p build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native

# Build
make -j$(nproc) VERBOSE=1

echo "Build complete."

#!/bin/bash
# Nsight Compute profiling wrapper
# Usage: bash benchmarks/profile.sh [shape] [dtype] [kernel_version]
# Example: bash benchmarks/profile.sh "256,4096" fp32 0

set -euo pipefail

SHAPE=${1:-"256,4096"}
DTYPE=${2:-"fp32"}
VERSION=${3:-"0"}

IFS=',' read -r N D <<< "$SHAPE"

# Map dtype to Python dtype string
case $DTYPE in
    fp32)  PY_DTYPE="torch.float32" ;;
    fp16)  PY_DTYPE="torch.float16" ;;
    bf16)  PY_DTYPE="torch.bfloat16" ;;
    *)     echo "Unknown dtype: $DTYPE"; exit 1 ;;
esac

REPORT_DIR="ncu-reports"
mkdir -p "$REPORT_DIR"

echo "Profiling RMSNorm kernel v${VERSION} with shape (${N}, ${D}), dtype=${DTYPE}"
echo "Report: ${REPORT_DIR}/rmsnorm_v${VERSION}_${N}x${D}_${DTYPE}.ncu-rep"

# Key Nsight Compute metrics
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
sm__inst_executed.sum,\
smsp__average_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
smsp__cycles_elapsed.avg,\
gpu__compute_memory_latency.avg"

ncu --set full \
    --metrics "$METRICS" \
    --export "$REPORT_DIR/rmsnorm_v${VERSION}_${N}x${D}_${DTYPE}" \
    python3 benchmarks/benchmark.py --quick 2>&1 | tee "$REPORT_DIR/rmsnorm_v${VERSION}_${N}x${D}_${DTYPE}.log"

echo "Profile complete. Report saved to ${REPORT_DIR}/"

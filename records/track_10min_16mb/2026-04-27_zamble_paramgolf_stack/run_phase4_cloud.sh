#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Cloud Runner: execution gate only.
# Run from the repo root on a real CUDA/H100 cloud machine:
#   bash records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/run_phase4_cloud.sh
#
# This script does not edit train_gpt.py and does not run ablations.
# It fails loudly if preflight checks fail or required metric categories
# are absent from the produced log. Exact metric values must be read from
# train_h100_debug_seed0.log; never fabricate them.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CANDIDATE_DIR="$REPO_ROOT/records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack"
LOG_FILE="$CANDIDATE_DIR/train_h100_debug_seed0.log"

cd "$REPO_ROOT"

echo "[phase4] repo root: $REPO_ROOT"

if [[ ! -d "$CANDIDATE_DIR" ]]; then
  echo "[phase4][fatal] Missing candidate directory: $CANDIDATE_DIR" >&2
  exit 9
fi

if [[ ! -f "$REPO_ROOT/data/cached_challenge_fineweb.py" ]]; then
  echo "[phase4][fatal] Missing required file: data/cached_challenge_fineweb.py" >&2
  exit 10
fi

if [[ ! -f "$CANDIDATE_DIR/train_gpt.py" ]]; then
  echo "[phase4][fatal] Missing candidate train_gpt.py: $CANDIDATE_DIR/train_gpt.py" >&2
  exit 10
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[phase4][fatal] torchrun not found in PATH." >&2
  exit 11
fi

python3 - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"[phase4][fatal] PyTorch import failed: {exc}", file=sys.stderr)
    raise SystemExit(12)

if not torch.cuda.is_available():
    print("[phase4][fatal] CUDA is not visible to PyTorch.", file=sys.stderr)
    raise SystemExit(13)

print(f"[phase4] torch={torch.__version__}")
print(f"[phase4] cuda_available={torch.cuda.is_available()}")
print(f"[phase4] cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.device_count() > 0:
    print(f"[phase4] cuda_device_0={torch.cuda.get_device_name(0)}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[phase4] nvidia-smi -L"
  nvidia-smi -L
else
  echo "[phase4][warn] nvidia-smi not found; relying on torch CUDA detection."
fi

echo "[phase4] Building cached challenge dataset variant sp8192"
python3 data/cached_challenge_fineweb.py --variant sp8192

echo "[phase4] Running training command"
cd "$CANDIDATE_DIR"
RUN_ID=h100_debug_seed0 \
SEED=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$LOG_FILE"

echo "[phase4] Validating required metrics in $LOG_FILE"

require_log_pattern() {
  local label="$1"
  local pattern="$2"
  if ! grep -Eiq "$pattern" "$LOG_FILE"; then
    echo "[phase4][fatal] Missing required metric in log: $label" >&2
    echo "[phase4][fatal] Expected pattern: $pattern" >&2
    exit 20
  fi
}

require_log_pattern "BPB" "\\b(bpb|val_bpb)\\b"
require_log_pattern "artifact size" "\\b(artifact|size|bytes)\\b"
require_log_pattern "wallclock" "\\b(wallclock|seconds|sec|elapsed|time)\\b"

echo "[phase4] Metric presence gate passed."
echo "[phase4] IMPORTANT: read exact values from log; do not fabricate metrics."

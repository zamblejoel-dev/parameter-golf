#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Cloud Runner: execution gate only.
# Run from the repo root on a real CUDA/H100 cloud machine:
#   bash records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/run_phase4_cloud.sh
#
# Default is 1 GPU for Phase 4 debug. If the cloud allocation must be used as
# a full 8xH100 run, call with:
#   PHASE4_NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/run_phase4_cloud.sh
#
# This script does not edit train_gpt.py and does not run ablations.
# It fails loudly if preflight checks fail or required metric categories
# are absent from the produced log. Exact metric values must be read from
# train_h100_debug_seed0.log; never fabricate them.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CANDIDATE_DIR="$REPO_ROOT/records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack"
LOG_FILE="$CANDIDATE_DIR/train_h100_debug_seed0.log"
NPROC_PER_NODE="${PHASE4_NPROC_PER_NODE:-1}"
ALLOW_NON_H100="${PHASE4_ALLOW_NON_H100:-0}"
DRY_RUN="${PHASE4_DRY_RUN:-0}"

cd "$REPO_ROOT"

echo "[phase4] repo root: $REPO_ROOT"
echo "[phase4] nproc_per_node: $NPROC_PER_NODE"
echo "[phase4] allow_non_h100: $ALLOW_NON_H100"
echo "[phase4] dry_run: $DRY_RUN"

if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "[phase4][fatal] PHASE4_NPROC_PER_NODE must be a positive integer; got '$NPROC_PER_NODE'." >&2
  exit 8
fi

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

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "" && "$DRY_RUN" != "false" && "$DRY_RUN" != "False" ]]; then
  echo "[phase4] DRY RUN: skipping CUDA checks and execution."
  echo
  echo "[phase4] Would build cached challenge dataset:"
  echo "  python3 data/cached_challenge_fineweb.py --variant sp8192"
  echo
  echo "[phase4] Would run training command (from $CANDIDATE_DIR):"
  echo "  RUN_ID=h100_debug_seed0 SEED=0 MAX_WALLCLOCK_SECONDS=600 torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py | tee $LOG_FILE"
  exit 0
fi

PHASE4_NPROC_PER_NODE="$NPROC_PER_NODE" PHASE4_ALLOW_NON_H100="$ALLOW_NON_H100" python3 - <<'PY'
import os
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
    name0 = torch.cuda.get_device_name(0)
    print(f"[phase4] cuda_device_0={name0}")
    allow_non_h100 = os.environ.get("PHASE4_ALLOW_NON_H100", "0") not in {"0", "", "false", "False"}
    if (not allow_non_h100) and ("H100" not in name0.upper()):
        print(
            f"[phase4][fatal] GPU 0 is not an H100 (reported: {name0!r}). "
            "Set PHASE4_ALLOW_NON_H100=1 only for non-evidence debugging.",
            file=sys.stderr,
        )
        raise SystemExit(15)

nproc = int(os.environ["PHASE4_NPROC_PER_NODE"])
if torch.cuda.device_count() < nproc:
    print(
        f"[phase4][fatal] Requested nproc_per_node={nproc}, "
        f"but only {torch.cuda.device_count()} CUDA device(s) are visible.",
        file=sys.stderr,
    )
    raise SystemExit(14)
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
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py | tee "$LOG_FILE"

echo "[phase4] Validating required metrics in $LOG_FILE"

require_log_pattern() {
  local label="$1"
  local pattern="$2"
  if ! grep -Eiq "$pattern" "$LOG_FILE"; then
    echo "[phase4][fatal] Missing required metric in log: $label" >&2
    echo "[phase4][fatal] Expected pattern: $pattern" >&2
    echo "[phase4][fatal] Log tail (last 200 lines):" >&2
    tail -n 200 "$LOG_FILE" >&2 || true
    exit 20
  fi
}

require_log_pattern "val_bpb" "\\bval_bpb\\b"
require_log_pattern "total submission bytes" "Total submission size[^:]*: [0-9]+ bytes"
require_log_pattern "wallclock cap" "stopping_early: wallclock_cap"

echo "[phase4] Metric presence gate passed."
echo "[phase4] IMPORTANT: read exact values from log; do not fabricate metrics."

echo
echo "[phase4] Key log lines (for manual reporting)"
grep -Ei "stopping_early: wallclock_cap|Total submission size|\\bval_bpb\\b" "$LOG_FILE" | tail -n 50 || true

echo
echo "[phase4] Parsed metrics (best-effort)"
LOG_FILE="$LOG_FILE" python3 - <<'PY' || true
import os
import re
from pathlib import Path

log_path = Path(os.environ["LOG_FILE"])
text = log_path.read_text(errors="replace")

def last_match(pattern: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None

total_bytes = last_match(r"Total submission size[^:]*:\s*([0-9]+)\s*bytes")
wallclock_ms = last_match(r"stopping_early:\s*wallclock_cap.*?train_time:\s*([0-9]+)ms")
val_bpb_ttt = last_match(r"quantized_ttt .*?val_bpb:\s*([0-9.]+)")
val_bpb_any = last_match(r"\bval_bpb:\s*([0-9.]+)")

if total_bytes is not None:
    total = int(total_bytes)
    print(f"[phase4] total_submission_bytes={total}")
    if total > 16_000_000:
        print(f"[phase4][fatal] artifact_bytes_over_cap: {total - 16_000_000}", flush=True)
        raise SystemExit(21)
else:
    print("[phase4] total_submission_bytes=unknown")
if wallclock_ms is not None:
    print(f"[phase4] wallclock_s={int(wallclock_ms) / 1000:.3f}")
else:
    print("[phase4] wallclock_s=unknown")
print(f"[phase4] val_bpb_quantized_ttt={val_bpb_ttt or 'unknown'}")
print(f"[phase4] val_bpb_last_seen={val_bpb_any or 'unknown'}")
PY

#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="records/track_10min_16mb/2026-04-29_zamble_final_window_probe"
LOG_PATH="${RECORD_DIR}/train_h100_debug_seed0.log"
mkdir -p "${RECORD_DIR}"

{
  echo "[zamble-h100-debug] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[zamble-h100-debug] host=$(hostname || true)"
  echo "[zamble-h100-debug] pwd=$(pwd)"
  echo "[zamble-h100-debug] git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "[zamble-h100-debug] nvidia_smi_begin"
  nvidia-smi || true
  echo "[zamble-h100-debug] nvidia_smi_end"
  python3 - <<'PY'
import torch
print(f"[zamble-h100-debug] torch={torch.__version__}")
print(f"[zamble-h100-debug] cuda_available={torch.cuda.is_available()}")
print(f"[zamble-h100-debug] cuda_version={torch.version.cuda}")
print(f"[zamble-h100-debug] cuda_device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"[zamble-h100-debug] cuda_device_{i}={torch.cuda.get_device_name(i)}")
PY
  echo "[zamble-h100-debug] download_dataset_if_needed_begin"
  python3 data/cached_challenge_fineweb.py --variant sp1024
  echo "[zamble-h100-debug] download_dataset_if_needed_end"
  echo "[zamble-h100-debug] train_begin"
  RUN_ID="${RUN_ID:-zamble_final_window_h100_debug_seed0}" \
  SEED="${SEED:-0}" \
  DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}" \
  TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}" \
  VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train_gpt.py
  echo "[zamble-h100-debug] train_end"
  echo "[zamble-h100-debug] finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG_PATH}"

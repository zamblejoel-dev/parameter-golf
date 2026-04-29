#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_DIR="records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
STATUS_DIR="records/track_10min_16mb/2026-04-29_zamble_final_window_probe"
RUN_STATUS="${STATUS_DIR}/prior_candidate_run_status.txt"
SEEDS_CSV="${SEEDS:-42,314,999}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "${STATUS_DIR}"

if [[ ! -d "${CANDIDATE_DIR}" ]]; then
  echo "prior_candidate_missing: run stage_prior_candidate.sh first" | tee "${RUN_STATUS}"
  exit 1
fi

required=(
  "${CANDIDATE_DIR}/README.md"
  "${CANDIDATE_DIR}/submission.json"
  "${CANDIDATE_DIR}/train_gpt.py"
)

for path in "${required[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "prior_candidate_invalid: missing_or_empty ${path}" | tee "${RUN_STATUS}"
    exit 1
  fi
done

{
  echo "[prior-candidate-h100] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[prior-candidate-h100] host=$(hostname || true)"
  echo "[prior-candidate-h100] pwd=$(pwd)"
  echo "[prior-candidate-h100] git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "[prior-candidate-h100] candidate_dir=${CANDIDATE_DIR}"
  echo "[prior-candidate-h100] seeds=${SEEDS_CSV}"
  echo "[prior-candidate-h100] nproc_per_node=${NPROC_PER_NODE}"
  echo "[prior-candidate-h100] nvidia_smi_begin"
  nvidia-smi || true
  echo "[prior-candidate-h100] nvidia_smi_end"
  python3 - <<'PY'
import torch
print(f"[prior-candidate-h100] torch={torch.__version__}")
print(f"[prior-candidate-h100] cuda_available={torch.cuda.is_available()}")
print(f"[prior-candidate-h100] cuda_version={torch.version.cuda}")
print(f"[prior-candidate-h100] cuda_device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"[prior-candidate-h100] cuda_device_{i}={torch.cuda.get_device_name(i)}")
PY
  echo "[prior-candidate-h100] deps_check_begin"
  python3 - <<'PY'
mods = ["brotli", "sentencepiece"]
for name in mods:
    __import__(name)
    print(f"[prior-candidate-h100] import_ok={name}")
try:
    from flash_attn_interface import flash_attn_func  # noqa: F401
    print("[prior-candidate-h100] import_ok=flash_attn_interface")
except Exception as exc:
    print(f"[prior-candidate-h100] import_warn=flash_attn_interface {type(exc).__name__}: {exc}")
PY
  echo "[prior-candidate-h100] deps_check_end"
  echo "[prior-candidate-h100] dataset_begin"
  MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}" python3 data/cached_challenge_fineweb.py --variant sp8192
  echo "[prior-candidate-h100] dataset_end"
} 2>&1 | tee "${STATUS_DIR}/prior_candidate_h100_preamble.log"

IFS=',' read -ra seed_list <<< "${SEEDS_CSV}"
for seed in "${seed_list[@]}"; do
  seed="$(echo "${seed}" | tr -d ' ')"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  log_path="${CANDIDATE_DIR}/fresh_h100_seed${seed}.log"
  {
    echo "[prior-candidate-h100] seed=${seed} run_begin=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    SEED="${seed}" \
    QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}" \
    TTT_ENABLED="${TTT_ENABLED:-1}" \
    TTT_LR="${TTT_LR:-0.005}" \
    TTT_EPOCHS="${TTT_EPOCHS:-3}" \
    MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${CANDIDATE_DIR}/train_gpt.py"
    echo "[prior-candidate-h100] seed=${seed} run_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "${log_path}"
done

echo "prior_candidate_h100_runs_finished" | tee "${RUN_STATUS}"
python3 "${STATUS_DIR}/parse_prior_candidate.py"

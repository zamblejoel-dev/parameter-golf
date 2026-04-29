#!/usr/bin/env bash
set -euo pipefail

SOURCE_REF="${SOURCE_REF:-origin/codex/zamble-paramgolf-attempt}"
CANDIDATE_DIR="records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
STATUS_DIR="records/track_10min_16mb/2026-04-29_zamble_final_window_probe"
STAGE_STATUS="${STATUS_DIR}/prior_candidate_stage_status.txt"

mkdir -p "${STATUS_DIR}"

{
  echo "[stage-prior-candidate] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[stage-prior-candidate] source_ref=${SOURCE_REF}"
  echo "[stage-prior-candidate] candidate_dir=${CANDIDATE_DIR}"
  echo "[stage-prior-candidate] current_commit=$(git rev-parse HEAD)"

  echo "[stage-prior-candidate] fetch_source_begin"
  git fetch origin codex/zamble-paramgolf-attempt
  echo "[stage-prior-candidate] fetch_source_end"

  echo "[stage-prior-candidate] checkout_candidate_begin"
  git checkout "${SOURCE_REF}" -- "${CANDIDATE_DIR}"
  echo "[stage-prior-candidate] checkout_candidate_end"

  required=(
    "${CANDIDATE_DIR}/README.md"
    "${CANDIDATE_DIR}/submission.json"
    "${CANDIDATE_DIR}/train_gpt.py"
    "${CANDIDATE_DIR}/train_seed42.log"
    "${CANDIDATE_DIR}/train_seed314.log"
    "${CANDIDATE_DIR}/train_seed999.log"
  )

  missing=0
  empty=0
  for path in "${required[@]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "[stage-prior-candidate][missing] ${path}"
      missing=$((missing + 1))
      continue
    fi
    bytes=$(wc -c < "${path}" | tr -d ' ')
    echo "[stage-prior-candidate][file] ${path} bytes=${bytes}"
    if [[ "${bytes}" -le 0 ]]; then
      echo "[stage-prior-candidate][empty] ${path}"
      empty=$((empty + 1))
    fi
  done

  echo "[stage-prior-candidate] python_compile_begin"
  if python3 -m py_compile "${CANDIDATE_DIR}/train_gpt.py"; then
    compile_status="passed"
  else
    compile_status="failed"
  fi
  echo "[stage-prior-candidate] python_compile=${compile_status}"

  if [[ "${missing}" -eq 0 && "${empty}" -eq 0 && "${compile_status}" == "passed" ]]; then
    echo "[stage-prior-candidate] status=prior_candidate_staged"
    echo "prior_candidate_staged" > "${STAGE_STATUS}"
  else
    echo "[stage-prior-candidate] status=prior_candidate_stage_failed missing=${missing} empty=${empty} compile=${compile_status}"
    echo "prior_candidate_stage_failed missing=${missing} empty=${empty} compile=${compile_status}" > "${STAGE_STATUS}"
    exit 1
  fi

  echo "[stage-prior-candidate] next_run_command_begin"
  cat <<'CMD'
cd records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 ../../../data/cached_challenge_fineweb.py --variant sp8192
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 torchrun --standalone --nproc_per_node=8 train_gpt.py
CMD
  echo "[stage-prior-candidate] next_run_command_end"
  echo "[stage-prior-candidate] finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "${STATUS_DIR}/prior_candidate_stage.log"

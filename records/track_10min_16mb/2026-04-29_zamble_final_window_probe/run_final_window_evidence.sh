#!/usr/bin/env bash
set -euo pipefail

STATUS_DIR="records/track_10min_16mb/2026-04-29_zamble_final_window_probe"
CANDIDATE_DIR="records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
FINAL_STATUS="${STATUS_DIR}/final_window_evidence_status.txt"
FINAL_LOG="${STATUS_DIR}/final_window_evidence_orchestrator.log"

mkdir -p "${STATUS_DIR}"

{
  echo "[final-window-evidence] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[final-window-evidence] host=$(hostname || true)"
  echo "[final-window-evidence] pwd=$(pwd)"
  echo "[final-window-evidence] git_commit=$(git rev-parse HEAD 2>/dev/null || true)"

  echo "[final-window-evidence] stage_begin"
  bash "${STATUS_DIR}/stage_prior_candidate.sh"
  echo "[final-window-evidence] stage_end"

  echo "[final-window-evidence] h100_run_begin"
  bash "${STATUS_DIR}/run_prior_candidate_h100.sh"
  echo "[final-window-evidence] h100_run_end"

  echo "[final-window-evidence] parse_begin"
  python3 "${STATUS_DIR}/parse_prior_candidate.py"
  echo "[final-window-evidence] parse_end"

  if [[ ! -s "${STATUS_DIR}/prior_candidate_h100_status.json" ]]; then
    echo "[final-window-evidence] status=final_window_evidence_failed missing prior_candidate_h100_status.json"
    echo "final_window_evidence_failed missing_status_json" > "${FINAL_STATUS}"
    exit 1
  fi

  python3 - <<'PY'
import json
from pathlib import Path
status_path = Path("records/track_10min_16mb/2026-04-29_zamble_final_window_probe/prior_candidate_h100_status.json")
status = json.loads(status_path.read_text())
print("[final-window-evidence] parsed_status=" + str(status.get("status")))
print("[final-window-evidence] mean_val_bpb=" + str(status.get("mean_val_bpb")))
print("[final-window-evidence] run_count=" + str(status.get("run_count")))
if status.get("status") == "prior_candidate_h100_passed":
    Path("records/track_10min_16mb/2026-04-29_zamble_final_window_probe/final_window_evidence_status.txt").write_text("final_window_evidence_passed\n")
else:
    Path("records/track_10min_16mb/2026-04-29_zamble_final_window_probe/final_window_evidence_status.txt").write_text("final_window_evidence_not_passed " + str(status.get("status")) + "\n")
PY

  echo "[final-window-evidence] files_to_commit_begin"
  cat <<EOF
${STATUS_DIR}/prior_candidate_stage.log
${STATUS_DIR}/prior_candidate_stage_status.txt
${STATUS_DIR}/prior_candidate_h100_preamble.log
${STATUS_DIR}/prior_candidate_run_status.txt
${STATUS_DIR}/prior_candidate_h100_status.json
${STATUS_DIR}/final_window_evidence_status.txt
${STATUS_DIR}/final_window_evidence_orchestrator.log
${CANDIDATE_DIR}/fresh_h100_seed42.log
${CANDIDATE_DIR}/fresh_h100_seed314.log
${CANDIDATE_DIR}/fresh_h100_seed999.log
EOF
  echo "[final-window-evidence] files_to_commit_end"

  echo "[final-window-evidence] suggested_git_commands_begin"
  cat <<'EOF'
git checkout -b evidence/final-window-h100

git add \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/prior_candidate_stage.log \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/prior_candidate_stage_status.txt \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/prior_candidate_h100_preamble.log \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/prior_candidate_run_status.txt \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/prior_candidate_h100_status.json \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/final_window_evidence_status.txt \
  records/track_10min_16mb/2026-04-29_zamble_final_window_probe/final_window_evidence_orchestrator.log \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/fresh_h100_seed42.log \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/fresh_h100_seed314.log \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/fresh_h100_seed999.log

git commit -m "Add final-window H100 evidence logs"
git push origin evidence/final-window-h100
EOF
  echo "[final-window-evidence] suggested_git_commands_end"
  echo "[final-window-evidence] finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${FINAL_LOG}"

#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

STATUS_DIR = Path(__file__).resolve().parent
REPO_ROOT = STATUS_DIR.parent.parent.parent
CANDIDATE_DIR = REPO_ROOT / "track_10min_16mb" / "2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
OUT_PATH = STATUS_DIR / "prior_candidate_h100_status.json"

SEED_RE = re.compile(r"(?:^|\b)seed[:=\s]+([0-9]+)\b", re.IGNORECASE | re.MULTILINE)
BPB_RE = re.compile(r"quantized_ttt\s+val_loss:[0-9.]+\s+val_bpb:([0-9.]+)", re.IGNORECASE)
FALLBACK_BPB_RE = re.compile(r"val_bpb[:\s]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
ARTIFACT_RE = re.compile(r"Total submission size quantized\+brotli:\s*([0-9][0-9_,]*)", re.IGNORECASE)
FALLBACK_ARTIFACT_RE = re.compile(r"(?:artifact|submission).*?(?:bytes|size)[^0-9]*([0-9][0-9_,]*)", re.IGNORECASE)
TRAIN_MS_RE = re.compile(r"stopping_early:\s*wallclock_cap\s*train_time:\s*([0-9]+)ms", re.IGNORECASE)
EVAL_MS_RE = re.compile(r"quantized_ttt\s+val_loss:[0-9.]+\s+val_bpb:[0-9.]+\s+eval_time:([0-9]+)ms", re.IGNORECASE)

GPU_MARKERS = ("H100", "CUDA", "cuda_available=True", "NVIDIA")
ERROR_MARKERS = ("Traceback", "RuntimeError", "CUDA out of memory", "failed", "error")


def parse_int(text: str) -> int | None:
    return int(text.replace(",", "").replace("_", "")) if text else None


def first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return parse_int(match.group(1)) if match else None


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    seed = first_int(SEED_RE, text)
    val_bpb = first_float(BPB_RE, text)
    if val_bpb is None:
        val_bpb = first_float(FALLBACK_BPB_RE, text)
    artifact_bytes = first_int(ARTIFACT_RE, text)
    if artifact_bytes is None:
        artifact_bytes = first_int(FALLBACK_ARTIFACT_RE, text)
    train_ms = first_int(TRAIN_MS_RE, text)
    eval_ms = first_int(EVAL_MS_RE, text)
    has_error = any(marker.lower() in text.lower() for marker in ERROR_MARKERS)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "seed": seed,
        "val_bpb": val_bpb,
        "artifact_bytes": artifact_bytes,
        "train_ms": train_ms,
        "eval_ms": eval_ms,
        "train_under_600s": train_ms is not None and train_ms <= 600_000,
        "eval_under_600s": eval_ms is not None and eval_ms <= 600_000,
        "artifact_under_16000000": artifact_bytes is not None and artifact_bytes < 16_000_000,
        "has_error": has_error,
    }


def main() -> int:
    preamble_path = STATUS_DIR / "prior_candidate_h100_preamble.log"
    preamble = preamble_path.read_text(errors="replace") if preamble_path.exists() else ""
    has_gpu_evidence = any(marker in preamble for marker in GPU_MARKERS)
    has_h100_evidence = "H100" in preamble
    has_cuda_evidence = "cuda_available=True" in preamble or "CUDA" in preamble

    logs = sorted(CANDIDATE_DIR.glob("fresh_h100_seed*.log")) if CANDIDATE_DIR.exists() else []
    runs = [parse_log(path) for path in logs]
    bpbs = [run["val_bpb"] for run in runs if isinstance(run.get("val_bpb"), float)]

    all_have_scores = bool(runs) and all(run.get("val_bpb") is not None for run in runs)
    all_artifacts_ok = bool(runs) and all(run.get("artifact_under_16000000") is True for run in runs)
    all_train_ok = bool(runs) and all(run.get("train_under_600s") is True for run in runs)
    all_eval_ok = bool(runs) and all(run.get("eval_under_600s") is True for run in runs)
    any_error = any(run.get("has_error") is True for run in runs)

    if not runs:
        status = "WAITING-FOR-H100-EVIDENCE"
    elif any_error or not has_gpu_evidence:
        status = "prior_candidate_h100_failed_runtime"
    elif not all_have_scores:
        status = "prior_candidate_h100_failed_score"
    elif not all_artifacts_ok:
        status = "prior_candidate_h100_failed_artifact"
    elif not all_train_ok or not all_eval_ok:
        status = "prior_candidate_h100_failed_timeout"
    else:
        status = "prior_candidate_h100_passed"

    result = {
        "status": status,
        "candidate_dir": str(CANDIDATE_DIR.relative_to(REPO_ROOT)),
        "run_count": len(runs),
        "mean_val_bpb": statistics.mean(bpbs) if bpbs else None,
        "std_val_bpb": statistics.pstdev(bpbs) if len(bpbs) > 1 else 0.0 if len(bpbs) == 1 else None,
        "has_gpu_evidence": has_gpu_evidence,
        "has_h100_evidence": has_h100_evidence,
        "has_cuda_evidence": has_cuda_evidence,
        "runs": runs,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

RECORD_DIR = Path(__file__).resolve().parent
LOG_PATH = RECORD_DIR / "train_h100_debug_seed0.log"
STATUS_PATH = RECORD_DIR / "h100_debug_status.json"

BPB_RE = re.compile(r"val_bpb[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
ARTIFACT_RE = re.compile(r"(?:artifact|compressed model|model).*?(?:bytes|size)[^0-9]*([0-9][0-9_,]*)", re.IGNORECASE)
SECONDS_RE = re.compile(r"(?:wallclock|elapsed|total).*?(?:seconds|sec|s)[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
SEED_RE = re.compile(r"\bSEED[=:\s]+([0-9]+)\b")

GPU_MARKERS = ("H100", "CUDA", "cuda_available=True", "NVIDIA")
ERROR_MARKERS = ("Traceback", "RuntimeError", "CUDA out of memory", "failed", "error")


def first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", "").replace("_", ""))


def first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", "").replace("_", ""))


def classify(text: str) -> dict[str, object]:
    val_bpb = first_float(BPB_RE, text)
    artifact_bytes = first_int(ARTIFACT_RE, text)
    wallclock_seconds = first_float(SECONDS_RE, text)
    seed = first_int(SEED_RE, text)
    has_gpu_evidence = any(marker in text for marker in GPU_MARKERS)
    has_h100_evidence = "H100" in text
    has_cuda_evidence = "cuda_available=True" in text or "CUDA" in text
    has_error = any(marker.lower() in text.lower() for marker in ERROR_MARKERS)

    under_artifact_cap = artifact_bytes is not None and artifact_bytes < 16_000_000
    inside_runtime_limit = wallclock_seconds is not None and wallclock_seconds <= 600.0

    if val_bpb is None or artifact_bytes is None:
        status = "phase4_h100_debug_failed_runtime" if has_error else "WAITING-FOR-H100-EVIDENCE"
    elif not under_artifact_cap:
        status = "phase4_h100_debug_failed_artifact"
    elif wallclock_seconds is not None and not inside_runtime_limit:
        status = "phase4_h100_debug_failed_timeout"
    elif not has_gpu_evidence:
        status = "phase4_h100_debug_failed_runtime"
    else:
        status = "phase4_h100_debug_passed"

    return {
        "status": status,
        "log_path": str(LOG_PATH.relative_to(RECORD_DIR.parent.parent.parent)),
        "val_bpb": val_bpb,
        "artifact_bytes": artifact_bytes,
        "wallclock_seconds": wallclock_seconds,
        "seed": seed,
        "has_gpu_evidence": has_gpu_evidence,
        "has_h100_evidence": has_h100_evidence,
        "has_cuda_evidence": has_cuda_evidence,
        "under_16000000_bytes": under_artifact_cap,
        "inside_600_seconds": inside_runtime_limit,
    }


def main() -> int:
    if not LOG_PATH.exists():
        result = {
            "status": "WAITING-FOR-H100-EVIDENCE",
            "log_path": str(LOG_PATH.relative_to(RECORD_DIR.parent.parent.parent)),
            "reason": "missing train_h100_debug_seed0.log",
        }
    else:
        result = classify(LOG_PATH.read_text(errors="replace"))

    STATUS_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

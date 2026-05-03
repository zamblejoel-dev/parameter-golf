#!/usr/bin/env python3
"""
Report the best (lowest) val_bpb submission per track.

Usage:
  python3 records/_tools/report_best_record.py
  python3 records/_tools/report_best_record.py --track 10min_16mb
  python3 records/_tools/report_best_record.py --json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class BestRecord:
    track: str
    submission_json: str
    name: str
    date: str
    val_bpb_mean: float
    val_bpb_std: Optional[float]
    seeds: list[int]
    artifact_bytes_mean: Optional[int]
    artifact_bytes_max: Optional[int]
    train_wallclock_s_mean: Optional[float]
    train_wallclock_s_max: Optional[float]
    hardware: Optional[str]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _parse_wallclock_s(log_path: Path) -> Optional[float]:
    text = log_path.read_text(errors="replace")
    matches = re.findall(
        r"stopping_early:\s*wallclock_cap.*?train_time:\s*([0-9]+)ms", text
    )
    if matches:
        return int(matches[-1]) / 1000.0
    return None


def _summarize_submission(submission_path: Path, submission: dict[str, Any]) -> BestRecord:
    track = str(submission.get("track") or "unknown")
    name = str(submission.get("name") or submission_path.parent.name)
    date = str(submission.get("date") or "")
    seeds = submission.get("seeds") or []
    seeds = [int(s) for s in seeds if _safe_int(s) is not None]

    val_bpb = float(submission["val_bpb"])
    val_bpb_std = _safe_float(submission.get("val_bpb_std"))
    hardware = submission.get("hardware")
    if hardware is not None:
        hardware = str(hardware)

    seed_results = submission.get("seed_results") or {}
    artifact_bytes: list[int] = []
    for seed in seeds:
        seed_dict = seed_results.get(str(seed)) or {}
        b = _safe_int(seed_dict.get("artifact_bytes"))
        if b is not None:
            artifact_bytes.append(b)

    train_logs = sorted(submission_path.parent.glob("train_seed*.log"))
    wallclocks = [_parse_wallclock_s(p) for p in train_logs]
    wallclocks = [w for w in wallclocks if w is not None]

    return BestRecord(
        track=track,
        submission_json=str(submission_path),
        name=name,
        date=date,
        val_bpb_mean=val_bpb,
        val_bpb_std=val_bpb_std,
        seeds=seeds,
        artifact_bytes_mean=round(statistics.mean(artifact_bytes)) if artifact_bytes else None,
        artifact_bytes_max=max(artifact_bytes) if artifact_bytes else None,
        train_wallclock_s_mean=round(statistics.mean(wallclocks), 3) if wallclocks else None,
        train_wallclock_s_max=round(max(wallclocks), 3) if wallclocks else None,
        hardware=hardware,
    )


def _iter_submissions(records_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in glob.glob(str(records_root / "**" / "submission.json"), recursive=True):
        path = Path(p)
        try:
            submission = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(submission, dict):
            continue
        if _safe_float(submission.get("val_bpb")) is None:
            continue
        out.append((path, submission))
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="", help="Optional track filter (e.g. 10min_16mb).")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    records_root = repo_root / "records"
    submissions = _iter_submissions(records_root)
    if args.track:
        submissions = [(p, s) for (p, s) in submissions if str(s.get("track")) == args.track]

    best_by_track: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, submission in submissions:
        track = str(submission.get("track") or "unknown")
        val = float(submission["val_bpb"])
        cur = best_by_track.get(track)
        if cur is None or val < float(cur[1]["val_bpb"]):
            best_by_track[track] = (path, submission)

    if not best_by_track:
        print("No submissions found.")
        return 2

    best_records = [
        _summarize_submission(path, submission) for (path, submission) in best_by_track.values()
    ]
    best_records.sort(key=lambda r: (r.track, r.val_bpb_mean))

    if args.json:
        print(json.dumps([asdict(r) for r in best_records], indent=2, sort_keys=True))
        return 0

    for r in best_records:
        print(f"track: {r.track}")
        print(f"best:  {r.val_bpb_mean:.5f} bpb  ({r.name})")
        if r.val_bpb_std is not None:
            print(f"std:   {r.val_bpb_std:.5f}")
        if r.artifact_bytes_mean is not None:
            print(f"bytes: mean={r.artifact_bytes_mean}  max={r.artifact_bytes_max}")
        if r.train_wallclock_s_mean is not None:
            print(f"time:  mean={r.train_wallclock_s_mean:.3f}s  max={r.train_wallclock_s_max:.3f}s")
        if r.hardware:
            print(f"hw:    {r.hardware}")
        print(f"src:   {r.submission_json}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


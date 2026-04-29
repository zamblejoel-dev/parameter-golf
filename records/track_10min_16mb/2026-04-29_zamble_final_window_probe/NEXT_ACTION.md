# Next Action: H100 Evidence Pass

Status: **WAITING-FOR-H100-EVIDENCE**

GitHub issues are disabled for this repository, so this file is the tracked handoff for the required physical H100 run.

## Run from the H100 machine

```bash
cd /workspace/parameter-golf
git pull origin main

bash records/track_10min_16mb/2026-04-29_zamble_final_window_probe/run_h100_debug.sh
python3 records/track_10min_16mb/2026-04-29_zamble_final_window_probe/parse_h100_debug.py
```

## Required evidence files after the run

Commit only after the run completes:

```text
records/track_10min_16mb/2026-04-29_zamble_final_window_probe/train_h100_debug_seed0.log
records/track_10min_16mb/2026-04-29_zamble_final_window_probe/h100_debug_status.json
```

## Acceptance criteria

The log must contain:

- H100/CUDA evidence from `nvidia-smi` and PyTorch CUDA checks
- final `val_bpb`
- compressed artifact bytes
- wallclock/runtime evidence
- seed

The parser must emit exactly one status:

- `phase4_h100_debug_passed`
- `phase4_h100_debug_failed_runtime`
- `phase4_h100_debug_failed_score`
- `phase4_h100_debug_failed_artifact`
- `phase4_h100_debug_failed_timeout`
- `WAITING-FOR-H100-EVIDENCE`

## Truth rule

Do not claim `phase4_h100_debug_passed` unless the log exists and the parser emits that status from real evidence. No fabricated metrics, no assumed score, no inferred runtime.

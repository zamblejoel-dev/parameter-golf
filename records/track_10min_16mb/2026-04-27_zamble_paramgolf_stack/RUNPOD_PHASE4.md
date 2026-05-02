# RunPod Phase 4 Orchestration

Purpose: use RunPod MCP as the pod-control layer, then run the existing Phase 4
gate inside the cloud pod. This repo does not need a custom pod-runner plugin for
the current flow.

## Local Codex MCP Setup

Run once on the local machine with a real API key:

```bash
codex mcp add runpod \
  --env RUNPOD_API_KEY=YOUR_KEY_HERE \
  -- npx -y @runpod/mcp-server@latest
```

Confirm it is registered:

```bash
codex mcp list
```

Do not commit API keys, local MCP config, or generated credentials.

## Control Flow

```text
Codex CLI
  -> RunPod MCP server
  -> RunPod H100 pod
  -> parameter-golf checkout
  -> run_phase4_cloud.sh
  -> train_h100_debug_seed0.log
```

## Pod Request

Use a 1x H100 SXM pod for Phase 4 debug before any 8xH100 run. The pod should
start from the official Parameter Golf environment or an equivalent image with
CUDA, PyTorch, `torchrun`, and dataset access already working.

Inside the pod, clone or mount this repository, checkout the intended branch,
then run from the repo root:

```bash
bash records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/run_phase4_cloud.sh
```

If you are only debugging on non-H100 CUDA (not valid challenge evidence), you
can bypass the H100 guard:

```bash
PHASE4_ALLOW_NON_H100=1 \
bash records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/run_phase4_cloud.sh
```

Only after the 1xH100 debug run succeeds:

```bash
PHASE4_NPROC_PER_NODE=8 \
bash records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/run_phase4_cloud.sh
```

## Required Evidence

Return `records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/train_h100_debug_seed0.log`
and report only values present in that log:

- GPU model and count
- PyTorch and CUDA visibility
- final `val_bpb`
- `val_loss`, if printed
- total submission bytes
- train wallclock
- eval wallclock, if printed
- warnings or errors
- whether artifact bytes are below 16,000,000

Stop without reporting metrics if CUDA is unavailable, no H100 is visible,
`torchrun` is missing, dataset prep fails, training crashes, or the required log
patterns are absent.

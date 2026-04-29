# Prior Work Inventory

Status: **reference only — do not merge wholesale**

This inventory preserves the useful pre-April-29 context without importing old candidate folders into `main` as active submissions. The current clean gate remains:

```bash
bash records/track_10min_16mb/2026-04-29_zamble_final_window_probe/run_h100_debug.sh
python3 records/track_10min_16mb/2026-04-29_zamble_final_window_probe/parse_h100_debug.py
```

## Reviewed branches

### `codex/zamble-paramgolf-attempt`

Comparison against `main` showed this branch is large and diverged:

- Ahead of `main`: 50 commits
- Behind `main`: 2 commits
- Adds many historical `records/track_10min_16mb/*` folders
- Includes README/submission/log material for multiple candidate records
- Includes some compressed-wrapper `train_gpt.py` files
- The April 27 stack appeared in the compare view as placeholder-like/zero-line files, so it should not be merged blindly

Decision: **do not merge wholesale**.

### `codex/add-leaderboard-comparison-and-innovation-phase`

Comparison against `main` showed this branch is also large and diverged:

- Ahead of `main`: 46 commits
- Behind `main`: 2 commits
- Modifies top-level `README.md`
- Adds many historical candidate folders

Decision: **do not merge wholesale**.

## Strongest prior candidate reference

The strongest prior candidate found in the old branch is:

```text
records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/
```

Its `submission.json` reports:

```json
{
  "val_bpb": 1.08100,
  "val_bpb_std": 0.00020,
  "seeds": [42, 314, 999],
  "hardware": "8xH100 80GB SXM",
  "technique_summary": "SP8192 + 3-Layer Depth Recurrence (L3-5) + Parallel Residuals (L7+) + QK-Gain 5.25 + EMA 0.9965 + WD 0.095 + Score-First TTT (SGD 3ep) + GPTQ SDClip + Brotli"
}
```

Seed 42 log evidence from the old branch included:

```text
world_size: 8
seed: 42
stopping_early: wallclock_cap train_time: 588047ms step: 4550/20000
Serialized model quantized+brotli: 15975300 bytes
Total submission size quantized+brotli: 15991930 bytes
quantized_ttt val_loss:2.79180191 val_bpb:1.08079352 eval_time:366525ms
```

This is useful, but it is still not enough to overwrite the current truth gate. It should be treated as a candidate lineage/reference until revalidated cleanly.

## Integration rule

Only integrate prior work if it passes all of these checks:

1. The candidate folder has non-empty `README.md`, `submission.json`, `train_gpt.py`, and train logs.
2. `train_gpt.py` can compile and run from inside its record folder.
3. The logs contain real H100/CUDA evidence, final BPB, artifact bytes, wallclock, and seed.
4. Artifact bytes are below `16,000,000`.
5. Training runtime is inside the challenge limit.
6. The score claim is reproduced or intentionally downgraded to reference-only.
7. No top-level `README.md` leaderboard edits are imported unless the submission is final and verified.

## Recommended next step

Do not import the whole old branch. Instead, use the April 9 candidate as the candidate-specific H100 target after the baseline debug gate passes:

```bash
git checkout codex/zamble-paramgolf-attempt -- records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
```

Then run a fresh H100 validation through the current gate and commit only real evidence.

# Zamble Final Window Probe

Status: **waiting for H100 evidence**.

This folder is a gated execution package for the final Parameter Golf window. It is **not** a leaderboard claim and it does **not** report a score until a real H100/CUDA run log exists.

## Purpose

The current public SOTA target in this fork is the `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` record at **1.1147 BPB** with artifacts around **15.91 MB**. Any record attempt needs real 8xH100 evidence, artifact bytes below **16,000,000**, wallclock inside the challenge limit, and enough repeated seeds to support the claim.

This package exists to keep the next move clean:

1. Run a strict H100 debug pass.
2. Capture the raw log.
3. Parse only real evidence from the log.
4. Classify the run without inventing metrics.

## Required artifact

The blocker is only cleared when this file exists and contains real CUDA/H100 evidence plus final score/artifact lines:

```text
records/track_10min_16mb/2026-04-29_zamble_final_window_probe/train_h100_debug_seed0.log
```

## Run command

From the repository root on an H100 CUDA machine:

```bash
bash records/track_10min_16mb/2026-04-29_zamble_final_window_probe/run_h100_debug.sh
```

The script writes:

```text
records/track_10min_16mb/2026-04-29_zamble_final_window_probe/train_h100_debug_seed0.log
```

## Classification rules

After the run, classify exactly one:

- `phase4_h100_debug_passed`
- `phase4_h100_debug_failed_runtime`
- `phase4_h100_debug_failed_score`
- `phase4_h100_debug_failed_artifact`
- `phase4_h100_debug_failed_timeout`
- `WAITING-FOR-H100-EVIDENCE`

Do not claim progress if the log is missing. Do not claim a BPB, artifact size, or wallclock unless it appears in the log.

## Evidence to extract

- `val_bpb`
- compressed artifact bytes
- wallclock seconds
- seed
- GPU/CUDA evidence
- artifact `< 16,000,000` bytes
- runtime inside required limit

## Initial debug configuration

The first pass intentionally uses the repo baseline path because it is reproducible and already wired for challenge-style logging:

```bash
RUN_ID=zamble_final_window_h100_debug_seed0 \
SEED=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=${NPROC_PER_NODE:-1} train_gpt.py
```

If this debug pass succeeds, the next phase is candidate-specific: copy the best candidate `train_gpt.py` into a fresh record folder, run 3 seeds on the target 8xH100 setup, and only then write `submission.json`.

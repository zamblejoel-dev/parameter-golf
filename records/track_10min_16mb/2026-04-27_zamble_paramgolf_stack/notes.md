# Zamble ParamGolf Stack Notes

Date: 2026-04-27

Local role: compare, scaffold, compile, tiny smoke, package.
Cloud role: CUDA debug, ablations, BPB measurement, final 8xH100 logs.

True status: local scaffold complete, base chosen, ready for 1xH100 cloud debug.
Not performance-validated. Not leaderboard-ready.

Do not modify `train_gpt.py` and do not begin ablations until the unmodified base
finishes a real 1xH100 run with BPB, artifact bytes, wallclock, and log output.

## Phase 4 Status

Phase 4 remains pending.

Any prior attempted execution outside a real H100/CUDA runner is invalid and must
not be treated as a Phase 4 result. Non-H100/container failures, missing
`torchrun`, incomplete checkouts, blocked network clones, or CPU-only torch
environments do not produce submission artifacts.

No real Phase 4 metrics exist yet:

- val_bpb: none
- val_loss: none
- compressed artifact bytes: none
- train wallclock: none
- eval wallclock: none
- valid `train_h100_debug_seed0.log`: none

Next required action: run the frozen unmodified base on an actual 1xH100 CUDA
machine after the environment gate passes.

RunPod MCP orchestration details live in `RUNPOD_PHASE4.md`.

## Phase 4 Local Attempt Evidence (Non-metric)

Date: 2026-04-29

Status: FAIL

Blocker: missing_valid_phase4_h100_debug (still active)

Local failure classification: blocked_by_no_cuda

Fatal line:

`[phase4][fatal] CUDA is not visible to PyTorch.`

Meaning: this machine/session cannot satisfy Phase 4 debug requirements. This is
not evidence about BPB, artifacts, wallclock, ablations, or architecture.

## Cloud Environment Gate

Before Phase 4, verify the cloud machine:

```bash
nvidia-smi

python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

which torchrun
```

Stop immediately if CUDA is unavailable, no H100/GPU is present, torch fails to
import, `torchrun` is missing, or the parameter-golf checkout is incomplete.

## Cloud 4-Agent Workflow

- Agent 1, Environment Gate: run the checks above and stop if any required
  condition fails.
- Agent 2, Runner: if the gate passes, run only the frozen Phase 4 command.
- Agent 3, Log Analyst: extract completion status, GPU, CUDA/PyTorch,
  `val_bpb`, `val_loss`, compressed artifact bytes, wallclock, warnings/errors,
  and artifact-under-16MB status.
- Agent 4, Notes/Package: update `notes.md` only after real H100 metrics exist
  and keep only valid H100 logs as submission artifacts.

## Top Run Comparison

| Rank | Record | BPB | Artifact | Context / vocab | Depth / width | Optimizer | Quantization | Eval method | Wallclock | Unique trick | Legality |
|---:|---|---:|---:|---|---|---|---|---|---|---|---|
| 1 | 2026-04-09 SP8192 3-layer recurrence + parallel residual + QK5.25 + legal TTT | 1.08100 | 15,991,930-15,993,232 | SP8192, eval seq 1024 sliding stride 64 | 11L physical, 512d, 8H/4KV, MLP 4x, 17 virtual layers via recurrence | MuonEq-R + AdamW scalars/embeds, WD .095, EMA .9965, warmdown .72 | GPTQ int6 matrices, int8 embeddings, SDClip, byte shuffle, Brotli | Legal score-first TTT, 32K chunks, SGD lr .005, 3 epochs | train ~588s, eval ~500s on 8xH100 | 3-layer recurrence at layers 3-5 plus QK gain 5.25 | Strongest legal public base; no SLOT/ETLB/ngram/pre-quant TTT |
| 2 | 2026-04-08 SP8192 parallel residual + score-first TTT | 1.08218 | 15,991,486 | SP8192 | 11L, 512d, 8H/4KV, MLP 4x | MuonEq-R, EMA | GPTQ embeddings, SDClip, Brotli | Legal score-first TTT | 8xH100 | Parallel residual from layer 7 combined with TTT | Legal score-first TTT |
| 3 | 2026-04-06 SP8192 QK5 legal TTT | 1.08279 | 15,989,058-15,992,546 | SP8192 | 11L, 512d, 8H/4KV, MLP 4x, layers 4-5 recurrence | MuonEq-R, EMA | GPTQ int6 matrices, int8 embeddings, SDClip | Legal score-first TTT | train ~588s, eval ~382s | QK_GAIN_INIT 5.0 plus legal TTT | Good compliance writeup; no cache/bias/SLOT |
| 4 | 2026-04-06 Hessian SDClip + progressive recurrence | 1.08354 | ~15,978,121 | SP8192 | 11L, 512d, MLP 4x, progressive recurrence | MuonEq-R lineage | Hessian-aware SDClip lambda .175 | Sliding only | 8xH100 | Per-row Hessian modulation of SDClip | Track A fixed predictor |
| 5 | 2026-04-05 SP8192 GPTQ embeddings + SDClip + loop45x2 | 1.08563 | ~15,985,678 | SP8192 | 11L, 512d, 8H/4KV, MLP 4x, loop layers 4-5 twice | Row-normalized Muon | GPTQ embeddings, SDClip k=12.85 matrices, k=20 embeds | Sliding only | ~4990 steps on 8xH100 | 8192 vocab plus embedding GPTQ | Track A fixed predictor |
| 6 | 2026-04-04 SP4096 depth recurrence + parallel residual + MuonEq-R | 1.08972 | 15,988,473-15,999,533 | SP4096 | 11L, 512d, MLP 4x, layers 4-5 recurrence | MuonEq-R, WD .090 | Full GPTQ int6 + Brotli | Sliding only | 8xH100 | SP4096 recurrence + parallel residual bridge | Track A fixed predictor |
| 7 | 2026-04-03 MuonEq-R + depth recurrence + all-int6 | 1.09120 | 15,959,253-15,967,483 | SP4096 | 11L + 2 virtual, 512d, MLP 4x | MuonEq-R, WD .090, EMA .997 | All 66 layers int6 GPTQ | Sliding only | train ~590s, eval ~83s | WD/compression synergy buys all-int6 | Track A fixed predictor |
| 8 | 2026-04-01 Vocab4096 MLP4 WD085 | 1.09785 | ~15,916,170 | SP4096 | 11L, 512d, MLP 4x | Muon + DDP, WD .085 | GPTQ + byte shuffle + Brotli | Sliding only | 8xH100 | Larger vocab and high WD simplify older stack | Track A fixed predictor |
| 9 | 2026-03-31 parallel residual + mini depth recurrence | 1.10625 | 15,919,617-15,946,657 | SP1024 lineage | 11L, parallel residual, delayed recurrence layers 4-5 | Muon lineage | Mixed quant + AR self-generated GPTQ | Sliding only | ~6243 steps, 96 ms/step | Learned two-lane residual routing | Track A fixed predictor |
| 10 | 2026-03-29 loader + full GPTQ + XSA11 + BigramHash | 1.11220 | 15,973,962-15,983,626 | SP1024 | 11L, 512d, 8H/4KV, MLP 3x | Parallel Muon + EMA | Full Hessian GPTQ, BigramHash 2816x112 | Sliding only; TTT was neutral/slightly negative | train ~586s, eval ~87s | Coprime loader + XSA all layers + BigramHash | Track A fixed predictor |

## Winning Invariants

- H100-specific FA3 path matters; local CPU/MPS smoke is only structural.
- Artifact budget is shaped by compression entropy, not just raw bit width.
- GPTQ with SDClip is the dominant quantization baseline.
- Bigger tokenizer won once SP8192 could fit; SP4096 is now a fallback, not the base.
- MuonEq-R, high weight decay, and EMA are table stakes.
- Recurrence is useful only when targeted and delayed enough to preserve step count.
- Parallel residuals are now part of the strongest stack.
- Legal TTT must score before update and must avoid caches, biasing, rescoring, or validation leakage.

## Fragile Tricks

- Margins are tiny: the best stack has only single-digit KB artifact headroom.
- TTT gains are real but bounded by eval time and compliance scrutiny.
- Increasing QK gain helped through 5.25, but likely has a sharp over-tuning cliff.
- Hessian-aware per-row SDClip had small gains and can hurt compression when lambda is too high.
- Extra recurrence depth trades BPB against wallclock; late activation is important.
- Any change to wrapper/code size can break the 16MB budget.

## Open Attack Surfaces

- Per-group SDClip allocation using stable Hessian group traces instead of noisy per-row traces.
- QK gain and recurrence-start joint tuning around the 5.25 / 0.35 frontier.
- TTT schedule improvements that preserve score-first legality: chunk size, freeze set, LR decay, optimizer.
- Artifact wrapper size: code compression, metadata layout, and optional runtime imports.
- Calibration set ordering and shard sampling for Hessian collection.
- Parallel residual start layer and lane merge initialization.

## Innovation Queue

1. **Per-group SDClip allocation**: use early/loop/mid/late group budgets rather than row-level Hessian noise.
2. **TTT freeze sweep**: compare all-trainable vs scalar/late-layer/final-norm-only updates under score-first ordering.
3. **QK gain micro-sweep**: 5.10, 5.25, 5.40 on the exact base; do not combine until measured.
4. **Recurrence activation sweep**: 0.32, 0.35, 0.38 activation for 3-layer recurrence.
5. **Artifact wrapper trim**: keep code size stable or lower before adding any feature.

## Recommended Base Stack

Use `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` unchanged as the reproduction base.

Reason: it is the strongest inspected legal public record, combines the best SP8192 quantization stack with parallel residuals and legal TTT, and already provides 3-seed 8xH100 logs under artifact/train/eval limits.

## Reproducibility Commands

Base cloud debug:

```bash
python3 data/cached_challenge_fineweb.py --variant sp8192

cd records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack

RUN_ID=h100_debug_seed0 \
SEED=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train_h100_debug_seed0.log
```

Report after Phase 4:

- completed: pending
- val_bpb: pending
- val_loss: pending
- compressed artifact bytes: pending
- train wallclock: pending
- eval wallclock: pending
- CUDA/GPU used: pending
- warnings/errors: pending
- artifact under 16,000,000 bytes: pending
- README/notes command corrections needed: pending

Base 8xH100:

```bash
RUN_ID=seed42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train_seed42.log
```

## Artifact Size Logic

The base script logs:

- serialized raw model bytes
- code bytes
- quantized compressed model bytes
- total submission size as `code_bytes + quant_file_bytes`

The critical size path is `gptq_mixed_quantize` -> byte shuffle -> Brotli-11 -> total-size logging. The 2026-04-09 base uses LZMA wrapping to reduce source bytes, while the model payload uses Brotli.

## Ablation A1: Per-Group SDClip Allocation

Blocked until Phase 4 base debug completes.

- Hypothesis: Stable group-level Hessian traces can allocate slightly wider/narrower SDClip thresholds by layer group, reducing quantization error without increasing compressed bytes.
- Expected BPB gain: 0.0002-0.0006 if tuned conservatively.
- Implementation risk: medium; touching quantization can silently blow artifact size.
- Artifact-size cost: target neutral; reject if worst seed loses more than 2KB margin.
- Wallclock cost: low if implemented as a multiplier over existing GPTQ pass.
- Legality risk: low; train-time quantization only.
- Rollback plan: restore uniform `matrix_clip_sigmas=12.85`.
- Smoke command: `python -m py_compile train_gpt.py`.
- Cloud measurement command: `RUN_ID=a1_sdclip_seed0 SEED=0 MAX_WALLCLOCK_SECONDS=600 torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train_a1_sdclip_seed0.log`.

## Ablation A2: TTT Freeze Schedule

Blocked until Phase 4 base debug completes.

- Hypothesis: Updating fewer tensors during score-first TTT may keep most of the adaptation gain while reducing eval time and overfit risk.
- Expected BPB gain: -0.0002 to +0.0004; speed improvement may still justify keeping.
- Implementation risk: medium; parameter filtering must be explicit and logged.
- Artifact-size cost: none.
- Wallclock cost: likely lower during eval.
- Legality risk: medium-low; must preserve score-before-update and single-pass scoring.
- Rollback plan: return to all TTT params from base.
- Smoke command: `python -m py_compile train_gpt.py`.
- Cloud measurement command: `RUN_ID=a2_ttt_freeze_seed0 SEED=0 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train_a2_ttt_freeze_seed0.log`.

## Ablation A3: QK Gain Micro-Sweep

Blocked until Phase 4 base debug completes.

- Hypothesis: 5.25 is near optimum but 5.10 or 5.40 may improve one-seed debug enough to justify 3-seed testing.
- Expected BPB gain: 0.0001-0.0004.
- Implementation risk: low if controlled by env var.
- Artifact-size cost: none.
- Wallclock cost: none.
- Legality risk: low.
- Rollback plan: use `QK_GAIN_INIT=5.25`.
- Smoke command: `python -m py_compile train_gpt.py`.
- Cloud measurement command: `for QK_GAIN_INIT in 5.10 5.25 5.40; do RUN_ID=a3_qk_${QK_GAIN_INIT}_seed0 SEED=0 QK_GAIN_INIT=$QK_GAIN_INIT torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train_a3_qk_${QK_GAIN_INIT}_seed0.log; done`.

## Local Acceptance Checklist

- [x] `train_gpt.py` compiles.
- [x] Folder structure complete.
- [x] `README.md` exists.
- [x] `submission.json` schema shape matches inspected records.
- [x] `notes.md` includes leaderboard comparison.
- [x] Artifact-size logic identified.
- [x] Exact cloud debug command prepared.
- [x] Tiny smoke decision recorded: skipped locally because CUDA is unavailable.
- [x] Leaderboard readiness explicitly denied until cloud validation.

## Cloud Acceptance Checklist

- [ ] 1xH100 debug has no crash.
- [ ] Final BPB prints.
- [ ] Final val_loss prints.
- [ ] Compressed artifact size prints.
- [ ] Train wallclock is known.
- [ ] Eval wallclock is known.
- [ ] CUDA/GPU used is recorded.
- [ ] Warnings/errors are summarized.
- [ ] Artifact under 16,000,000 bytes is checked.
- [ ] README/notes command corrections are applied if needed.
- [ ] `train_h100_debug_seed0.log` is saved in this record folder.
- [ ] 8xH100 run only after 1xH100 passes.
- [ ] Three final seeds complete.
- [ ] README and submission metadata updated with real results.

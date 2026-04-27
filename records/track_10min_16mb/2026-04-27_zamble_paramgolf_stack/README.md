# Zamble Parameter Golf Stack

Local scaffold for a leaderboard-valid OpenAI Parameter Golf attempt.

This folder intentionally starts from the strongest inspected legal public stack:
`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`.

Local results are only structural checks. Leaderboard claims require cloud runs on H100s.

## Base Stack

- Tokenizer: SentencePiece BPE 8192 (`sp8192`)
- Model: 11 physical layers, 512d, 8 query heads, 4 KV heads, MLP 4x
- Recurrence: 3-layer depth recurrence over layers 3-5
- Residual form: parallel residuals from layer 7+
- Optimizer: MuonEq-R plus AdamW for embeddings/scalars
- Schedule: warmdown-heavy training, EMA 0.9965
- Quantization: GPTQ int6 matrices, int8 embeddings, SDClip, byte shuffle, Brotli
- Eval: legal score-first TTT, no SLOT, no ETLB, no n-gram cache

## Local Checks

From this folder:

```bash
python -m py_compile train_gpt.py
```

From repo root:

```bash
git diff --check
```

Tiny smoke is optional and only proves process shape:

```bash
RUN_ID=local_smoke \
SEED=0 \
MAX_WALLCLOCK_SECONDS=60 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

If CUDA is unavailable locally, skip smoke. Do not report local smoke as BPB evidence.

## Cloud Debug

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
git checkout -b zamble-paramgolf-attempt

# copy this folder into:
# records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192

cd records/track_10min_16mb/2026-04-27_zamble_paramgolf_stack

RUN_ID=h100_debug_seed0 \
SEED=0 \
MAX_WALLCLOCK_SECONDS=600 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train_h100_debug_seed0.log
```

## Cloud Record Runs

Run only after the 1xH100 debug and measured ablations pass.

```bash
RUN_ID=seed0 SEED=0 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train_seed0.log

RUN_ID=seed42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train_seed42.log

RUN_ID=seed1234 SEED=1234 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train_seed1234.log
```

## Status

- Local scaffold: complete
- Local compile: passed on 2026-04-27
- Tiny smoke: skipped locally because CUDA is unavailable
- 1xH100 debug: pending
- 8xH100 record logs: pending

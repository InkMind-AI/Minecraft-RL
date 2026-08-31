#!/bin/bash
# Stage III: full-trajectory multi-step loss on parquet.
# Usage:
#   MODEL_PATH=<stage2 checkpoint> bash train_stage3.sh [--nproc N] [--attn-impl sdpa|flash_attention_2] [--linear-attn-kernels]
#
# FOCAL_DECAY (env var, default 0.75): VeOmni's repeated-action loss suppression --
# see collators.py::MultiStepVLMCollator. Set FOCAL_DECAY=1.0 to disable.
#
# KEEP_NO_OP_P (env var, default 1.0 = off): OpenHA/VPT-style DATA-level no-op dropping --
# probability of keeping each pure-no-op frame; a dropped frame loses both its observation
# image and its action. See dataset.py::_no_op_dropped_turns. Independent of and composable
# with FOCAL_DECAY (e.g. KEEP_NO_OP_P=0.2 FOCAL_DECAY=0.75).
#
# --linear-attn-kernels: opt-in, installs causal-conv1d + flash-linear-attention so
# Qwen3.5's linear_attention layers use their fused kernel instead of transformers'
# pure-PyTorch fallback (benchmarked ~24.5x faster on the core op alone -- see
# common.sh's LINEAR_ATTN_KERNELS comment). No effect on Qwen2-VL/other architectures
# without linear attention layers.
#
# W&B: see train_stage1.sh header. A per-stage default run name is applied below.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/data/train-*.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage3-output}"
MAX_STEPS="${MAX_STEPS:-3400}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-19456}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
# VeOmni's `qwen2_5vlwithfocal` repeat-suppression (see collators.py::MultiStepVLMCollator
# docstring): 0.75 matches VeOmni's own published TextVLA training recipe. Set to 1.0 to
# disable (restores the plain "every assistant token gets equal loss weight" behaviour).
FOCAL_DECAY="${FOCAL_DECAY:-0.75}"
# Data-level no-op dropping (dataset.py::_no_op_dropped_turns). Defaults to 1.0 (OFF) so
# this stays byte-identical to previous runs unless explicitly opted into: OpenHA's own
# text-action route also leaves its `keep_no_op_p` at 1.0 and relies on focal masking
# alone. Set e.g. KEEP_NO_OP_P=0.2 to additionally remove ~80% of the 24.8% pure-no-op
# steps, matching VPT's "skip null actions" data pipeline more closely.
KEEP_NO_OP_P="${KEEP_NO_OP_P:-1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

: "${MODEL_PATH:?stage3 needs an explicit MODEL_PATH (a Stage II checkpoint), e.g. MODEL_PATH=s3://.../stage2-qwen35-9b bash train_stage3.sh}"

export WANDB_RUN_NAME="${WANDB_RUN_NAME:-minecraft-sft-stage3-qwen35-9b}"

# Upload every saved checkpoint to S3 right after it's written (crash-safe resume:
# a 48h deadline kill loses at most the in-flight checkpoint). Override with
# S3_OUTPUT_DIR=... or disable with S3_OUTPUT_DIR="" (empty string).
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://arcwm-code-us-west-2/axiom/model/$WANDB_RUN_NAME}"

echo "=== Stage III training (full-trajectory multi-step loss): NPROC=$NPROC ==="
bootstrap_env
localize_stage3_parquet_dataset
echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS S3_OUTPUT_DIR=$S3_OUTPUT_DIR"

# Crash-safe resume: if the local output dir has no checkpoint yet but S3 does
# (e.g. a previous run was killed by the 48h deadline), pull the latest checkpoints
# down first so `--resume_from_checkpoint auto` can pick them up.
if [ -n "$S3_OUTPUT_DIR" ] && ! compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null 2>&1; then
    echo "No local checkpoint; pulling latest from S3: $S3_OUTPUT_DIR"
    # For a brand-new S3_OUTPUT_DIR (fresh run, no checkpoint uploaded yet), `aws s3
    # ls` on a nonexistent/empty prefix itself exits 1 (confirmed: not just "no
    # output" -- an actual nonzero exit), independent of the `grep -oE ... || true`
    # below. Under `set -o pipefail`, that alone makes the WHOLE pipe's exit status
    # nonzero (pipefail reports the rightmost nonzero exit among ALL stages, not just
    # the last one) even though grep/sort/tail all individually succeed. This
    # assignment is a bare statement, not inside an `if`/`&&`, so `set -e` then kills
    # the whole script right here -- silently, with no error message, before ever
    # reaching `torchrun` below. Confirmed by two debug-pod repros: fixing only the
    # grep (a prior, INCOMPLETE fix) still died at this exact line; the trailing
    # `|| true` on the full assignment (neutralizing `aws s3 ls`'s own exit code) is
    # what actually gets past it. This is also the confirmed cause of two real 8-GPU
    # training jobs failing in 2-5 minutes with no further log output.
    LATEST_CKPT=$(aws s3 ls "$S3_OUTPUT_DIR/" 2>/dev/null | { grep -oE 'checkpoint-[0-9]+' || true; } | sort -t- -k2 -n | tail -1) || true
    if [ -n "$LATEST_CKPT" ]; then
        echo "Latest: $LATEST_CKPT"
        if command -v s5cmd >/dev/null 2>&1; then
            s5cmd sync --concurrency 16 "${S3_OUTPUT_DIR%/}/${LATEST_CKPT}/*" "$OUTPUT_DIR/${LATEST_CKPT}/"
        else
            aws s3 sync "$S3_OUTPUT_DIR/$LATEST_CKPT" "$OUTPUT_DIR/$LATEST_CKPT" --only-show-errors
        fi
        if [ -f "$OUTPUT_DIR/$LATEST_CKPT/latest" ]; then
            echo "Checkpoint downloaded OK: $LATEST_CKPT"
        else
            echo "ERROR: checkpoint download failed" >&2
            exit 1
        fi
    else
        echo "No checkpoint found on S3, starting from scratch."
    fi
fi

torchrun --nproc_per_node="$NPROC" --tee 3 train_sft.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --data_format parquet \
    --download_model "$DOWNLOAD_CACHE" \
    --output_dir "$OUTPUT_DIR" \
    --s3_output_dir "$S3_OUTPUT_DIR" \
    --resume_from_checkpoint auto \
    --full_trajectory \
    --focal_decay "$FOCAL_DECAY" \
    --keep_no_op_p "$KEEP_NO_OP_P" \
    --attn_implementation "$ATTN_IMPL" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --gradient_checkpointing \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --learning_rate 8e-6 \
    --weight_decay 0.05 \
    --warmup_steps 102 \
    --lr_scheduler_type cosine \
    --deepspeed ds_zero2_no_offload.json \
    --save_steps 200 \
    --logging_steps 10

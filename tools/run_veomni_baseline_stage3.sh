#!/bin/bash
# ============================================================================
# OpenHA/VeOmni ORIGINAL-CODEBASE Stage III baseline (TextVLA), for a like-for-like
# comparison against our own trl_sft Stage III runs.
#
# Same inputs as our runs:
#   - init weights: Qwen2-VL-7B-Instruct-stage2-8gpu-20260817 (our Stage II ckpt)
#   - data: minecraft-text-action-dataset (all 363 parquet shards, ~215K trajectories)
#   - 8 GPUs, 1 node
# Differences are exactly the thing under test: VeOmni's own trainer
# (FSDP2 + Ulysses SP=4, its own chat template / focal masking / vit_lr split)
# instead of our TRL SFTTrainer + DeepSpeed ZeRO-2 stack.
#
# Hyperparameters mirror OpenHA's own published TextVLA script
# (OpenHA/CrossAgent/SFT/sft_script/vla-qwen2-vl-7b-text_250929.sh):
#   lr 8e-6, vit_lr 4e-6, lr_min 1e-6, weight_decay 0.05, cosine, seed 42,
#   max_seq_len 19456, ulysses_parallel_size 4, micro_batch_size 4, grad_accum 1,
#   warmup = clamp(3% of max_steps, 50, 200), rmpad_with_pos_ids, fsdp2.
#
# Two deliberate deviations from that script, both forced:
#   1. --data.chat_template qwen2_5vlwithfocal   (NOT `qwen2_5vlwithfocal_onlylaststep`)
#      The `_onlylaststep` name in OpenHA's script does not exist in VeOmni's
#      TEMPLATES registry (veomni/data/multimodal/multimodal_chat_template.py only
#      registers "qwen2_5vlwithfocal"), so it would fail at build time. The registered
#      focal template is what actually trains on every assistant turn with 0.75^k
#      repeat suppression.
#   2. `pip install swanlab`. tasks/omni/train_qwen2_5_vl.py does a top-level
#      `import swanlab` (line 12); omitting it is what killed the earlier attempt
#      (axiomjin-veomni-realdata-normal-20260826-151300) after the env + model +
#      data steps had all succeeded. SWANLAB_MODE=disabled still needs the import to
#      resolve.
#   3. --train.ckpt_manager dcp   (VeOmni's default is "bytecheckpoint")
#      build_checkpointer() with the default does `from bytecheckpoint import
#      FSDP2Checkpointer`, and that package is not installed -- which killed the
#      second attempt (axiomjin-veomni-base-normal-20260829-002936) right after
#      torchrun started, again with all 363 shards already converted. We cannot simply
#      pip install it either: bytecheckpoint 0.0.2 pins `torch>=2.1.0,<=2.5.0` while
#      VeOmni's own install needs torch 2.5.1, so it would force a torch downgrade.
#      VeOmni already ships the alternative `dcp` manager (plain
#      torch.distributed.checkpoint, requires torch>=2.4 -- satisfied by 2.5.1), so we
#      select that instead of perturbing the torch version.
# ============================================================================
set -e
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false

WORK=/local-ssd/veomni_baseline
MODEL_LOCAL="$WORK/stage2_model"
DATA_RAW="$WORK/parquet"
DATA_JSONL="$WORK/jsonl"          # MUST contain only .jsonl (VeOmni os.listdir()s it)
DATA_IMAGES="$WORK/jsonl_images"  # images live outside DATA_JSONL, see above
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/output}"
S3_OUTPUT="${S3_OUTPUT:-s3://arcwm-code-us-west-2/axiom/model/veomni-baseline-qwen2vl-stage3}"

echo "=== [1/6] Setting up VeOmni conda env ==="
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda env list | grep -q "^veomni " || conda create -y -n veomni python=3.10 2>&1 | tail -3
conda activate veomni

python -c "import torch" 2>/dev/null || \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 2>&1 | tail -3
# swanlab is REQUIRED: train_qwen2_5_vl.py imports it unconditionally at module level.
# VeOmni's custom Qwen2-VL implementation targets Transformers 4.51.x. Pin the
# exact version: the Stage II checkpoint was exported by Transformers 5.15 (its
# config is normalized below), while newer Transformers API changes are not covered
# by VeOmni's model implementation.
python -c "import transformers, veomni, swanlab; assert transformers.__version__ == '4.51.1'" 2>/dev/null || {
    pip install --no-cache-dir "transformers==4.51.1" accelerate datasets peft hf-transfer \
        tensordict torchdata codetiming hydra-core pandas "pyarrow>=15.0.0" pylatexenc qwen-vl-utils \
        wandb swanlab ninja liger-kernel blobfile diffusers tiktoken packaging pillow boto3 s3fs 2>&1 | tail -5
    MAX_JOBS=8 pip install --no-cache-dir flash-attn --no-build-isolation 2>&1 | tail -5
    cd /data/work/run_codes/OpenHA/CrossAgent/SFT/VeOmni
    pip install --no-cache-dir -e . 2>&1 | tail -5
}

echo "=== [2/6] Downloading Stage II checkpoint ==="
mkdir -p "$MODEL_LOCAL"
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
    # Exclude checkpoint-*/ : the Stage II output dir keeps its final merged model at
    # the root and every intermediate DeepSpeed checkpoint (per-rank optimizer states,
    # ~5-8x model size each) in subdirs we must not pull.
    s5cmd sync --concurrency 16 --exclude 'checkpoint-*/*' \
        "s3://arcwm-code-us-west-2/axiom/model/Qwen2-VL-7B-Instruct-stage2-8gpu-20260817/*" "$MODEL_LOCAL/"
fi

# The Stage II model was saved by Transformers 5.15, whose Qwen2-VL config stores
# all text-model fields under `text_config`. VeOmni deliberately pins 4.51.1, where
# Qwen2VLConfig has no `text_config` sub-config: AutoConfig leaves it as a dict, then
# GenerationConfig.from_model_config calls dict.to_dict() during model construction.
# Flatten the new schema into the exact 4.51 schema once, retaining a local source
# backup for diagnosis. This only changes config serialization; model weights and
# processor files are untouched.
python - "$MODEL_LOCAL/config.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open() as f:
    config = json.load(f)

if config.get("model_type") == "qwen2_vl" and isinstance(config.get("text_config"), dict):
    backup_path = config_path.with_name("config.transformers5.json")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    text_config = dict(config.pop("text_config"))
    text_config.pop("model_type", None)
    for key, value in text_config.items():
        config.setdefault(key, value)

    # Transformers 5.15 serializes this as `rope_parameters`; 4.51 expects
    # `rope_scaling` and already receives rope_theta at the top level above.
    rope_parameters = config.pop("rope_parameters", None)
    if rope_parameters is not None:
        rope_parameters.pop("rope_theta", None)
        config["rope_scaling"] = rope_parameters

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Normalized Transformers-5 Qwen2-VL config for VeOmni: {config_path}")
else:
    print(f"VeOmni-compatible Qwen2-VL config already present: {config_path}")
PY

# Transformers 5.15 writes `extra_special_tokens` as a list, while Transformers
# 4.51 assumes this field is a dict and calls `.keys()` while constructing the
# tokenizer. The token inventory itself remains in tokenizer.json, so removing this
# incompatible redundant metadata preserves tokenization and unblocks AutoProcessor.
python - "$MODEL_LOCAL/tokenizer_config.json" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
if not config_path.exists():
    raise FileNotFoundError(f"Missing tokenizer config: {config_path}")
config = json.loads(config_path.read_text())
if isinstance(config.get("extra_special_tokens"), list):
    config.pop("extra_special_tokens")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Removed Transformers-5-only extra_special_tokens metadata: {config_path}")
else:
    print(f"Tokenizer config already compatible with Transformers 4.51: {config_path}")
PY

# Transformers 4.51's Qwen2VLProcessor looks for the legacy flat
# `preprocessor_config.json`, while the Stage II export from Transformers 5.15 only
# carries a merged `processor_config.json`. Without this conversion every rank loads
# model weights successfully, then dies in AutoProcessor.from_pretrained() before the
# dataloader is built. Keep processor_config.json intact as the conversion source; the
# old processor only needs the flat companion file.
python - "$MODEL_LOCAL/processor_config.json" "$MODEL_LOCAL/preprocessor_config.json" <<'PY'
import json
import sys
from pathlib import Path

processor_path = Path(sys.argv[1])
preprocessor_path = Path(sys.argv[2])
if preprocessor_path.exists():
    print(f"Legacy preprocessor config already present: {preprocessor_path}")
elif not processor_path.exists():
    raise FileNotFoundError(
        f"Missing both {preprocessor_path.name} and {processor_path.name}; "
        "cannot construct a Transformers-4.51 Qwen2-VL processor config."
    )
else:
    source = json.loads(processor_path.read_text())
    image_processor = source.get("image_processor")
    if not isinstance(image_processor, dict):
        raise ValueError(f"{processor_path} has no image_processor object")
    size = image_processor.get("size") or {}
    required = ("shortest_edge", "longest_edge")
    if not all(key in size for key in required):
        raise ValueError(f"{processor_path} has invalid image_processor.size: {size!r}")
    output = {
        "min_pixels": size["shortest_edge"],
        "max_pixels": size["longest_edge"],
        "patch_size": image_processor["patch_size"],
        "temporal_patch_size": image_processor["temporal_patch_size"],
        "merge_size": image_processor["merge_size"],
        "image_mean": image_processor["image_mean"],
        "image_std": image_processor["image_std"],
        "image_processor_type": image_processor["image_processor_type"],
        "processor_class": source["processor_class"],
    }
    preprocessor_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Derived legacy preprocessor config for VeOmni: {preprocessor_path}")
PY

# ProcessorMixin in Transformers 4.51 first builds the image processor from the
# legacy file, then also reads the newer merged processor_config.json and passes a
# second image_processor argument into Qwen2VLProcessor, producing
# "multiple values for argument image_processor". Once the flat config exists, move
# the merged source aside so the old loader follows its normal two-component
# (image_processor + tokenizer) path. Keep it under a diagnostic filename rather than
# deleting it.
PROCESSOR_CONFIG="$MODEL_LOCAL/processor_config.json"
if [ -f "$PROCESSOR_CONFIG" ]; then
    mv -f "$PROCESSOR_CONFIG" "${PROCESSOR_CONFIG}.transformers5"
    echo "Moved incompatible merged processor config aside for Transformers 4.51"
fi
ls -la "$MODEL_LOCAL" | head -9

echo "=== [3/6] Downloading text-action parquet (all 363 shards) ==="
mkdir -p "$DATA_RAW"
N_HAVE=$(find "$DATA_RAW" -name '*.parquet' | wc -l)
if [ "$N_HAVE" -lt 363 ]; then
    s5cmd sync --concurrency 16 \
        "s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/data/train-*.parquet" "$DATA_RAW/"
fi
find "$DATA_RAW" -name '*.parquet' | wc -l

echo "=== [4/6] Converting parquet -> VeOmni jsonl (+ images on disk) ==="
mkdir -p "$DATA_JSONL" "$DATA_IMAGES"
# Marker kept OUTSIDE $DATA_JSONL: VeOmni's build_dataset does a bare
# os.listdir($DATA_JSONL) and tries to load EVERY entry as a dataset file, so a
# .done marker (or an images/ subdir) sitting in there would break data loading.
if [ ! -f "$WORK/.jsonl_done" ]; then
    python /data/work/run_codes/Minecraft-CoT/tools/convert_text_action_to_veomni_jsonl.py \
        --parquet_path "$DATA_RAW/*.parquet" \
        --out_dir "$DATA_JSONL" \
        --image_root "$DATA_IMAGES" \
        --num_workers 16
    touch "$WORK/.jsonl_done"
fi
cat "$DATA_JSONL"/*.jsonl | wc -l
du -sh "$DATA_JSONL" "$DATA_IMAGES"

echo "=== [5/6] Launching VeOmni training ==="
cd /data/work/run_codes/OpenHA/CrossAgent/SFT/VeOmni

# VeOmni exposes gradient accumulation through `global_batch_size`: its dataloader
# splits each optimizer update into `global_batch_size / (micro_batch_size * dp_size)`
# micro-batches. Defaults mirror the original OpenHA recipe (global batch 8). Override
# GRAD_ACCUM=8 to match our TRL Stage III effective global batch 64 without changing
# per-rank sequence memory or Ulysses sequence parallelism.
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
ULYSSES="${ULYSSES:-4}"
MICRO_BSZ="${MICRO_BSZ:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
REAL_DATASET_LEN=$(cat "$DATA_JSONL"/*.jsonl | wc -l)
TOTAL_WORKERS=$(( NPROC * NNODES / ULYSSES ))
GLOBAL_BSZ=$(( TOTAL_WORKERS * MICRO_BSZ * GRAD_ACCUM ))
MAX_STEPS=$(( REAL_DATASET_LEN / GLOBAL_BSZ ))

WARMUP_STEPS=$(( MAX_STEPS * 3 / 100 ))
(( WARMUP_STEPS < 50 )) && WARMUP_STEPS=50
(( WARMUP_STEPS > 200 )) && WARMUP_STEPS=200
(( WARMUP_STEPS >= MAX_STEPS )) && WARMUP_STEPS=$(( MAX_STEPS - 1 ))
LR_WARMUP_RATIO=$(python -c "print(f'{$WARMUP_STEPS / $MAX_STEPS:.6f}')")

echo "dataset_len=$REAL_DATASET_LEN global_bsz=$GLOBAL_BSZ max_steps=$MAX_STEPS warmup_steps=$WARMUP_STEPS ratio=$LR_WARMUP_RATIO"

export WANDB_API_KEY=9775ea57c312e2b1445afe756e7e68b72a1307b7
export WANDB_PROJECT="${WANDB_PROJECT:-minecraft-sft}"
export SWANLAB_MODE=disabled
WANDB_NAME="${WANDB_NAME:-veomni-baseline-qwen2vl-stage3}"

# VeOmni has no equivalent of our trl_sft S3CheckpointUploadCallback, so checkpoints
# would only exist on the node's ephemeral /local-ssd until the very end -- a crash or
# the job's wall-clock limit would lose every one of them. Mirror them to S3 in the
# background while training runs (uploads are incremental, so re-syncing the same dir
# is cheap).
mkdir -p "$OUTPUT_DIR"
(
    while true; do
        sleep 600
        s5cmd sync --concurrency 8 "$OUTPUT_DIR/" "$S3_OUTPUT/" >/dev/null 2>&1 || true
    done
) &
UPLOADER_PID=$!
trap 'kill "$UPLOADER_PID" 2>/dev/null || true' EXIT

torchrun --nnodes=$NNODES --nproc-per-node=$NPROC --master-port=24009 \
    tasks/omni/train_qwen2_5_vl.py \
    configs/multimodal/qwen2_vl/qwen2_vl.yaml \
    --model.model_path "$MODEL_LOCAL" \
    --data.train_path "$DATA_JSONL" \
    --data.chat_template qwen2_5vlwithfocal \
    --data.train_size 100000000000000000000 \
    --data.max_seq_len 19456 \
    --data.source_name craftjarvis \
    --data.dataloader_type native \
    --data.datasets_type mapping \
    --train.rmpad_with_pos_ids true \
    --train.seed 42 \
    --train.lr 8e-6 \
    --train.vit_lr 4e-6 \
    --train.lr_min 1e-6 \
    --train.lr_warmup_ratio "$LR_WARMUP_RATIO" \
    --train.weight_decay 0.05 \
    --train.lr_decay_style cosine \
    --train.save_steps 200 \
    --train.max_steps "$MAX_STEPS" \
    --train.num_train_epochs 1 \
    --train.output_dir "$OUTPUT_DIR" \
    --train.data_parallel_mode fsdp2 \
    --train.ckpt_manager dcp \
    --train.wandb_project "$WANDB_PROJECT" \
    --train.wandb_name "$WANDB_NAME" \
    --train.dyn_bsz_buffer_size 200 \
    --train.ulysses_parallel_size $ULYSSES \
    --train.context_parallel_size 1 \
    --train.tensor_parallel_size 1 \
    --train.expert_parallel_size 1 \
    --train.pipeline_parallel_size 1 \
    --train.global_batch_size $GLOBAL_BSZ \
    --train.micro_batch_size $MICRO_BSZ

echo "=== [6/6] Uploading checkpoints to S3 ==="
s5cmd sync --concurrency 16 "$OUTPUT_DIR/" "$S3_OUTPUT/"
echo "=== VEOMNI BASELINE DONE ==="

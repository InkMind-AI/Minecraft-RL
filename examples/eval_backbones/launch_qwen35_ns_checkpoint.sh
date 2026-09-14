#!/bin/bash
# ============================================================================
# 评测 ns-noop1（KEEP_NO_OP_P=1.0，数据级完全不丢弃 no-op 帧）训练的
# Stage III Qwen3.5-9B checkpoint
# —— 对应任务 axiomjin-q35-ns-noop1-*-20260903-*，即：
#   数据: minecraft-text-action-dataset-noop-filtered（删除了 7.15% 纯 no-op 轨迹）
#   训练: focal_decay=0.75 + KEEP_NO_OP_P=1.0（不删帧，训推分布一致）+ fla 线性注意力
#
# 与 launch_qwen35_nf_checkpoint.sh（KEEP_NO_OP_P=0.2/0.7 对照组）的区别仅在于
# 模型目录：minecraft-sft-stage3-qwen35-9b-ns-noop1。
#
# 用法：CKPT=<step> bash launch_qwen35_ns_checkpoint.sh
#   例：CKPT=1000 bash launch_qwen35_ns_checkpoint.sh
# 可用 MODEL_LOCAL_NAME 覆盖输出目录名（复评时避免覆盖旧结果）。
#
# 架构: Qwen3_5ForConditionalGeneration (混合线性/全注意力, 需 vllm>=0.17.0)
# ============================================================================
set -o pipefail
: "${CKPT:?must set CKPT, e.g. CKPT=1000 bash $0}"

export MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-q35ns1-ckpt${CKPT}-20260903}"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen35-9b-ns-noop1/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q35ns1-ckpt${CKPT}"
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0，单独装环境
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

#!/bin/bash
# ============================================================================
# 评测 ns-noop1（KEEP_NO_OP_P=1.0，数据级完全不丢弃 no-op 帧）训练的
# Stage III Qwen2-VL-7B checkpoint
# —— 对应任务 axiomjin-q2vl-ns-noop1-*-20260903-*，即：
#   数据: minecraft-text-action-dataset-noop-filtered（删除了 7.15% 纯 no-op 轨迹）
#   训练: focal_decay=0.75 + KEEP_NO_OP_P=1.0（不删帧，训推分布一致）
#
# 与 launch_qwen2vl_nf_checkpoint.sh（KEEP_NO_OP_P=0.7 对照组）的区别仅在于
# 模型目录：minecraft-sft-stage3-qwen2vl-7b-ns-noop1。
#
# 用法：CKPT=<step> bash launch_qwen2vl_ns_checkpoint.sh
#   例：CKPT=1000 bash launch_qwen2vl_ns_checkpoint.sh
# 可用 MODEL_LOCAL_NAME 覆盖输出目录名（复评时避免覆盖旧结果）。
#
# 架构: Qwen2VLForConditionalGeneration (vllm 0.8.5 直接兼容)
# ============================================================================
set -o pipefail
: "${CKPT:?must set CKPT, e.g. CKPT=1000 bash $0}"

export MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-q2vlns1-ckpt${CKPT}-20260904}"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen2vl-7b-ns-noop1/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q2vlns1-ckpt${CKPT}"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

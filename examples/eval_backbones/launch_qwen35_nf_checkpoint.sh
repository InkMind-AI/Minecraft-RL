#!/bin/bash
# ============================================================================
# 评测 no-op-filtered 数据集（nf- 前缀）训练的 Stage III Qwen3.5-9B checkpoint
# —— 对应任务 axiomjin-q35-nf-noop{2,5,7}-normal-20260902-*，即：
#   数据: minecraft-text-action-dataset-noop-filtered（删除了 7.15% 纯 no-op 轨迹）
#   训练: focal_decay=0.75 + keep_no_op_p（数据级 no-op 帧删除）+ fla 线性注意力
#
# 三个 KEEP_NO_OP_P 对照组：
#   NOOP_TAG=2  -> KEEP_NO_OP_P=0.2
#   NOOP_TAG=5  -> KEEP_NO_OP_P=0.5
#   NOOP_TAG=7  -> KEEP_NO_OP_P=0.7
#
# 用法：NOOP_TAG=<2|5|7> CKPT=<step> bash launch_qwen35_nf_checkpoint.sh
#   例：NOOP_TAG=5 CKPT=400 bash launch_qwen35_nf_checkpoint.sh
#
# 架构: Qwen3_5ForConditionalGeneration (混合线性/全注意力, 需 vllm>=0.17.0)
# ============================================================================
set -o pipefail
: "${NOOP_TAG:?must set NOOP_TAG (2, 5, or 7), e.g. NOOP_TAG=5 CKPT=400 bash $0}"
: "${CKPT:?must set CKPT, e.g. CKPT=400 bash $0}"

export MODEL_LOCAL_NAME="q35nf${NOOP_TAG}-ckpt${CKPT}-20260902"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen35-9b-nf-noop${NOOP_TAG}/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q35nf${NOOP_TAG}-ckpt${CKPT}"
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0，单独装环境
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

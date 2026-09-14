#!/bin/bash
# ============================================================================
# 评测 no-op-filtered 数据集（nf- 前缀）训练的 Stage III Qwen2-VL-7B checkpoint
# —— 对应任务 axiomjin-q2vl-nf-noop7-*-20260903-*，即：
#   数据: minecraft-text-action-dataset-noop-filtered（删除了 7.15% 纯 no-op 轨迹）
#   训练: focal_decay=0.75 + keep_no_op_p=0.7（数据级 no-op 帧删除）
#
# 与 launch_qwen35_nf_checkpoint.sh 同配置的 qwen2vl 底座对照。
#
# 用法：NOOP_TAG=<7> CKPT=<step> bash launch_qwen2vl_nf_checkpoint.sh
#   例：NOOP_TAG=7 CKPT=1000 bash launch_qwen2vl_nf_checkpoint.sh
# 可用 MODEL_LOCAL_NAME 覆盖输出目录名（复评时避免覆盖旧结果）。
#
# 架构: Qwen2VLForConditionalGeneration (vllm 0.8.5 直接兼容)
# ============================================================================
set -o pipefail
: "${NOOP_TAG:?must set NOOP_TAG (e.g. 7), e.g. NOOP_TAG=7 CKPT=1000 bash $0}"
: "${CKPT:?must set CKPT, e.g. CKPT=1000 bash $0}"

export MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-q2vlnf${NOOP_TAG}-ckpt${CKPT}-20260903}"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen2vl-7b-nf-noop${NOOP_TAG}/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q2vlnf${NOOP_TAG}-ckpt${CKPT}"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

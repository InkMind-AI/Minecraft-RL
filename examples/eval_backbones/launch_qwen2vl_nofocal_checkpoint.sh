#!/bin/bash
# ============================================================================
# 评测 focal 消融（FOCAL_DECAY=1.0，关闭 focal）训练的 Stage III Qwen2-VL-7B
# checkpoint —— 对应任务 axiomjin-q2vl-{nf7,ns1}-nofocal-*-20260904/05-*，即：
#   数据: minecraft-text-action-dataset-noop-filtered
#   训练: focal 关闭 + KEEP_NO_OP_P（VARIANT=nf7 -> 0.7；VARIANT=ns1 -> 1.0）
#
# 用法：VARIANT=<nf7|ns1> CKPT=<step> bash launch_qwen2vl_nofocal_checkpoint.sh
#   例：VARIANT=nf7 CKPT=3122 bash launch_qwen2vl_nofocal_checkpoint.sh
# 可用 MODEL_LOCAL_NAME 覆盖输出目录名（复评时避免覆盖旧结果）。
#
# 架构: Qwen2VLForConditionalGeneration (vllm 0.8.5 直接兼容)
# ============================================================================
set -o pipefail
: "${VARIANT:?must set VARIANT (nf7 or ns1), e.g. VARIANT=nf7 CKPT=3122 bash $0}"
: "${CKPT:?must set CKPT, e.g. CKPT=3122 bash $0}"

export MODEL_LOCAL_NAME="${MODEL_LOCAL_NAME:-q2vl${VARIANT}nof-ckpt${CKPT}-20260905}"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen2vl-7b-${VARIANT}-nofocal/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q2vl${VARIANT}nof-ckpt${CKPT}"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

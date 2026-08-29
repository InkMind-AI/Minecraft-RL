#!/bin/bash
# ============================================================================
# 评测带 focal 重复动作抑制机制 + fla 线性注意力加速的 Stage III Qwen3.5-9B
# 训练任务某个中间 checkpoint —— axiomjin-q35-focal-normal-20260828-151332
#
# 用法：CKPT=<step> bash launch_qwen35_focal_checkpoint.sh
#   例：CKPT=400 bash launch_qwen35_focal_checkpoint.sh
#
# 模型信息：
#   基座: Qwen3.5-9B-stage2-8gpu-20260817
#   训练: --full_trajectory SFT on minecraft-text-action-dataset, max_steps=3400,
#         collators.py::MultiStepVLMCollator 带 --focal_decay 0.75（VeOmni
#         qwen2_5vlwithfocal 的连续重复动作降权机制）+ --linear-attn-kernels
#         (causal-conv1d + flash-linear-attention 加速 24/32 层 linear_attention)。
#   来源: s3://.../minecraft-sft-stage3-qwen35-9b-focal/checkpoint-${CKPT}/
#   架构: Qwen3_5ForConditionalGeneration (混合线性/全注意力, 需 vllm>=0.17.0)
# ============================================================================
set -o pipefail
: "${CKPT:?must set CKPT, e.g. CKPT=400 bash $0}"

export MODEL_LOCAL_NAME="q35-focal-ckpt${CKPT}-20260828"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen35-9b-focal/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q35-focal-ckpt${CKPT}"
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0，单独装环境
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

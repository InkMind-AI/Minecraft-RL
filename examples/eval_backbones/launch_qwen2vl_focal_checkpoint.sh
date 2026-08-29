#!/bin/bash
# ============================================================================
# 评测带 focal 重复动作抑制机制的 Stage III Qwen2-VL-7B 训练任务
# 某个中间 checkpoint —— axiomjin-q2vl-focal-normal-20260828-151332
#
# 用法：CKPT=<step> bash launch_qwen2vl_focal_checkpoint.sh
#   例：CKPT=400 bash launch_qwen2vl_focal_checkpoint.sh
#
# 模型信息：
#   基座: Qwen2-VL-7B-Instruct-stage2-8gpu-20260817
#   训练: --full_trajectory SFT on minecraft-text-action-dataset, max_steps=3400,
#         collators.py::MultiStepVLMCollator 带 --focal_decay 0.75（VeOmni
#         qwen2_5vlwithfocal 的连续重复动作降权机制），对照组是不带 focal 的
#         q2vl-eosfix checkpoint（launch_qwen2vl_eosfix_checkpoint.sh）。
#   来源: s3://.../minecraft-sft-stage3-qwen2vl-focal/checkpoint-${CKPT}/
#   架构: Qwen2VLForConditionalGeneration (vllm 0.8.5 直接兼容)
# ============================================================================
set -o pipefail
: "${CKPT:?must set CKPT, e.g. CKPT=400 bash $0}"

export MODEL_LOCAL_NAME="q2vl-focal-ckpt${CKPT}-20260828"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen2vl-focal/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q2vl-focal-ckpt${CKPT}"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

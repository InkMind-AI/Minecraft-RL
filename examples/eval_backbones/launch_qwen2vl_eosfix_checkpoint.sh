#!/bin/bash
# ============================================================================
# 评测正在进行的 Stage III（EOS-loss 修复版）Qwen2-VL-7B 训练任务
# 某个中间 checkpoint —— axiomjin-q2vl-eosfix-normal-20260827-140645
#
# 用法：CKPT=<step> bash launch_qwen2vl_eosfix_checkpoint.sh
#   例：CKPT=200 bash launch_qwen2vl_eosfix_checkpoint.sh
#
# 模型信息：
#   基座: Qwen2-VL-7B-Instruct-stage2-8gpu-20260817
#   训练: --full_trajectory SFT on minecraft-text-action-dataset, max_steps=3400,
#         collators.py 已修复 assistant轮 EOS 被误 mask 的问题
#   来源: s3://.../minecraft-sft-stage3-qwen2vl-eosfix/checkpoint-${CKPT}/
#         (checkpoint-${CKPT}/ 根目录下直接是合并好的 model.safetensors，
#          global_step${CKPT}/ 子目录的 DeepSpeed 优化器分片由
#          run_backbone_eval.sh 的 --exclude 'global_step*/*' 自动排除)
#   架构: Qwen2VLForConditionalGeneration (vllm 0.8.5 直接兼容)
#   注意: 中间 checkpoint 训练尚未收敛，仅用于早期抽查格式/趋势，不代表最终效果。
# ============================================================================
set -o pipefail
: "${CKPT:?must set CKPT, e.g. CKPT=200 bash $0}"

export MODEL_LOCAL_NAME="q2vl-eosfix-ckpt${CKPT}-20260827"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen2vl-eosfix/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q2vl-eosfix-ckpt${CKPT}"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

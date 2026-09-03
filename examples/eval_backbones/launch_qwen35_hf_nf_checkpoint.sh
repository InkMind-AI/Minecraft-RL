#!/bin/bash
# ============================================================================
# 【hf直载模式】评测 no-op-filtered 数据集（nf- 前缀）训练的 Stage III Qwen3.5-9B
# checkpoint —— 与 launch_qwen35_nf_checkpoint.sh 评测同一个 checkpoint，但推理
# 后端是 transformers 进程内直载（--vlm_client_mode hf），不起 vLLM 服务。
#
# 用途：hf 直载 vs vLLM 服务的"效果 + 效率"对照实验。结果输出到独立的
# *-hf 目录，与 vLLM 评测结果互不覆盖。
#
# 用法：NOOP_TAG=<2|5|7> CKPT=<step> bash launch_qwen35_hf_nf_checkpoint.sh
#   例：NOOP_TAG=7 CKPT=1000 bash launch_qwen35_hf_nf_checkpoint.sh
# ============================================================================
set -o pipefail
: "${NOOP_TAG:?must set NOOP_TAG (2, 5, or 7)}"
: "${CKPT:?must set CKPT, e.g. CKPT=1000}"

export MODEL_LOCAL_NAME="q35nf${NOOP_TAG}-ckpt${CKPT}-hf-20260903"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen35-9b-nf-noop${NOOP_TAG}/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q35nf${NOOP_TAG}-ckpt${CKPT}-hf"
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_hf_backbone_eval.sh"

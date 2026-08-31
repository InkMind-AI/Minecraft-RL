#!/bin/bash
# ============================================================================
# 评测带 data-level keep_no_op_p no-op 帧删除 + focal 重复动作抑制 + fla 线性
# 注意力加速的 Stage III Qwen3.5-9B 训练任务某个中间 checkpoint。
#
# 三个并行的对照实验（同一基座/数据/超参，仅 KEEP_NO_OP_P 不同）：
#   NOOP_TAG=2  -> axiomjin-q35-noop2-normal-20260831-171123 (KEEP_NO_OP_P=0.2)
#   NOOP_TAG=5  -> axiomjin-q35-noop5-normal-20260831-171134 (KEEP_NO_OP_P=0.5)
#   NOOP_TAG=7  -> axiomjin-q35-noop7-normal-20260831-171142 (KEEP_NO_OP_P=0.7)
#
# 用法：NOOP_TAG=<2|5|7> CKPT=<step> bash launch_qwen35_noop_checkpoint.sh
#   例：NOOP_TAG=2 CKPT=600 bash launch_qwen35_noop_checkpoint.sh
#
# 模型信息：
#   基座: Qwen3.5-9B-stage2-8gpu-20260817
#   训练: --full_trajectory SFT on minecraft-text-action-dataset, max_steps=3400,
#         collators.py::MultiStepVLMCollator 默认 --focal_decay 0.75（连续重复
#         动作 loss 降权）+ dataset.py --keep_no_op_p（数据级 no-op 帧删除，
#         见 dataset.py::_no_op_dropped_turns）+ --linear-attn-kernels。
#   来源: s3://.../minecraft-sft-stage3-qwen35-9b-noop${NOOP_TAG}/checkpoint-${CKPT}/
#   架构: Qwen3_5ForConditionalGeneration (混合线性/全注意力, 需 vllm>=0.17.0)
# ============================================================================
set -o pipefail
: "${NOOP_TAG:?must set NOOP_TAG (2, 5, or 7), e.g. NOOP_TAG=2 CKPT=600 bash $0}"
: "${CKPT:?must set CKPT, e.g. CKPT=600 bash $0}"

export MODEL_LOCAL_NAME="q35-noop${NOOP_TAG}-ckpt${CKPT}-20260831"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-sft-stage3-qwen35-9b-noop${NOOP_TAG}/checkpoint-${CKPT}/"
export SERVED_MODEL_NAME="eval-q35-noop${NOOP_TAG}-ckpt${CKPT}"
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0，单独装环境
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

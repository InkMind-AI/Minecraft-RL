#!/bin/bash
# 评测 verl GRPO 训练产出的 HF 存档（rl_train/verl_jobs/run_grpo_minecraft_train.sh 上传的
# s3://.../minecraft-verl-grpo/<EXP>/global_step_<STEP>/）。
#
# 协议固定为 easy-h29，与 SFT 起点 v2-e4 的基线逐项一致，结果可以直接对比：
#   - EVAL_BENCHMARK=easy：202 个非 GUI 任务（Embodied 153 + Combat 49），seed=42 固定
#   - ROLLOUTS_PER_TASK=3：共约 606 次 rollout
#   - MAXIMUM_HISTORY_LENGTH=29 / LIMIT_MM_IMAGE=30：与训练时的 h29 上下文一致
#   - 基线：cal-v2e4-ckpt520-easy-h29 = 28.9%（174/603），原测 cotv2-e4-ckpt520-easy-h29 = 29.2%
# 202 个任务里有 88 个属于训练任务池（tasks/mixed_v2e4_easy_h29.txt），另外 114 个没参与
# 训练，可以分开统计训练内与训练外的表现。
#
# 用法：
#   EXP=grpo_h29_mixed88_resume20 STEP=50 TOTAL_STEP=70 bash submit_eval_job.sh launch_verl_grpo_checkpoint.sh
# EXP/STEP 定位存档；TOTAL_STEP 是从 SFT 起点算起的总 GRPO 步数（续训任务的 step 是从
# 起点重新计数的），只用于结果命名，不填时等于 STEP。
set -o pipefail
: "${EXP:?must set EXP, e.g. EXP=grpo_h29_mixed88_resume20}"
: "${STEP:?must set STEP, e.g. STEP=50}"
TOTAL_STEP="${TOTAL_STEP:-$STEP}"

export EVAL_BENCHMARK="${EVAL_BENCHMARK:-easy}"
export ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-3}"
export MAXIMUM_HISTORY_LENGTH="${MAXIMUM_HISTORY_LENGTH:-29}"
export LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-30}"

export MODEL_LOCAL_NAME="verl-${EXP#grpo_}-s${STEP}-t${TOTAL_STEP}-easy-h29"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-verl-grpo/${EXP}/global_step_${STEP}/"
export SERVED_MODEL_NAME="eval-verl-t${TOTAL_STEP}"
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"

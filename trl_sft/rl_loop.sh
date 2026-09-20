#!/bin/bash
# RL 迭代式主循环（自研 GRPO v1，09-20 设计）
#
# 架构（迭代式，非在线——第一版牺牲吞吐换简单性，跑通后升级热同步）：
#   while 未收敛:
#     ① 任务采样 + G 路 rollout（复用现有 eval harness：vLLM 起服务 + Malmo 并行）
#     ② episode 转训练样本（rollout 输出的 raw_action + 帧序列 → parquet，同 SFT 格式）
#     ③ grpo_core 离线算组内优势（任务为组，成功/失败为奖励）
#     ④ train_grpo.py 更新策略（DeepSpeed ZeRO-2）
#     ⑤ 新权重发布 → 下一轮 ① 用新权重起 vLLM
#
# 状态（09-20）：阶段 1 完成（trainer + advantage 核心 + 单测）；
# 阶段 2 中 ② 的 episode→parquet 适配器待写（需核对 episode.jsonl 的帧存储格式）。
#
# 用法（单轮手动验证各环节）：
#   bash rl_loop.sh --policy s3://.../rl_policy_iter0 --iter 1
set -euo pipefail

POLICY="${POLICY:?需设置 POLICY=当前策略权重(S3)}"
ITER="${ITER:-0}"
RL_ROOT="${RL_ROOT:-/local-ssd/rl}"
GROUP_SIZE="${GROUP_SIZE:-8}"          # 每任务 rollout 数（GRPO 组大小）
N_TASKS="${N_TASKS:-50}"              # 每轮任务数
EVAL_BENCHMARK="${EVAL_BENCHMARK:-easy}"
LR="${RL_LR:-1e-6}"
S3_OUT="${S3_OUT:-s3://arcwm-code-us-west-2/axiom/model/minecraft-rl-policy}"

mkdir -p "$RL_ROOT"

echo "=== [RL iter $ITER] ① rollout（${N_TASKS}任务 × ${GROUP_SIZE}路）==="
# 复用现有评测管线：ROLLOUTS_PER_TASK=$GROUP_SIZE 即 GRPO 的组采样
export REPO_ROOT=/data/work/run_codes/Minecraft-CoT
export EVAL_BENCHMARK="$EVAL_BENCHMARK" ROLLOUTS_PER_TASK="$GROUP_SIZE"
export MAXIMUM_HISTORY_LENGTH=29 LIMIT_MM_IMAGE=30
export MODEL_LOCAL_NAME="rl-iter${ITER}"
export MODEL_S3_URI="$POLICY"
export SERVED_MODEL_NAME="rl-policy-iter${ITER}"
export VLLM_CONDA_ENV=vllm35
cd "$REPO_ROOT"
bash examples/eval_backbones/run_backbone_eval.sh || true   # 输出含 per-episode 成败 + raw_action

echo "=== [RL iter $ITER] ② episode → 训练样本 parquet（适配器待接）==="
python3 trl_sft/rl_build_batch.py \
    --eval-output "/local-ssd/eval_output/${MODEL_LOCAL_NAME}-text_action" \
    --out "$RL_ROOT/batch_${ITER}.parquet" \
    --rewards "$RL_ROOT/rewards_${ITER}.npy" \
    --groups "$RL_ROOT/groups_${ITER}.json"

echo "=== [RL iter $ITER] ③ 组内优势 ==="
python3 -c "
import sys, json, numpy as np
sys.path.insert(0, 'trl_sft')
from grpo_core import compute_group_advantages, group_coverage_report
rewards = np.load('$RL_ROOT/rewards_${ITER}.npy')
groups = json.load(open('$RL_ROOT/groups_${ITER}.json'))
adv, stats = compute_group_advantages(rewards.tolist(), groups, mode='std')
np.save('$RL_ROOT/adv_${ITER}.npy', adv)
print('[rl] 组利用率:', group_coverage_report(stats))
"

echo "=== [RL iter $ITER] ④ 策略更新 ==="
torchrun --nproc_per_node="${RL_GPUS:-8}" trl_sft/train_grpo.py \
    --model_path "$POLICY" \
    --data_path "$RL_ROOT/batch_${ITER}.parquet" \
    --advantages_file "$RL_ROOT/adv_${ITER}.npy" \
    --output_dir "$RL_ROOT/train_${ITER}" \
    --s3_output_dir "$S3_OUT/iter${ITER}" \
    --learning_rate "$LR"

echo "=== [RL iter $ITER] ⑤ 发布新权重（下一轮 ① 的 vLLM 将用它）==="
echo "POLICY=${S3_OUT}/iter${ITER}/final"
echo "=== iter $ITER 完成；bash rl_loop.sh 传入新 POLICY 开启 iter $((ITER+1)) ==="

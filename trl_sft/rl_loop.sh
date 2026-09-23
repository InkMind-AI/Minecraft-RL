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
# 状态（09-22）：阶段 1/2 完成（trainer + advantage 核心 + episode→parquet 适配器）；
# 新增 ReST-EM 自举模式（BOOTSTRAP=1，见下）：把 thought 当隐变量，让策略自己
# 生成候选、打分筛选、再拿来教自己，见 cot_annotation_schemes.md 的自举设计讨论。
#
# 用法（单轮手动验证各环节，标准 RL）：
#   bash rl_loop.sh --policy s3://.../rl_policy_iter0 --iter 1
# 用法（ReST-EM CoT 自举，MVP：信号2言行一致 + 信号3组内优势，跳过信号1真值
# verifier 和信号4局部进展塑形，留待后续版本）：
#   BOOTSTRAP=1 bash rl_loop.sh --policy s3://.../cot-pilot-v2/checkpoint-520 --iter 1
set -euo pipefail

POLICY="${POLICY:?需设置 POLICY=当前策略权重(S3)}"
ITER="${ITER:-0}"
RL_ROOT="${RL_ROOT:-/local-ssd/rl}"
GROUP_SIZE="${GROUP_SIZE:-8}"          # 每任务 rollout 数（GRPO 组大小）
# ⚠ 09-22 修复：原变量名 N_TASKS 从未被 run_backbone_eval.sh 读取——真正控制
# 任务数的变量是 EASY_NUM_TASKS（build_task_list.py --num_tasks，默认 300！）。
# 旧版此处设的 N_TASKS 是死代码，实际总是跑默认 300 个任务（bootstrap-verify
# 冒烟测试因此跑了远超预期的任务数，耗时暴涨——已实测复现并修复）。
N_TASKS="${N_TASKS:-50}"              # 每轮任务数（对外仍叫 N_TASKS，下面转译）
EVAL_BENCHMARK="${EVAL_BENCHMARK:-easy}"
LR="${RL_LR:-1e-6}"
S3_OUT="${S3_OUT:-s3://arcwm-code-us-west-2/axiom/model/minecraft-rl-policy}"
BOOTSTRAP="${BOOTSTRAP:-0}"            # 1 = ReST-EM CoT 自举模式

mkdir -p "$RL_ROOT"

echo "=== [RL iter $ITER] ① rollout（${N_TASKS}任务 × ${GROUP_SIZE}路，BOOTSTRAP=${BOOTSTRAP}）==="
# 复用现有评测管线：ROLLOUTS_PER_TASK=$GROUP_SIZE 即 GRPO 的组采样
export REPO_ROOT=/data/work/run_codes/Minecraft-CoT
export EVAL_BENCHMARK="$EVAL_BENCHMARK" ROLLOUTS_PER_TASK="$GROUP_SIZE"
export EASY_NUM_TASKS="$N_TASKS"      # 真正生效的任务数变量（见上方注释）
export MAXIMUM_HISTORY_LENGTH=29 LIMIT_MM_IMAGE=30
export MODEL_LOCAL_NAME="rl-iter${ITER}"
export MODEL_S3_URI="$POLICY"
export SERVED_MODEL_NAME="rl-policy-iter${ITER}"
export VLLM_CONDA_ENV=vllm35
if [ "$BOOTSTRAP" = "1" ]; then
    # 强制每步以 "Thought: " 续写（见 run_backbone_eval.sh），解耦"要不要想"和
    # "想得好不好"——决策点密度过滤放到②做，不依赖模型自发触发意愿。
    export FORCE_THOUGHT=1
fi
cd "$REPO_ROOT"
bash examples/eval_backbones/run_backbone_eval.sh || true   # 输出含 per-episode 成败 + raw_action

# ⚠ 09-23 修复（verify3 实测 OOM 后定位）：run_backbone_eval.sh 结尾的
# `kill "${VLLM_PID}"` 只杀了 `conda run` 包装进程的顶层 PID，vLLM 引擎真正的
# worker 子进程未必被连带杀死（尤其 TP>1 时是独立进程树），遗留进程继续占着
# 显存——④ train_grpo.py 起 DeepSpeed 时看到的是"126GB 已被某遗留进程占用"
# 而 OOM。硬清理：按名字 pkill 常见 vLLM 进程模式 + 兜底扫 nvidia-smi 的
# compute-apps 列表杀掉一切非本 shell 持有的 GPU 进程，再等显存真正回落。
echo "=== [RL iter $ITER] 清理 rollout 阶段遗留的 vLLM/GPU 进程 ==="
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::" 2>/dev/null || true
pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true
sleep 5
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r pid mem; do
    pid="$(echo "$pid" | xargs)"
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
done
sleep 5
echo "[cleanup] GPU 显存状态:"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo "=== [RL iter $ITER] ② episode → 训练样本 parquet ==="
# ⚠ 09-22 修复：真实目录结构是 eval_output/<MODEL_LOCAL_NAME>/<SERVED_MODEL_NAME>
# -text_action/（两层），不是 eval_output/<MODEL_LOCAL_NAME>-text_action/（一层）
# ——旧版路径拼接是从未被端到端跑过的死代码（verify2 实测 FileNotFoundError 后发现）。
BUILD_ARGS=(
    --eval-output "/local-ssd/eval_output/${MODEL_LOCAL_NAME}/${SERVED_MODEL_NAME}-text_action"
    --out "$RL_ROOT/batch_${ITER}.parquet"
    --rewards "$RL_ROOT/rewards_${ITER}.npy"
    --groups "$RL_ROOT/groups_${ITER}.json"
)
if [ "$BOOTSTRAP" = "1" ]; then
    BUILD_ARGS+=(--strip-non-decision-thought --thought-signal "$RL_ROOT/thought_sig_${ITER}.npy")
fi
python3 trl_sft/rl_build_batch.py "${BUILD_ARGS[@]}"

echo "=== [RL iter $ITER] ③ 优势（BOOTSTRAP=1 时叠乘信号2）==="
python3 -c "
import sys, json, numpy as np
sys.path.insert(0, 'trl_sft')
from grpo_core import compute_group_advantages, group_coverage_report
rewards = np.load('$RL_ROOT/rewards_${ITER}.npy')
groups = json.load(open('$RL_ROOT/groups_${ITER}.json'))
adv, stats = compute_group_advantages(rewards.tolist(), groups, mode='std')
print('[rl] 组利用率:', group_coverage_report(stats))

bootstrap = '$BOOTSTRAP' == '1'
if bootstrap:
    # 信号2（言行一致，rl_build_batch.py 算好的每窗口打分，中性值=1.0）与
    # 信号3（本 episode 的组内优势）相乘：thought 言行一致的窗口权重放大，
    # 前后矛盾的窗口权重收缩，无法判定的（中性1.0）不受影响。裁剪到
    # [0.5, 1.5] 防止极端值把某几个窗口的梯度贡献过度放大/抹平。
    sig = np.load('$RL_ROOT/thought_sig_${ITER}.npy')
    factor = np.clip(sig, 0.5, 1.5)
    final_adv = adv * factor
    print(f'[bootstrap] 信号2因子: mean={factor.mean():.3f} '
          f'min={factor.min():.3f} max={factor.max():.3f}')
else:
    final_adv = adv
np.save('$RL_ROOT/adv_${ITER}.npy', final_adv)
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
# ⚠ 09-23 修复：ITER 支持任意字符串（如冒烟测试用的 "verify4"），对它做
# $((ITER+1)) 算术展开在非纯数字场景会报 "unbound variable" 并让脚本在最后
#一行退出非零（verify4 实测：训练/权重发布都已成功，只是这行收尾提示语句
# 崩了）。只有 ITER 是纯数字时才提示"+1"的下一轮号，否则只提示手动指定。
if [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "=== iter $ITER 完成；bash rl_loop.sh 传入新 POLICY 开启 iter $((ITER+1)) ==="
else
    echo "=== iter $ITER 完成；bash rl_loop.sh 传入新 POLICY 并指定下一个 ITER 值继续 ==="
fi

#!/usr/bin/env bash
# 小规模正式训练（迁移第 4 步的第一轮）：在 smoke16 通过的配置上，把冒烟时刻意关掉的
# 几条路径全部打开，一次验完：
#   ① 学习信号：每局步数 16 → 200（与评测 MAX_STEPS_NUM=200 一致），组大小 2 → 4。
#      任务选 mine_block:oak_log——SFT 起点 v2-e4 在评测里 2/3 成功、成功局平均 153 帧，
#      200 步内组内大概率有成有败，GRPO 才有非零优势（smoke16 step2 pg_loss=0 就是
#      因为 16 步内全员 reward=0）
#   ② 验证路径：val_before_train + 末步验证（此前从未执行过）
#   ③ 存档路径：末步存 FSDP 分片 + HF 格式，HF 目录上传 S3，供评测脚本直接加载
#   ④ 规模放大后的显存与 Malmo 长时间稳定性
#
# 用法（koala，4 卡）：
#   bash rl_train/verl_jobs/run_grpo_minecraft_small.sh
# 产物：
#   s3://arcwm-code-us-west-2/axiom/model/minecraft-verl-grpo/$EXPERIMENT_NAME/global_step_N/
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

export N_GPUS="${N_GPUS:-4}"
# ⚠ TRAIN_BATCH 必须是 N_GPUS 的整数倍：ray_trainer._validate_config 断言的是
# data.train_batch_size × actor_rollout_ref.rollout.n（默认 1，不是 env.rollout.n）
# 能被卡数整除，所以 TRAIN_BATCH=2 在 4 卡上会在训练开始前直接断言失败。
export TRAIN_BATCH="${TRAIN_BATCH:-$N_GPUS}"   # 4 组 × 4 = 16 个训练环境 + 4 个验证环境
export GROUP_SIZE="${GROUP_SIZE:-4}"
export MAX_STEPS_ENV="${MAX_STEPS_ENV:-200}"
export TASKS="${TASKS:-mine_block:oak_log}"
export TOTAL_STEPS="${TOTAL_STEPS:-3}"
# 每步约 16 环境 × ~170 步 ≈ 2700 个样本；mini batch 64 → 每卡 64、全局 256 样本，
# 一次 rollout 约做 10 次优化器更新。
export PPO_MINI_BATCH="${PPO_MINI_BATCH:-64}"
export PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
export LOGPROB_MICRO_BATCH="${LOGPROB_MICRO_BATCH:-4}"  # 只做前向，显存开销小，提速
# 64 足够纯动作（"Action: move(0, 0) and press()" 约 15 token），但 v2 起点约 4% 的步
# 会先输出 Thought（中位 158 字符 ≈ 40-50 token），64 可能把动作截掉
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
export PROJECT_NAME="${PROJECT_NAME:-verl_minecraft}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_small_oaklog_$(date +%m%d_%H%M)}"
export SAVE_FREQ="${SAVE_FREQ:-$TOTAL_STEPS}"
export TEST_FREQ="${TEST_FREQ:-$TOTAL_STEPS}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export SAVE_HF="${SAVE_HF:-1}"
export CKPT_DIR="${CKPT_DIR:-/local-ssd/verl_ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
S3_OUT="${S3_OUT:-s3://arcwm-code-us-west-2/axiom/model/minecraft-verl-grpo/$EXPERIMENT_NAME}"

echo "[grpo-small] experiment=$EXPERIMENT_NAME gpus=$N_GPUS groups=$TRAIN_BATCH x $GROUP_SIZE" \
     "max_steps=$MAX_STEPS_ENV total_steps=$TOTAL_STEPS ckpt=$CKPT_DIR -> $S3_OUT"

set +e
bash rl_train/verl_jobs/run_grpo_minecraft_smoke.sh
TRAIN_RC=$?
set -e
echo "[grpo-small] 训练进程退出码 $TRAIN_RC"

# 只上传 HF 目录（可直接评测，约 18G）；FSDP 分片含优化器状态 100G+，不上传
UPLOADED=0
for HF in "$CKPT_DIR"/global_step_*/actor/huggingface; do
    [ -d "$HF" ] || continue
    STEP_DIR="$(basename "$(dirname "$(dirname "$HF")")")"
    echo "[grpo-small] 上传 $HF -> $S3_OUT/$STEP_DIR/"
    ls -la "$HF"
    aws s3 sync "$HF" "$S3_OUT/$STEP_DIR/" --only-show-errors
    UPLOADED=$((UPLOADED + 1))
done
echo "[grpo-small] 已上传 $UPLOADED 个 HF 存档"
if [ "$UPLOADED" -eq 0 ] && [ "$TRAIN_RC" -eq 0 ]; then
    echo "[grpo-small][WARN] 训练正常结束但没有找到 HF 存档，检查 checkpoint.contents / save_freq" >&2
fi
echo GRPO_SMALL_DONE
exit "$TRAIN_RC"

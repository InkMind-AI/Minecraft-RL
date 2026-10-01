#!/usr/bin/env bash
# verl GRPO 正式训练（Minecraft，h29 多轮上下文，对齐评测）。
#
# 与 run_grpo_minecraft_small.sh 的区别：
#   - HISTORY_LENGTH=29：训练 prompt 与评测 h29 逐字一致（30 张图 + 29 条历史回复）
#   - 任务池 = SFT 起点 v2-e4 在 easy-h29 评测里"有成有败"的 88 个任务
#     （tasks/mixed_v2e4_easy_h29.txt）：只有组内有成有败，GRPO 才有非零优势
#   - 每条轨迹只抽 TRAIN_STEPS_PER_TRAJ 步参与梯度（h29 每样本 ~155MB pixel_values）
#   - 训练过程中后台定期把 HF 存档和日志同步到 S3（容器挂了也不丢）
#
# 用法（koala，8 卡）：bash rl_train/verl_jobs/run_grpo_minecraft_train.sh
# 产物：s3://arcwm-code-us-west-2/axiom/model/minecraft-verl-grpo/$EXPERIMENT_NAME/
#         global_step_N/   ← HF 格式，可直接作 run_backbone_eval.sh 的 MODEL_S3_URI
#         train.log
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

export N_GPUS="${N_GPUS:-8}"
export TRAIN_BATCH="${TRAIN_BATCH:-$N_GPUS}"     # 任务组数；须为 N_GPUS 整数倍
export GROUP_SIZE="${GROUP_SIZE:-8}"             # 8 组 × 8 = 64 个训练环境
export VAL_BATCH="${VAL_BATCH:-16}"              # 16 个验证环境（每个随机任务）
export MAX_STEPS_ENV="${MAX_STEPS_ENV:-200}"     # = 评测 MAX_STEPS_NUM
export HISTORY_LENGTH="${HISTORY_LENGTH:-29}"    # = 评测 MAXIMUM_HISTORY_LENGTH
export TASKS_FILE="${TASKS_FILE:-$REPO_ROOT/rl_train/verl_jobs/tasks/mixed_v2e4_easy_h29.txt}"
export TRAIN_STEPS_PER_TRAJ="${TRAIN_STEPS_PER_TRAJ:-4}"   # 64 轨迹 × 4 = 256 样本/步
export PPO_MINI_BATCH="${PPO_MINI_BATCH:-64}"    # 256 样本 → 每步 4 次优化器更新
export PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
export LOGPROB_MICRO_BATCH="${LOGPROB_MICRO_BATCH:-2}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.8}" # = 评测 TEMPERATURE / TOP_P
export VAL_TOP_P="${VAL_TOP_P:-0.99}"
export TOTAL_STEPS="${TOTAL_STEPS:-100}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export SAVE_HF=1
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"  # 本地只留 2 份 FSDP 分片（每份 ~100G+）
export PROJECT_NAME="${PROJECT_NAME:-verl_minecraft}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_h29_mixed88_$(date +%m%d_%H%M)}"
export CKPT_DIR="${CKPT_DIR:-/local-ssd/verl_ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
S3_OUT="${S3_OUT:-s3://arcwm-code-us-west-2/axiom/model/minecraft-verl-grpo/$EXPERIMENT_NAME}"
UPLOAD_INTERVAL_S="${UPLOAD_INTERVAL_S:-600}"

mkdir -p "$CKPT_DIR"
LOG="$CKPT_DIR/train.log"
echo "[grpo-train] experiment=$EXPERIMENT_NAME gpus=$N_GPUS envs=${TRAIN_BATCH}x${GROUP_SIZE}" \
     "h=$HISTORY_LENGTH steps_per_traj=$TRAIN_STEPS_PER_TRAJ total_steps=$TOTAL_STEPS" \
     "tasks=$(grep -c . "$TASKS_FILE") ckpt=$CKPT_DIR -> $S3_OUT"

# 只上传已完整落盘的存档：trainer 存完一个 step 才会写 latest_checkpointed_iteration.txt
upload_ready_ckpts() {
    local latest
    latest="$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
    for HF in "$CKPT_DIR"/global_step_*/actor/huggingface; do
        [ -d "$HF" ] || continue
        local step_dir step
        step_dir="$(basename "$(dirname "$(dirname "$HF")")")"
        step="${step_dir#global_step_}"
        [ "$step" -le "$latest" ] || continue
        [ -f "$CKPT_DIR/.uploaded_$step" ] && continue
        echo "[grpo-train] 上传存档 $step_dir -> $S3_OUT/$step_dir/"
        if aws s3 sync "$HF" "$S3_OUT/$step_dir/" --only-show-errors; then
            touch "$CKPT_DIR/.uploaded_$step"
        fi
    done
    aws s3 cp "$LOG" "$S3_OUT/train.log" --only-show-errors 2>/dev/null || true
}

bash rl_train/verl_jobs/run_grpo_minecraft_smoke.sh > "$LOG" 2>&1 &
TRAIN_PID=$!
tail -n +1 -F "$LOG" 2>/dev/null &
TAIL_PID=$!

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "$UPLOAD_INTERVAL_S" &
    wait $! 2>/dev/null
    upload_ready_ckpts
done
wait "$TRAIN_PID"
TRAIN_RC=$?
sleep 5
kill "$TAIL_PID" 2>/dev/null || true
upload_ready_ckpts
echo "[grpo-train] 训练进程退出码 $TRAIN_RC；已上传：$(ls "$CKPT_DIR"/.uploaded_* 2>/dev/null | sed 's/.*_//' | tr '\n' ' ')"
echo GRPO_TRAIN_DONE
exit "$TRAIN_RC"

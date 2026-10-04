#!/usr/bin/env bash
# verl GRPO 正式训练（Minecraft，h29 多轮上下文，对齐评测）。
#
# 与 run_grpo_minecraft_small.sh 的区别：
#   - HISTORY_LENGTH=29：训练 prompt 与评测 h29 逐字一致（30 张图 + 29 条历史回复）
#   - 任务池 = SFT 起点 v2-e4 在 easy-h29 评测里"有成有败"的 88 个任务
#     （tasks/mixed_v2e4_easy_h29.txt）：只有组内有成有败，GRPO 才有非零优势
#   - 每条轨迹只抽 TRAIN_STEPS_PER_TRAJ 步参与梯度（h29 每样本 ~155MB pixel_values）
#   - 训练过程中后台定期把 HF 存档和日志同步到 S3（容器挂了也不丢）
#   - 10-04 卡死看门狗：train2/wandb 两个任务曾分别在 step 10/25 静默挂起 45h/19h
#     （无报错、无输出、koala 仍显示 Running）。现在超过 STALL_TIMEOUT_S 没有新的
#     `step:N` 行即判定卡死：先抓现场（py-spy 调用栈 / nvidia-smi / ps / Ray 日志）
#     传到 S3 的 hang_diag_*/，再杀掉整套进程，用同一 CKPT_DIR 重新拉起——verl 默认
#     trainer.resume_mode=auto，会从本地最新 FSDP 存档（含优化器状态）续跑，最多丢
#     SAVE_FREQ 步；连续 MAX_STALL_RESTARTS 次仍卡死则退出码 124 结束任务。
#   - VERL_ROLLOUT_KEEP_AWAKE=1：rollout 期间不再每个环境步全量同步一次权重（见
#     verl/workers/sharding_manager/fsdp_vllm.py 的说明），既是卡死的头号嫌疑点，
#     也是 rollout 慢的主要原因之一
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
# 正常一步 ~1h（含验证/存档的步 ~1.4h）；首步还要加模型加载 + 训练前验证，3h 留足余量
STALL_TIMEOUT_S="${STALL_TIMEOUT_S:-10800}"
MAX_STALL_RESTARTS="${MAX_STALL_RESTARTS:-3}"
export VERL_ROLLOUT_KEEP_AWAKE="${VERL_ROLLOUT_KEEP_AWAKE:-1}"
# 卡死时若 py-spy 抓不到栈，向 worker 发 SIGABRT，faulthandler 会把所有线程的
# Python 栈打进 Ray worker 的 stderr 日志（随 ray_logs.tgz 一起上传）
export PYTHONFAULTHANDLER=1
# 看门狗重启后续写同一个 wandb run，而不是每次重启新开一条曲线
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_RUN_ID="${WANDB_RUN_ID:-$(echo "$EXPERIMENT_NAME" | tr -cd 'a-zA-Z0-9' | tr 'A-Z' 'a-z')}"
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
fi

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

count_steps() {
    local n
    n="$(grep -c 'step:[0-9]* - ' "$LOG" 2>/dev/null)"
    echo "${n:-0}"
}

# 卡死现场：调用栈 / GPU / 进程 / Ray 日志，全部传 S3（容器回收后仍可查）
dump_hang_diag() {
    local D="$CKPT_DIR/hang_diag_$(date +%m%d_%H%M%S)"
    mkdir -p "$D"
    { date; echo "step 行数=$(count_steps) 无进展秒数=$1 restart=$RESTARTS"; } > "$D/info.txt"
    nvidia-smi > "$D/nvidia-smi.txt" 2>&1
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$D/nvidia-smi.txt" 2>&1
    ps -eo pid,ppid,stat,etime,pcpu,pmem,rss,args --sort=-pcpu 2>/dev/null | cut -c1-250 > "$D/ps.txt"
    if command -v py-spy >/dev/null 2>&1; then
        for pid in $(pgrep -f "ray::WorkerDict|ray::TaskRunner|verl.trainer.main_ppo"); do
            echo "===== pid $pid: $(ps -o args= -p "$pid" | cut -c1-160)"
            timeout 90 py-spy dump --native --pid "$pid" 2>&1 || timeout 60 py-spy dump --pid "$pid" 2>&1
        done > "$D/py-spy_trainer.txt"
        for pid in $(pgrep -f "ray::MinecraftWorker"); do
            echo "===== pid $pid"
            timeout 30 py-spy dump --pid "$pid" 2>&1
        done > "$D/py-spy_envs.txt"
    fi
    if ! grep -q "Thread" "$D/py-spy_trainer.txt" 2>/dev/null; then
        echo "[grpo-train][WATCHDOG] py-spy 未拿到栈，改用 SIGABRT + faulthandler" | tee -a "$D/info.txt"
        for pid in $(pgrep -f "ray::WorkerDict"); do kill -ABRT "$pid" 2>/dev/null; done
        sleep 15
    fi
    tar czhf "$D/ray_logs.tgz" -C /tmp/ray session_latest/logs 2>/dev/null
    tail -n 3000 "$LOG" > "$D/train_tail.log"
    aws s3 sync "$D" "$S3_OUT/$(basename "$D")/" --only-show-errors \
        && echo "[grpo-train][WATCHDOG] 现场已上传 $S3_OUT/$(basename "$D")/"
}

kill_training() {
    pkill -TERM -f "verl.trainer.main_ppo" 2>/dev/null
    sleep 10
    ray stop --force >/dev/null 2>&1
    pkill -KILL -f "verl.trainer.main_ppo" 2>/dev/null
    pkill -KILL -f "ray::" 2>/dev/null
    pkill -KILL java 2>/dev/null      # Malmo JVM
    pkill -KILL Xvfb 2>/dev/null
    kill -KILL "$TRAIN_PID" 2>/dev/null
    sleep 20
}

launch_training() {
    # 重启时不再做训练前验证（续跑的起点已经验证过）
    if [ "$RESTARTS" -gt 0 ]; then
        VAL_BEFORE_TRAIN=False bash rl_train/verl_jobs/run_grpo_minecraft_smoke.sh >> "$LOG" 2>&1 &
    else
        bash rl_train/verl_jobs/run_grpo_minecraft_smoke.sh > "$LOG" 2>&1 &
    fi
    TRAIN_PID=$!
}

RESTARTS=0
STALLED=0
launch_training
tail -n +1 -F "$LOG" 2>/dev/null &
TAIL_PID=$!
LAST_N=0
LAST_PROGRESS_T=$(date +%s)

while true; do
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep "$UPLOAD_INTERVAL_S" &
        wait $! 2>/dev/null
        upload_ready_ckpts
        N=$(count_steps); NOW=$(date +%s)
        if [ "$N" -gt "$LAST_N" ]; then
            LAST_N=$N; LAST_PROGRESS_T=$NOW
        elif [ $((NOW - LAST_PROGRESS_T)) -ge "$STALL_TIMEOUT_S" ]; then
            echo "[grpo-train][WATCHDOG] $((NOW - LAST_PROGRESS_T))s 无新 step（阈值 ${STALL_TIMEOUT_S}s），判定卡死"
            STALLED=1
            break
        fi
    done
    [ "$STALLED" = "1" ] || break
    dump_hang_diag $((NOW - LAST_PROGRESS_T))
    kill_training
    upload_ready_ckpts
    if [ "$RESTARTS" -ge "$MAX_STALL_RESTARTS" ]; then
        echo "[grpo-train][WATCHDOG] 已重启 $RESTARTS 次仍卡死，放弃"
        kill "$TAIL_PID" 2>/dev/null || true
        echo GRPO_TRAIN_STALLED
        exit 124
    fi
    RESTARTS=$((RESTARTS + 1))
    echo "[grpo-train][WATCHDOG] 第 $RESTARTS 次重启：从 $CKPT_DIR 最新存档续跑（resume_mode=auto）"
    STALLED=0
    LAST_PROGRESS_T=$(date +%s)
    launch_training
done
wait "$TRAIN_PID"
TRAIN_RC=$?
sleep 5
kill "$TAIL_PID" 2>/dev/null || true
upload_ready_ckpts
echo "[grpo-train] 训练进程退出码 $TRAIN_RC；看门狗重启 $RESTARTS 次；已上传：$(ls "$CKPT_DIR"/.uploaded_* 2>/dev/null | sed 's/.*_//' | tr '\n' ' ')"
echo GRPO_TRAIN_DONE
exit "$TRAIN_RC"

#!/usr/bin/env bash
# GRPO on Minecraft，冒烟规模（迁移第 3 步验收）：1 组 x group_size=4，
# 只求跑通 rollout -> advantage -> update 至少 2 步，不求成绩。
#
# 关键对齐点（否则会重演 iter1 的分布偏移）：
#   - apply_chat_template_kwargs.enable_thinking=false —— 与 SFT/评测的
#     _CHAT_TEMPLATE_KWARGS 完全一致（chat_template.jinja 里 enable_thinking=false
#     时才会吐出 "<think>\n\n</think>\n\n" 空推理块）。用 `+` 前缀（而非
#     `data.apply_chat_template_kwargs={...}` 整体替换）：Hydra struct 模式下，
#     即使是给一个空字典 `{}` 赋新的字面量值，也会按"合并"语义逐键校验，
#     `enable_thinking` 这个键不在原 schema 里就直接拒绝（"Key ... is not in
#     struct"）——09-28 grpo-smoke1 实测踩到，`+` 前缀是 Hydra 自己在报错里
#     建议的写法：显式声明"这是新增键"而非覆盖已有键。
#   - env.minecraft.system_message_tag=text_action —— 与评测 SYSTEM_MESSAGE_TAG 一致
#   - actor_rollout_ref.model.path 用 continue-SFT 起点（v2-e4），不是原始基座
#   - enable_chunked_prefill=True 是**必需项**，不是性能选项——09-29 smoke6 实测：
#     Qwen3.5 的 fla 混合注意力（mamba/gated-delta-net cache）在当前 vLLM 版本下
#     要求 cache mode 'align'，而这要求 chunked prefill 开启，False 会在
#     VllmConfig 校验阶段直接 pydantic ValidationError（"Chunked prefill is
#     required for mamba cache mode 'align'"），根本起不来 engine。
#
# 已知限制（见 MinecraftEnvironmentManager 文档串）：每样本仅 1 张图，非 h29 多图
# 历史——冒烟阶段可接受，度量吞吐/验证训练循环不受影响；真实性能评测需另行处理。
set -x
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/local-ssd/model_cache}"
DATA_DIR="${DATA_DIR:-/local-ssd/verl_data}"
GROUP_SIZE="${GROUP_SIZE:-2}"   # GRPO 组内相对优势至少需要 2；冒烟取最小值以减少 Malmo 实例数
N_GPUS="${N_GPUS:-4}"
# ⚠ 10-01 smoke12/14：9B 全参数 Adam 的 optimizer.step() 峰值约 144G（fp32 主参数
# 36G + 梯度 36G + exp_avg/exp_avg_sq 72G），**单卡物理上放不下**。而 verl 的
# param_offload/optimizer_offload 不是 CPU 计算：update_actor 开头会
# load_fsdp_model_to_gpu + load_fsdp_optimizer 全部搬回 GPU 再 step（见
# fsdp_workers.py update_actor），只能降低 rollout 阶段的占用，降不了 step 峰值
# ——smoke14 开了 offload 仍在 Adam _init_group OOM 即为实证。唯一有效手段是
# FSDP 多卡分片：4 卡时每卡约 36G + vLLM 份额，可放下。
if [ "$N_GPUS" -lt 4 ]; then
    echo "[grpo-smoke][WARN] N_GPUS=$N_GPUS < 4：9B 全参数 Adam 的 step 峰值大概率 OOM" \
         "（offload 不降低 step 峰值，见上方注释）" >&2
fi
# ⚠ 09-29：verl 断言 `real_train_batch_size % n_gpus == 0`（rollout batch 要能
# 均分到各 GPU）。TRAIN_BATCH（任务组数）默认与 N_GPUS 对齐，而非固定 1——
# grpo-smoke3 实测 TRAIN_BATCH=1 + N_GPUS=2 直接在训练循环起步前断言失败退出。
TRAIN_BATCH="${TRAIN_BATCH:-$N_GPUS}"  # = 任务组数；总 worker 数 = TRAIN_BATCH*GROUP_SIZE
MAX_STEPS_ENV="${MAX_STEPS_ENV:-16}"   # 冒烟用短 episode，缩短 rollout 耗时
# ⚠ 10-01 smoke15：占位 parquet 只有 TRAIN_BATCH 行 = 恰好 1 个 batch，
# total_epochs=1 时 dataloader 一轮就耗尽，fit() 在第 1 步后**正常退出**（无报错，
# Progress 停在 1/2）——total_training_steps 只截断、不补数据。真实 prompt/图像都
# 来自环境，占位行只决定"每步几组"，所以让 epoch 数 = 目标步数即可（每 epoch 1 步）。
TOTAL_STEPS="${TOTAL_STEPS:-2}"
TASKS="${TASKS:-mine_block:oak_log}"
# ⚠ 10-01 grpo-smoke12：rollout→reward→advantage→log_prob→backward 全部走通，
# 却在 actor optimizer.step()（Adam 首次 _init_group）OOM：9B 全参数 Adam 在
# FSDP 混合精度下需 fp32 主参数 36G + 梯度 36G + exp_avg/exp_avg_sq 72G ≈ 144G，
# 单卡 140G 本来就放不下，何况 vLLM 还占着 gpu_memory_utilization 那一份。
# 卡数少于 4 时仍默认开 offload（可降低 rollout 阶段显存占用），但它**解决不了
# step 峰值**（见 N_GPUS 处注释），<4 卡基本跑不通；≥4 卡 FSDP 分片后默认不卸载。
if [ "$N_GPUS" -lt 4 ]; then _OFFLOAD_DEFAULT=True; else _OFFLOAD_DEFAULT=False; fi
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-$_OFFLOAD_DEFAULT}"
ACTOR_OPTIM_OFFLOAD="${ACTOR_OPTIM_OFFLOAD:-$_OFFLOAD_DEFAULT}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.4}"

# ─── 10-01：从冒烟走向小规模正式训练所需的可调项（默认值 = smoke16 通过时的配置）───
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-64}"
# ppo_mini_batch_size 在 fsdp_workers 里会先 ×rollout.n 再 ÷world_size 得到"每卡每次
# 优化器更新的样本数"。注意 verl-agent 的样本是**逐环境步**展开的（一条 200 步轨迹 =
# 200 个样本），所以一次 rollout 会切成很多个 mini batch、做多次优化器更新。
# 默认 TRAIN_BATCH*GROUP_SIZE，与 smoke16 逐字一致。
PPO_MINI_BATCH="${PPO_MINI_BATCH:-$((TRAIN_BATCH * GROUP_SIZE))}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-1}"
LOGPROB_MICRO_BATCH="${LOGPROB_MICRO_BATCH:-1}"
PROJECT_NAME="${PROJECT_NAME:-verl_minecraft_smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_5_9b_smoke}"
# 10-02：wandb 可选接入。只要 WANDB_API_KEY 非空就自动把 wandb 加进
# trainer.logger（与 console 并存，不互斥），否则保持原来纯 console 行为——
# 不传 key 的旧调用方式（冒烟脚本的全部历史调用）不受影响。project/name 直接
# 复用下面已有的 trainer.project_name/experiment_name（tracking.py 的
# WandbLogger 就是拿这两个做 wandb.init(project=, name=)，不需要额外变量）。
WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
    LOGGER_BACKENDS="['console','wandb']"
else
    LOGGER_BACKENDS="['console']"
fi
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# 存档：vendored 版 FSDPCheckpointManager.save_checkpoint **无条件**写每个 rank 的
# model/optim/extra 分片（fsdp_checkpoint_manager.py:162-185 不看 contents），9B 约
# 100G+/次，且评测脚本加载不了分片。SAVE_HF=1 时额外追加 'hf_model'：rank0 汇总成
# 标准 HF 目录 <CKPT_DIR>/global_step_N/actor/huggingface/，可直接给
# run_backbone_eval.sh 当 MODEL_S3_URI 用。默认落在 /local-ssd（--large-ssd 节点盘
# 28T），容器结束即回收，需要保留的由调用方自行上传 S3。
CKPT_DIR="${CKPT_DIR:-/local-ssd/verl_ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
if [ "${SAVE_HF:-0}" = "1" ]; then
    _CKPT_CONTENTS="['model','optimizer','extra','hf_model']"
else
    _CKPT_CONTENTS="['model','optimizer','extra']"
fi
# ⚠ 不能设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True：vLLM 的 sleep-mode
# 显存池（free_cache_engine=True 依赖的 CuMemAllocator）直接断言二者不兼容
# （smoke13 实测 AssertionError，pytorch#147851）。显存余量靠上面的 offload 解决。

# ─── 10-01：h29 多轮上下文（对齐评测 MAXIMUM_HISTORY_LENGTH=29 / LIMIT_MM_IMAGE=30）───
# HISTORY_LENGTH=0 保持 smoke16 的单帧 prompt；>0 时 MinecraftEnvironmentManager 按
# openha.gen_response 逐字构造多轮对话（本地已验证与评测列表形式渲染逐字一致）。
# h29 实测 prompt 约 7.9k token（30 图 × 220 + system ~500 + 29 条回复），上限留到 10240；
# 偶发超长时 rollout 从最旧历史开始丢弃，不会中断训练。
HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
if [ "$HISTORY_LENGTH" -gt 0 ]; then
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-10240}"
    LIMIT_IMAGES="${LIMIT_IMAGES:-$((HISTORY_LENGTH + 1))}"
else
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
    LIMIT_IMAGES="${LIMIT_IMAGES:-1}"
fi
# vLLM 开 chunked prefill 时要求 max_num_batched_tokens >= max_model_len(=prompt+response)
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$(( (MAX_PROMPT_LEN + MAX_RESPONSE_LEN) > 8192 ? 16384 : 8192 ))}"
# 每条轨迹参与梯度的环境步数（0=全部）。h29 下每个样本带 30 张图约 155MB pixel_values，
# 必须抽样；见 rollout_loop.gather_rollout_data。
TRAIN_STEPS_PER_TRAJ="${TRAIN_STEPS_PER_TRAJ:-0}"
VAL_BATCH="${VAL_BATCH:-$TRAIN_BATCH}"
TASKS_FILE="${TASKS_FILE:-}"
if [ -n "$TASKS_FILE" ]; then _TASK_ARGS=("env.minecraft.tasks_file=$TASKS_FILE"); else _TASK_ARGS=("env.minecraft.tasks=$TASKS"); fi
# 验证采样与评测 harness 一致（run_backbone_eval.sh: TEMPERATURE=0.8 TOP_P=0.99）
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-null}"
# ⚠ 10-01 train1：h29 下 vLLM 0.17 的多模态处理器缓存（前端 sender ↔ 引擎 receiver
# 两份按 mm_hash 对齐的 LRU，默认 4GB）在每个请求带 30 张图、且相邻请求大量复用同一批
# 历史帧时会失步——前端以为某帧已缓存只发 hash，引擎侧已把它淘汰，于是
# `AssertionError: Expected a cached item for mm_hash=...`，在训练前验证的第一批生成就
# 炸了。我们的 rollout 是离线 LLM.generate、不存在跨进程重复上传的问题，缓存收益很小，
# 直接设 0 关掉（每次都完整传图，彻底消除失步）。单帧模式不受影响，也一并关掉。
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-0}"
# 10-07：三次 rollout 卡死的现场都是单卡 vLLM 前向 kernel 永不结束，prefix caching（Qwen3.5
# 混合注意力会因此进入 mamba cache 'align' 模式）是头号嫌疑。设 False 做对照。
# 默认 True，与此前所有任务一致
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-True}"

python rl_train/verl_jobs/prepare_minecraft_data.py --out "$DATA_DIR" \
  --train "$TRAIN_BATCH" --val "$VAL_BATCH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH \
    data.val_batch_size=$VAL_BATCH \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIM_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    +actor_rollout_ref.rollout.limit_images=$LIMIT_IMAGES \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=$MM_PROCESSOR_CACHE_GB \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=$ENABLE_PREFIX_CACHING \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Minecraft \
    env.seed=0 \
    env.max_steps=$MAX_STEPS_ENV \
    env.history_length=$HISTORY_LENGTH \
    env.rollout.train_steps_per_traj=$TRAIN_STEPS_PER_TRAJ \
    env.rollout.n=$GROUP_SIZE \
    "${_TASK_ARGS[@]}" \
    env.minecraft.difficulty=easy \
    env.minecraft.system_message_tag=text_action \
    env.resources_per_worker.num_cpus=1 \
    trainer.critic_warmup=0 \
    trainer.logger=$LOGGER_BACKENDS \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_TO_KEEP \
    trainer.default_local_dir="$CKPT_DIR" \
    "actor_rollout_ref.actor.checkpoint.contents=$_CKPT_CONTENTS" \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_STEPS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN "$@"

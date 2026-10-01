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
# ⚠ 不能设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True：vLLM 的 sleep-mode
# 显存池（free_cache_engine=True 依赖的 CuMemAllocator）直接断言二者不兼容
# （smoke13 实测 AssertionError，pytorch#147851）。显存余量靠上面的 offload 解决。

python rl_train/verl_jobs/prepare_minecraft_data.py --out "$DATA_DIR" \
  --train "$TRAIN_BATCH" --val "$TRAIN_BATCH"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH \
    data.val_batch_size=$TRAIN_BATCH \
    data.max_prompt_length=1024 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH * GROUP_SIZE)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIM_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Minecraft \
    env.seed=0 \
    env.max_steps=$MAX_STEPS_ENV \
    env.history_length=0 \
    env.rollout.n=$GROUP_SIZE \
    env.minecraft.tasks="$TASKS" \
    env.minecraft.difficulty=easy \
    env.minecraft.system_message_tag=text_action \
    env.resources_per_worker.num_cpus=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_minecraft_smoke' \
    trainer.experiment_name='grpo_qwen3_5_9b_smoke' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_STEPS \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.val_before_train=False $@

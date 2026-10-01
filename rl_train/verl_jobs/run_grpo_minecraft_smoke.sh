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
GROUP_SIZE="${GROUP_SIZE:-4}"
N_GPUS="${N_GPUS:-2}"
# ⚠ 09-29：verl 断言 `real_train_batch_size % n_gpus == 0`（rollout batch 要能
# 均分到各 GPU）。TRAIN_BATCH（任务组数）默认与 N_GPUS 对齐，而非固定 1——
# grpo-smoke3 实测 TRAIN_BATCH=1 + N_GPUS=2 直接在训练循环起步前断言失败退出。
TRAIN_BATCH="${TRAIN_BATCH:-$N_GPUS}"  # = 任务组数；总 worker 数 = TRAIN_BATCH*GROUP_SIZE
MAX_STEPS_ENV="${MAX_STEPS_ENV:-16}"   # 冒烟用短 episode，缩短 rollout 耗时
TASKS="${TASKS:-mine_block:oak_log}"
# ⚠ 10-01 grpo-smoke12：rollout→reward→advantage→log_prob→backward 全部走通，
# 却在 actor optimizer.step()（Adam 首次 _init_group）OOM：9B 全参数 Adam 在
# FSDP 混合精度下需 fp32 主参数 36G + 梯度 36G + exp_avg/exp_avg_sq 72G ≈ 144G，
# 单卡 140G 本来就放不下，何况 vLLM 还占着 gpu_memory_utilization 那一份。
# 卡数少于 4 时默认把 actor 参数+优化器状态卸到 CPU（koala 每卡 220G 内存，够放），
# 慢一些但冒烟只求跑通；≥4 卡时 FSDP 分片后显存够用，默认不卸载。
if [ "$N_GPUS" -lt 4 ]; then _OFFLOAD_DEFAULT=True; else _OFFLOAD_DEFAULT=False; fi
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-$_OFFLOAD_DEFAULT}"
ACTOR_OPTIM_OFFLOAD="${ACTOR_OPTIM_OFFLOAD:-$_OFFLOAD_DEFAULT}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.val_before_train=False $@

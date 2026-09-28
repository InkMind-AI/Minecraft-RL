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
#
# 已知限制（见 MinecraftEnvironmentManager 文档串）：每样本仅 1 张图，非 h29 多图
# 历史——冒烟阶段可接受，度量吞吐/验证训练循环不受影响；真实性能评测需另行处理。
set -x
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/local-ssd/model_cache}"
DATA_DIR="${DATA_DIR:-/local-ssd/verl_data}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TRAIN_BATCH="${TRAIN_BATCH:-1}"        # = 任务组数；总 worker 数 = TRAIN_BATCH*GROUP_SIZE
MAX_STEPS_ENV="${MAX_STEPS_ENV:-16}"   # 冒烟用短 episode，缩短 rollout 耗时
N_GPUS="${N_GPUS:-2}"
TASKS="${TASKS:-mine_block:oak_log}"

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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
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

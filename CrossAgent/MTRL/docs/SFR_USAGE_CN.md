# Simple Failure Ranking（SFR）使用说明

本文档说明 Minecraft-RL 中 Simple Failure Ranking（SFR）的实现、数据流、配置、测试和运行方式。SFR 是 trajectory-level 的失败轨迹排序方法：一条完整 episode 结束后只产生一个标量，不产生 step-wise reward。实现支持纯规则 q 和可选的 model-based completion judge；model judge 默认关闭。

## 1. 解决的问题

Minecraft 的早期探索阶段经常出现同一个 GRPO group 内所有 episode 都失败的情况。如果只使用最终 success/fail reward，那么该 group 内所有 episode 的 return 都可能为 0，GRPO 没有相对排序依据，学习信号会退化。

SFR 的行为如下：

1. 只在同一个 GRPO group 全部失败时排序。
2. 根据完整失败轨迹中的任务进度和执行质量计算 q(tau)。
3. 在 group 内对 q(tau) 做 mid-rank，并映射为小幅度的相对 reward。
4. 没有可靠差异证据时 abstain，保持原始 reward。
5. group 中有成功轨迹，或出现 infra error 时，保持原始 reward。

除了规则模式外，`model_rank` 模式会把每条失败轨迹的终止状态摘要发送给一个 OpenAI-compatible chat-completions 服务，由模型判断任务完成度，再对模型输出的 completion 分数做 group 内排序。模型 judge 只判断任务完成情况，不判断语言表达质量。

算法形式：

    q(tau) = weighted trajectory-level quality
    r_sfr(tau_i) = beta * (2 * rank(q_i) / (K - 1) - 1)

K 是 group size。默认 beta=0.2，K=4 且没有并列时，rank reward 为：

    -0.2, -0.0667, +0.0667, +0.2

该 reward 只替代完整 episode 的 outcome reward，不会被分发到环境的每个 step。

## 2. 代码位置和训练数据流

核心算法位于：

    agent_system/multi_turn_rollout/failure_ranking.py

核心接口：

    relabel_episode_rewards(
        trajectory_steps,
        episode_rewards,
        episode_lengths,
        success,
        max_steps,
        trajectory_infos=None,
        mode="sparse",
        beta=0.2,
        quality_epsilon=0.02,
        random_seed=0,
        feature_weights=None,
        model_config=None,
    )

返回 new_episode_rewards 和 details。details 包含 q、rank、abstain 标志及日志指标。

rollout_loop.py 在 vanilla_multi_turn_loop 和 async_dynamic_multi_turn_loop 两条路径中都在 episode 完成后调用 SFR。完整顺序是：

    环境 rollout
        -> envs.success_evaluator(...)
        -> _apply_failure_ranking(...)
        -> dynamic filter（如果启用）
        -> gather_rollout_data
        -> EpisodeRewardManager
        -> token_level_scores
        -> GRPO advantage

gather_rollout_data 会把以下元数据写入有效 trajectory step：

    raw_episode_reward   原始环境 episode reward
    sfr_reward           SFR 改写后的 episode reward
    sfr_quality          q(tau)
    sfr_rank             归一化 group rank；未排序时为 -1
    sfr_abstain          是否 abstain
    sfr_all_failed       所在 group 是否 all-failed

现有 agent_system/reward_manager/episode.py 会把 episode_rewards 放到 response 最后一个有效 token。因此 SFR 没有改动 PPO/GRPO 的 token-level reward 接口。

### 2.3 model-based completion judge

model-based 模式不训练新的 reward model，而是在 episode 结束时调用外部或本地部署的 LLM judge。当前实现要求服务提供 OpenAI-compatible API：

    POST <endpoint>/chat/completions

其中 endpoint 可以填写 `http://127.0.0.1:8000/v1`，代码会自动补上 `/chat/completions`。请求包含任务名、任务描述、初始和终止背包、累计事件、health/food、最近状态和动作统计。不会发送完整视频，也不会把 response token id 当作任务状态。

模型必须返回 JSON，例如：

    {"completion": 0.63, "confidence": 0.91, "reason": "The target item was obtained but the final condition is incomplete."}

`completion` 会裁剪到 `[0, 1]`，表示任务完成度而非成功概率。全失败 group 中的 completion 分数随后使用与规则 SFR 相同的 mid-rank reward。`confidence_threshold` 可以过滤低置信度判断；judge 调用失败时，`fallback=abstain` 保持原始 reward，`fallback=rule` 使用规则 q 排序。

支持的模型类型包括 GPT 类闭源 API 和通过 vLLM/TGI 等暴露 OpenAI-compatible endpoint 的 Qwen 类本地模型。API key 只从 `api_key_env` 指定的环境变量读取，不写入 YAML、命令行日志或 trajectory metadata。

## 3. q(tau) 质量分数

默认权重：

    q = 0.30 * event
      + 0.25 * inventory
      + 0.15 * validity
      + 0.15 * survival
      - 0.05 * noop
      - 0.05 * loop
      - 0.10 * death
      - 0.05 * timeout

实现会按照实际可用字段重新归一化权重，避免某个字段缺失导致整个 q 被无意义缩小。

### 3.1 任务事件进度

从每一步 info 中读取常见事件：

    pickup
    break_item
    craft_item
    mine_block
    kill_entity
    use_item
    drop
    entity_killed_by
    custom

任务名按照 task_type:target 解析，例如：

    mine_block:oak_log
    craft_item:wooden_pickaxe
    smelt_item:iron_ingot

任务相关事件优先使用 task_type:target 和 pickup:target。smelt_item 额外允许 craft_item:target。

累计进度 x 使用有界函数：

    bounded_progress(x) = x / (1 + x)

这样会把进度限制在 [0, 1)，但不会把获得 1 个和获得 2 个目标物品截断为同一个分数。

### 3.2 背包进度

兼容下列常见结构：

    {"oak_log": 3}
    [{"type": "oak_log", "quantity": 3}]
    [("oak_log", 3)]

背包目标增量为：

    max(final_target_count - initial_target_count, 0)

随后经过 x / (1+x) 变换。若环境提供 initial_inventory，优先使用；否则使用 rollout 中第一个 inventory 作为初始参考。

### 3.3 动作有效性

动作有效性优先读取 trajectory step 的 is_action_valid，缺失时回退到 info 的 is_action_valid。该特征是整条轨迹的有效动作比例。

同步路径在 EnvironmentManagerBase.step() 中记录 projection 返回的 valids；异步 Minecraft 路径在 MinecraftEnvironmentManager.step_one() 中也记录同一 projection 结果。

### 3.4 no-op 和 loop

Minecraft worker 在执行 raw action 后记录 action_is_noop。当所有按钮没有按下且 camera motion 为 0 时视为 no-op。

如果环境显式提供 action_is_loop 或 is_loop，优先使用显式字段；否则根据 location_stats 或 player_pos 的位置、视角和 GUI 状态检测重复窗口。当前检测使用 8 帧窗口：状态不超过 2 个不同值且窗口没有事件时，计入重复无效窗口。

### 3.5 survival、death、timeout

- survival：根据最终 health 和 food_level 计算，默认 20 点为满血和满饥。
- death：任一步 death_detected、dead、health 小于等于 0 或 respawn_detected 为真。
- timeout：episode_length >= env.max_steps。

死亡和超时是负项。没有 health/food 时 survival 不参与 q；death 和 timeout 仍可根据显式字段及 episode length 判断。

## 4. 保护逻辑和 group 一致性

group 由 rollout 中已有的 uid 定义，env.rollout.n 是 group size。逻辑如下：

    if any(success_i > 0 for i in group):
        keep_original_rewards()
    elif any(infra_error_i for i in group):
        keep_original_rewards()
    elif mode == "sfr" and no_reliable_quality_difference:
        keep_original_rewards()
    else:
        assign_group_relative_rank_reward()

infra error 不能被当成“更差的策略”参与训练排序。

为了使 group-relative ranking 有意义，MinecraftMultiProcessEnv.reset_workers() 每个 group 只生成一次 task config，并把同一配置的深拷贝传给 group 内所有 worker。每个 info 还记录 task_config_hash，可用来检查同组任务、seed、出生位置和初始配置是否一致。reset_single() 也优先复用当前 group 配置。

## 5. 配置

配置文件：

    verl/trainer/config/ppo_trainer.yaml

默认配置：

    algorithm:
      sfr:
        enabled: False
        mode: sfr
        beta: 0.2
        quality_epsilon: 0.02
        random_seed: 0
        feature_weights:
          event: 0.30
          inventory: 0.25
          validity: 0.15
          survival: 0.15
          noop: 0.05
          loop: 0.05
          death: 0.10
          timeout: 0.05
        model:
          endpoint: null
          model_name: null
          api_key_env: OPENAI_API_KEY
          timeout_seconds: 30
          max_tokens: 128
          max_prompt_chars: 12000
          temperature: 0.0
          confidence_threshold: 0.0
          fallback: abstain

参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| algorithm.sfr.enabled | False | 是否启用 SFR。关闭时保持原始 reward。 |
| algorithm.sfr.mode | sfr | sparse、sfr、random_rank、absolute_q、length_rank 或 model_rank。 |
| algorithm.sfr.beta | 0.2 | rank reward 的最大绝对幅度。建议先用 0.1 或 0.2。 |
| algorithm.sfr.quality_epsilon | 0.02 | q 差异小于该值时视为同一质量等级。 |
| algorithm.sfr.random_seed | 0 | random_rank 对照实验的随机种子。 |
| algorithm.sfr.feature_weights.* | 见 YAML | 各 trajectory-level 特征的权重。 |
| algorithm.sfr.model.endpoint | null | OpenAI-compatible 服务的 base endpoint，仅 model_rank 使用。 |
| algorithm.sfr.model.model_name | null | 服务端模型名。 |
| algorithm.sfr.model.api_key_env | OPENAI_API_KEY | 读取 API key 的环境变量名。 |
| algorithm.sfr.model.timeout_seconds | 30 | 单条轨迹 judge 请求超时时间。 |
| algorithm.sfr.model.max_prompt_chars | 12000 | 发送给 judge 的摘要最大字符数。 |
| algorithm.sfr.model.confidence_threshold | 0.0 | 低于该置信度的判断视为失败。 |
| algorithm.sfr.model.fallback | abstain | judge 失败时使用 abstain 或 rule。 |

各 mode：

| mode | 用途 | 行为 |
| --- | --- | --- |
| sparse | 原始 baseline | 不改 reward，只添加诊断元数据。 |
| sfr | 主方法 | 使用规则 q、all-failed gate 和 abstention。 |
| random_rank | 随机 sanity check | all-failed group 内使用固定随机数排序。 |
| absolute_q | 绝对分数对照 | 按 q 的 min-max 结果给 [0, beta] reward。 |
| length_rank | 长度对照 | 仅按 episode length 排序。 |
| model_rank | model-based 对照 | 使用 LLM judge 的 completion 分数在 all-failed group 内排序。 |

`model` 是 `model_rank` 的别名。model-based 模式必须显式设置 `SFR_ENABLED=True`，不会因为配置了 endpoint 而自动发送请求。

## 6. 测试和启动

在 CrossAgent/MTRL 目录运行 CPU 单元测试：

    python -m pytest tests/test_failure_ranking.py -q

静态编译检查：

    python -m py_compile \
      agent_system/multi_turn_rollout/failure_ranking.py \
      agent_system/multi_turn_rollout/rollout_loop.py \
      agent_system/environments/env_manager.py \
      agent_system/environments/env_package/minecraft/envs.py \
      verl/trainer/ppo/ray_trainer.py

启动脚本语法检查：

    bash -n examples/grpo_trainer/run_minecraft.sh

原始 sparse baseline：

    N_GPUS=2 GROUP_SIZE=4 SFR_ENABLED=False \
      bash examples/grpo_trainer/run_minecraft.sh vllm

启用 SFR：

    N_GPUS=2 \
    GROUP_SIZE=4 \
    SFR_ENABLED=True \
    SFR_MODE=sfr \
    SFR_BETA=0.2 \
    SFR_EPSILON=0.02 \
    SFR_SEED=0 \
    bash examples/grpo_trainer/run_minecraft.sh vllm

运行对照：

    SFR_ENABLED=True SFR_MODE=random_rank bash examples/grpo_trainer/run_minecraft.sh vllm
    SFR_ENABLED=True SFR_MODE=absolute_q bash examples/grpo_trainer/run_minecraft.sh vllm
    SFR_ENABLED=True SFR_MODE=length_rank bash examples/grpo_trainer/run_minecraft.sh vllm

model-based completion judge：

    export SFR_MODEL_ENDPOINT=http://127.0.0.1:8000/v1
    export SFR_MODEL_NAME=Qwen/Qwen3.5-9B
    export SFR_MODEL_FALLBACK=abstain
    SFR_ENABLED=True SFR_MODE=model_rank \
      bash examples/grpo_trainer/run_minecraft.sh vllm

如果 endpoint 需要鉴权，例如：

    export OPENAI_API_KEY=...
    export SFR_MODEL_ENDPOINT=https://api.openai.com/v1
    export SFR_MODEL_NAME=<model-name>

不要把真实 key 写进 shell 脚本或提交到仓库。

启动脚本保留末尾的 $@，也可使用 Hydra override：

    bash examples/grpo_trainer/run_minecraft.sh vllm \
      algorithm.sfr.enabled=True \
      algorithm.sfr.beta=0.1 \
      algorithm.sfr.quality_epsilon=0.01

启动脚本不会对 `N_GPUS` 做代码层面的上限限制，`trainer.n_gpus_per_node` 直接使用 `N_GPUS`。实际使用多少张 GPU 由运行者和集群资源配置决定。

Minecraft worker 的 reset 默认保留原有的随机 `0-180` 秒错峰等待。为了进行快速环境验证，可以设置：

    MINECRAFT_RESET_MAX_DELAY=0

该变量只控制 reset 前的等待时间，不会改变环境状态、任务配置或 SFR 算法；不设置时默认为 `180`，正式训练可保持默认行为。

## 7. 日志检查

SFR 开启后，训练日志应出现：

    sfr/sfr_total_trajectories
    sfr/sfr_all_failed_groups
    sfr/sfr_ranked_groups
    sfr/sfr_abstain_groups
    sfr/sfr_rank_coverage
    sfr/sfr_mean_quality
    sfr/sfr_quality_std
    sfr/sfr_reward_mean
    sfr/sfr_reward_std
    sfr/sfr_infra_error_groups
    sfr/sfr_model_calls
    sfr/sfr_model_successes
    sfr/sfr_model_errors
    sfr/sfr_model_fallbacks
    sfr/sfr_model_mean_score
    sfr/sfr_model_score_std

解释：

1. sfr_all_failed_groups 为 0：当前 batch 没有全失败 group，SFR 不会介入。
2. sfr_ranked_groups 为 0 且 abstain 很高：info 中没有足够任务进度差异。
3. sfr_infra_error_groups 异常升高：应先排查环境或模型基础设施。
4. sfr_rank_coverage：ranked all-failed groups / all-failed groups。
5. 对照 raw_episode_reward 和 sfr_reward，确认只有符合条件的 group 被改写。
6. 检查同一 uid 内 task_config_hash 是否一致。
7. model_rank 模式检查 `sfr_model_successes / sfr_model_calls`，以及 `sfr_model_errors` 和 `sfr_model_fallbacks`。

如果没有事件、背包、validity、no-op、loop、death 或 timeout 证据，SFR 会主动 abstain。这是保护逻辑，不是训练崩溃。

## 8. Mentor 运行前检查清单

    [ ] SFR_ENABLED=False 时 baseline 行为不变
    [ ] SFR_ENABLED=True 时只改 all-failed group
    [ ] 混合成功/失败 group 保留原始 reward
    [ ] infra error group 不参与排序
    [ ] 同一 uid group 的 task_config_hash 一致
    [ ] vanilla 和 async rollout 都调用 SFR
    [ ] token_level_scores 能看到改写后的 outcome reward
    [ ] sfr_* 指标进入训练日志
    [ ] 没有残留 breakpoint/pdb 导致训练暂停
    [ ] N_GPUS 与本次集群资源配置一致
    [ ] 先完成短 smoke test，再启动正式训练

## 9. 当前边界

SFR 不是 Minecraft planner、dense reward 或 PRM。它依赖环境 info 中的低成本统计量：

- 任务名最好可以按 task_type:target 解析；复杂自然语言任务不会自动得到精确 DAG 距离。
- event counter 最好是 episode 内累计计数；如果环境只返回单步 delta，当前实现不会自动累计 delta。
- inventory schema 需要在真实服务器上确认；当前实现兼容多种常见结构。
- loop 检测需要位置字段或显式 loop 字段；没有字段时不会伪造证据。
- q 是启发式 trajectory-level proxy，不应被解释为真实成功概率。
- random_rank 和 length_rank 只用于诊断和消融。
- model_rank 的质量依赖 judge 对终止状态摘要的理解；模型输出不是环境真值。
- model_rank 会增加每个 all-failed group 的 judge 请求数，建议先用短 rollout 验证 endpoint、响应 JSON 格式和调用延迟。
- model judge 不可用时，默认 fallback=abstain，不会自动伪造模型分数；需要鲁棒运行时可显式设置 fallback=rule。

真实服务器 smoke test 后，应保存一小段 total_infos 或 rollout 日志，确认实际字段名和结构，再调整 feature_weights 或增加 Minecraft 专属解析。

# verl-agent 迁移：源码接入为我们的 RL 训练实现

> **源码 vendor verl-agent**（不是 pip 依赖，不是裸 verl）。
> verl-agent = 完整 verl + agent_system 多轮环境框架 + GiGPO（NeurIPS 2025 官方
> 实现）——多轮 env 框架和 GiGPO 现成，正好覆盖我们最重的两步工作。
> 上游 verl 已内置 qwen3_5 适配器（模型后端零移植）；verl-agent 无 Minecraft 环境（verl_port/env_minecraft 是唯一现成实现）。

## 为什么 vendor verl-agent 而非裸 verl / pip

| 选项 | 结论 |
|---|---|
| pip install verl | ❌ koala 每次装包有版本漂移风险；env/适配器只能外部 monkey patch |
| vendor 裸 verl | 可行但多轮 env 框架要自搭、GiGPO 要另抄 |
| **vendor verl-agent**（选定） | ✅ 多轮 agent loop 现成（Malmo 需要的正是它）、GiGPO 现成、内嵌完整 verl；koala 经 S3 sync 零安装 |

## 目录规划

```
rl_train/
├── verl_migration.md    ← 本文档
├── verl_agent/          ← vendor 的 verl-agent 源码快照（VERL_AGENT_COMMIT 记录）
├── verl_port/           ← 不可再生的抢救资产（env_minecraft + 参考超参）
└── verl_jobs/           ← 我们的：reward、启动脚本、超参配置、对 verl_agent 的补丁
```

## 迁移实现（verl-agent）

### vendor 源码 + 环境验证

```
① 浅克隆 verl-agent → 拷入 rl_train/verl_agent/（记录 commit）
② 集群冒烟（koala 8 卡，sft env）：
   pip install -e rl_train/verl_agent/
   验证三件事：
   a. transformers 支持 Qwen3_5ForConditionalGeneration（版本可能需升）
   b. qwen3_5 适配器可用——若 verl_agent 内嵌的 2025-06 版 verl 没有它，
      从上游 verl 拷 verl/models/transformers/qwen3_5.py 单文件进 vendor
   c. FSDP 包装我们的 checkpoint-520 → 8 卡前向 → log_probs 形状正确
```

**风险点**：FSDP × fla 线性注意力内核从未在 koala 验证（自研线是 DeepSpeed
ZeRO-2）。不通则试 verl 的 DeepSpeed 后端，再不行退回自研线（止损 1 天）。

### Minecraft 环境接入

```
直接把 verl_port/env_minecraft/minecraft/ 放进
verl_agent/agent_system/environments/env_package/
对接项：minestudio/Malmo 部署路径、任务集（复用 build_task_list.py 的 easy 池）
多轮循环、obs/step/reward 管线全部复用 verl-agent 现成框架
```

### Reward 函数

```python
def minecraft_reward(frames_count, ...):
    return 1.0 if frames_count < 180 else 0.0   # 早停=成功，沿用 SUCCESS_MAX_FRAMES
```
注册为 custom reward。首版零 shaping（G=8 组内相对比较已含信号）。

### 首个 GRPO 配置

| 配置项 | 值 | 依据 |
|---|---|---|
| 起点模型 | cot-pilot-v2/checkpoint-520 | 与自研线同起点，A/B 可对照 |
| rollout | verl-agent 多轮机制，n=8/任务，temperature 1.0 | 替代 HTTP eval-harness rollout |
| KL | low_var_kl，coef 0.01 | mc-mix_coa 参考值；**顺带修复 iteration-1 的 pg 失控** |
| LR | 5e-6 起步 | mc-mix_coa 参考值 |

### 对照验证与切换决策

- verl 产物 vs 自研线同起点产物，同协议评测（easy-ng × h29 × 3 rollouts）
- 吞吐对比（verl-agent rollout vs HTTP rollout）
- 通过标准：成绩不劣于自研线 + 吞吐 ≥3× → 主线切换；自研 RL 文件（trl_sft/ 下
  5 个）归档保留


## 已经踩过的坑

1. **fla 内核**：任何环境必装 flash-linear-attention + causal-conv1d
2. **vLLM 版本冲突**：koala 的 vllm35 env 是评测 harness 专用；verl-agent 的
   rollout 有自己的 vLLM 要求——在 sft env 独立装，不动 vllm35
3. 只在明确需要时 cherry-pick 上游修复，不追新；同步后更新
   本文件 VERL_AGENT_COMMIT
4. verl-agent 全仓较大，vendor 时评估只留必要子树
   （verl/ + agent_system/ + gigpo/ + examples/ 参考 + setup 文件），不带
   docs/tests/.github


## VERL_AGENT_COMMIT

（96MB → 3.3MB 最小快照）。vendor 内的本地补丁（相对上游 verl-agent）：

1. `verl/models/transformers/qwen3_5.py` —— 从上游 verl 拷入（661 行新版，含 packed seq / fla chunked Gated-DeltaNet / 多模态 embed）
2. `verl/models/transformers/monkey_patch.py` —— 移植 qwen3_5 两处注册分支（site-1 backend 选择 + site-2 完整 monkey patch；Ulysses SP 改为显式拒绝——比上游静默 patch 更安全）
3. `agent_system/environments/env_package/minecraft/` （verl_port 迁入）
4. `agent_system/environments/env_manager.py` —— 新增 minecraft elif 分支（Manager 暂复用 Webshop 实现）

测试脚本：`rl_train/verl_jobs/smoke_step1.py`（A import / B tiny 混合模型+适配器 / C 真实 checkpoint-520 原生前向+生成+patch 后 log_probs）

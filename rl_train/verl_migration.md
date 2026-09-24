# verl-agent 迁移：源码接入为我们的 RL 训练实现

> 方式（09-24 定稿）：**源码 vendor verl-agent**（不是 pip 依赖，不是裸 verl）。
> verl-agent = 完整 verl + agent_system 多轮环境框架 + GiGPO（NeurIPS 2025 官方
> 实现）——多轮 env 框架和 GiGPO 现成，正好覆盖我们最重的两步工作。
> 前置结论（09-23/24 实测查证）：上游 verl 已内置 qwen3_5 适配器（模型后端零
> 移植）；verl-agent 无 Minecraft 环境（verl_port/env_minecraft 是唯一现成实现）。

## 为什么 vendor verl-agent 而非裸 verl / pip

| 选项 | 结论 |
|---|---|
| pip install verl | ❌ koala 每次装包有版本漂移风险；env/适配器只能外部 monkey patch |
| vendor 裸 verl | 可行但多轮 env 框架要自搭、GiGPO 要另抄 |
| **vendor verl-agent**（选定） | ✅ 多轮 agent loop 现成（Malmo 需要的正是它）、GiGPO 现成、内嵌完整 verl；koala 经 S3 sync 零安装 |

**源码 vendor 的技术依据**：部署走"本地工作树 → S3 代码桶 → koala 挂载"，
文件物理存在才可靠（`external/SAM2` submodule 未初始化时 S3 上是空目录，
submodule 方案已实测否决）。仓库先例：`openagents/` 就是 17MB 纯源码 vendor。

## 目录规划

```
rl_train/
├── verl_migration.md    ← 本文档
├── verl_agent/          ← vendor 的 verl-agent 源码快照（VERL_AGENT_COMMIT 记录）
├── verl_port/           ← 不可再生的抢救资产（env_minecraft + 参考超参）
└── verl_jobs/           ← 我们的：reward、启动脚本、超参配置、对 verl_agent 的补丁
```

## 六步迁移（verl-agent 基座版）

### 第 1 步：vendor 源码 + 环境验证（1 天，风险最大，最先做）

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
ZeRO-2）。源码 vendor 的优势在此：有问题直接改 vendor 内文件。不通则试 verl
的 DeepSpeed 后端，再不行退回自研线（止损 1 天）。

### 第 2 步：Minecraft 环境接入（1-2 天，工作量因 verl-agent 大幅缩水）

```
直接把 verl_port/env_minecraft/minecraft/ 放进
verl_agent/agent_system/environments/env_package/（布局同源，当年就是这么设计的）
对接项：minestudio/Malmo 部署路径、任务集（复用 build_task_list.py 的 easy 池）
多轮循环、obs/step/reward 管线全部复用 verl-agent 现成框架
```

### 第 3 步：Reward 函数（半天）

```python
def minecraft_reward(frames_count, ...):
    return 1.0 if frames_count < 180 else 0.0   # 早停=成功，沿用 SUCCESS_MAX_FRAMES
```
注册为 custom reward。首版零 shaping（G=8 组内相对比较已含信号）。

### 第 4 步：首个 GRPO 配置（1-2 天）

| 配置项 | 值 | 依据 |
|---|---|---|
| 起点模型 | cot-pilot-v2/checkpoint-520 | 与自研线同起点，A/B 可对照 |
| rollout | verl-agent 多轮机制，n=8/任务，temperature 1.0 | 替代 HTTP eval-harness rollout |
| KL | low_var_kl，coef 0.01 | mc-mix_coa 参考值；**顺带修复 iteration-1 的 pg 失控** |
| LR | 5e-6 起步 | mc-mix_coa 参考值 |
| thought 触发 | **首版自由触发，不强制** | iteration-1 实测 FORCE_THOUGHT 致 29.2%→5.4%；概率触发留作后续实验 |

### 第 5 步：对照验证与切换决策（1 天）

- verl 产物 vs 自研线同起点产物，同协议评测（easy-ng × h29 × 3 rollouts）
- 吞吐对比（verl-agent rollout vs HTTP rollout）
- 通过标准：成绩不劣于自研线 + 吞吐 ≥3× → 主线切换；自研 RL 文件（trl_sft/ 下
  5 个）归档保留

### 第 6 步：GiGPO 升级（一天，verl-agent 基座下几乎免费）

配置切换即用（`examples/gigpo_trainer/` 现成脚本 + mc-mix_coa 超参），解决
"episode 稀疏奖励下信度分配"——自研线 signal-4 / 局部进展塑形想解决的同一问题。

## 已知坑（自研线经验直接继承）

1. **fla 内核**：任何环境必装 flash-linear-attention + causal-conv1d
   （自研线 test10 教训，见 `rl_grpo_design.md` §1.4）
2. **vLLM 版本冲突**：koala 的 vllm35 env 是评测 harness 专用；verl-agent 的
   rollout 有自己的 vLLM 要求——在 sft env 独立装，不动 vllm35
3. **verl 不是银弹**：FORCE_THOUGHT 伤害是算法问题换框架不解决（第 4 步
   "自由触发"已规避；KL 内置顺带修 pg 失控）
4. **上游同步纪律**：只在明确需要时 cherry-pick 上游修复，不追新；同步后更新
   本文件 VERL_AGENT_COMMIT
5. **vendor 体量纪律**：verl-agent 全仓较大，vendor 时评估只留必要子树
   （verl/ + agent_system/ + gigpo/ + examples/ 参考 + setup 文件），不带
   docs/tests/.github——吸取 CrossAgent 143MB 整项目 vendor 的教训

## 时间预算

~1 周（原 1.5 周，第 2/6 步因 verl-agent 基座缩水；第 1 步不通则止损 1 天）。

## VERL_AGENT_COMMIT

（vendor 时填：`________`）

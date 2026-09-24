# verl 迁移：把上游 verl 源码接入为我们的 RL 训练实现

> 方式（09-24 定）：**源码 vendor**（最小快照 + editable install），不是 pip 依赖。
> 目标：用 verl 替换自研 `train_grpo.py` 那套 RL 实现，承接 Malmo rollout →
> GRPO/GiGPO 训练的完整循环。
> 前置结论（09-23 实测）：上游已内置 qwen3_5 适配器（661 行，含 fla 内核路径），
> 模型后端零移植。

## 为什么源码接入而非 pip

| | pip 依赖 | 源码 vendor（选定） |
|---|---|---|
| koala 任务启动 | 每次 pip install（版本漂移 + 源可用性风险） | **零安装**——S3 sync 自带源码 |
| env / 适配器修改 | 外部注册/monkey patch | **直接改 vendor 源码**（CrossAgent 当年验证过的做法） |
| 版本锁定 | 松 | 快照锁死 commit |
| 上游更新 | pip upgrade | 手动 cherry-pick（可接受） |

**选源码的技术依据**：本仓库部署走"本地工作树 → S3 代码桶 → koala 挂载"，
文件物理存在才可靠（`external/SAM2` submodule 因未初始化，S3 上是空目录——
submodule 方案已实测不可行）。仓库先例：`openagents/` 就是 17MB 纯源码 vendor。

**最小化原则**（吸取 CrossAgent 教训——143MB 整项目 vendor 最终变成死代码）：
只拷 `verl/` 包本体 + setup 文件（~10-20MB），不带 examples/docs/tests/recipe
（参考配置已单独抢救在 `verl_port/ref_configs/`）。

## 目录规划

```
rl_train/
├── verl_migration.md    ← 本文档
├── verl/                ← vendor 的上游源码（快照，记录 commit hash）
├── verl_port/           ← CrossAgent 抢救资产（参考材料，不直接 import）
└── verl_jobs/           ← 我们的：env 注册、reward、启动脚本、超参配置
```

## 六步迁移

### 第 1 步：vendor 源码 + 环境验证（1 天，风险最大，最先做）

```
① 浅克隆上游（verl-project/verl）→ 拷 verl/ 包 + setup.py/requirements
   到 rl_train/verl/，VERL_COMMIT 记入本文件
② 集群冒烟（koala 8 卡，sft conda env）：
   pip install -e /data/work/run_codes/Minecraft-CoT/rl_train/verl/
   验证三件事：
   a. transformers 版本支持 Qwen3_5ForConditionalGeneration
   b. verl.models.transformers.qwen3_5 存在且 apply_monkey_patch 能注册
   c. FSDP 包装我们的 checkpoint-520 → 8 卡前向 → log_probs 形状正确
```

**风险点**：FSDP × fla 线性注意力内核组合从未在 koala 验证（自研线是
DeepSpeed ZeRO-2）。源码 vendor 的优势恰好在此：适配器有问题可以**直接改
`verl/models/transformers/qwen3_5.py`**。若仍不通，评估 verl 的 DeepSpeed 后端，
再不行退回自研线（止损成本 1 天）。

### 第 2 步：Minecraft 环境接入（3-4 天，工作量大头）

把 Malmo 包装成 verl 的多轮 agent 环境：

```
一次 episode = verl agent loop 的多轮对话
  每轮：obs = 游戏帧（PIL image）→ 模型生成 action 文本 → 动作投影执行 → 下一帧
env 实现：抢救的 rl_train/verl_port/env_minecraft/minecraft/
  （envs.py + projection.py）作为起点，对接现有 minestudio/Malmo 部署
任务集：复用 build_task_list.py 的 easy 池采样
接入位置：直接写进 vendor 源码的 env 注册表（verl_jobs/ 只放我们的配置和脚本）
```

### 第 3 步：Reward 函数（半天）

```python
def minecraft_reward(frames_count, ...):
    return 1.0 if frames_count < 180 else 0.0   # 早停=成功，沿用 SUCCESS_MAX_FRAMES
```
注册为 verl custom reward。首版不做任何 shaping（G=8 组内相对比较已含信号）。

### 第 4 步：首个 GRPO 配置（1-2 天）

| 配置项 | 值 | 依据 |
|---|---|---|
| 起点模型 | cot-pilot-v2/checkpoint-520 | 与自研线同起点，A/B 可对照 |
| rollout | vLLM colocated，n=8/任务，temperature 1.0 | 替代 HTTP eval-harness rollout（吞吐应显著提升） |
| KL | low_var_kl，coef 0.01 | mc-mix_coa 参考值；**同时修复 iteration-1 的 pg 失控**（0.5→95 漂移正是缺 KL 所致） |
| LR | 5e-6 起步 | mc-mix_coa 参考值 |
| thought 触发 | **首版自由触发，不强制** | iteration-1 实测 FORCE_THOUGHT 把成功率 29.2%→5.4%；"概率触发"留作后续配置实验 |

### 第 5 步：对照验证与切换决策（1 天）

- verl-GRPO 产物 vs 自研线同起点产物，同协议评测（easy-ng × h29 × 3 rollouts）
- 吞吐对比（colocated vLLM vs HTTP rollout 每千步耗时）
- 通过标准：成绩不劣于自研线 + 吞吐 ≥3× → 主线切到 verl；自研 RL 文件
  （trl_sft/ 下 5 个）归档保留（回归用）

### 第 6 步（后续迭代）：GiGPO 升级

从 verl-agent 取 GiGPO（episode + step 级嵌套组优势）实现，**直接写进 vendor
源码或 verl_jobs/**，替换 GRPO——解决"episode 稀疏奖励下信度分配"（自研线
signal-4 / 局部进展塑形想解决的同一问题）。

## 已知坑（自研线经验直接继承）

1. **fla 内核**：任何环境必须装 flash-linear-attention + causal-conv1d
   （自研线 test10 教训，见 `rl_grpo_design.md` §1.4）
2. **vLLM 版本冲突**：koala 的 vllm35 conda env 是评测 harness 专用；verl
   rollout 对 vLLM 有自己的版本要求——在 sft env 独立装一套，不动 vllm35
3. **verl 不是银弹**：iteration-1 的 FORCE_THOUGHT 伤害是算法/数据问题，换框架
   不解决（第 4 步"自由触发"已规避；KL 内置则顺带修了 pg 失控）
4. **上游同步纪律**：vendor 后上游仍在活跃开发——只在明确需要某个上游修复时
   cherry-pick，不追新；每次同步更新本文件的 VERL_COMMIT

## 时间预算

~1.5 周（第 1 步不通则止损，只花 1 天）。

## VERL_COMMIT

（vendor 时填：`________`）

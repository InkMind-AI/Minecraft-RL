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

#### ✅ 验收结果（09-27，`verl-c4`）

```
>>>>>>>>>> A+B（同进程）
[OK] ulysses_sp_size>1 被正确拒绝
[OK] torch backend 前向: log_probs(1, 11)
SMOKE_RESULT: A,B (2/2) ✅ 通过
>>>>>>>>>> C（独立进程，真实 9B 权重）
model_type=qwen3_5
[OK] 真实权重 PPO 前向（fla 内核 + 9B bf16）: log_probs(1, 90), entropy(1, 90)
SMOKE_RESULT: C (1/1) ✅ 通过
```

**适配器层验收闭合**：`qwen35-9b-nf2-c3000-slim` 真实权重 + fla 内核 + verl PPO
前向路径全部走通，log_probs/entropy 有限值。第 1 步的技术未知已清零。

**C 层此前失败四次，全部是冒烟脚本自身的问题，与 verl 无关**（教训值得记住）：

| 轮次 | 失败原因 | 修复 |
|---|---|---|
| `c` | `AutoModelForImageTextToText` 加载方式错 | 改用正确的 Auto 类 + `trust_remote_code` |
| `c2` | `isfinite(None)`：`apply_monkey_patch` 是**类级**补丁，check_b 调用后 check_c 的"原生前向"其实已是 PPO 路径（`logits=None`） | 改为校验实际存在的字段 |
| `c3` | **根本没入队**（提交输出被 watch 机制吞掉，误以为在跑） | 提交后必须核实队列/S3 日志目录 |
| `c4` | — | **通过**。`--only` 让 C 在独立进程跑，从机制上消除类级补丁泄漏，不再靠注释规避；同时改为失败即 `exit(1)`（此前部分通过也返回 0） |

**另一条有用的事实**：`forward_with_torch_backend` 在 `labels=None` 时回退到
`torch.roll(input_ids, -1)`，因此不传 labels 也能算出 log_probs——这正是 RL 训练
计算 `old_log_probs` 时的真实调用形态，C 层验的就是这条路径。

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
5. **openha env 不是镜像预装的**：它由 `run_backbone_eval.sh` 在运行时创建。环境侧
   verl 任务如果自己 `conda activate openha || true`，会静默失败并跑在 base env
   （verl-env-d1：`No module named 'torch'`）。统一改为
   `source rl_train/verl_jobs/setup_openha_env.sh`
6. **conda 钩子不兼容 `set -u`**：openjdk 包的 `deactivate.d/openjdk_deactivate.sh`
   直接读 `$JAVA_HOME_CONDA_BACKUP`，在调用方 `set -euo pipefail` 下装完 openjdk
   就退出（verl-env-d2）。setup 脚本进入时关 nounset、结束时按原样恢复
7. **提交 ≠ 入队**：koala 提交输出偶尔被 watch 机制吞掉（c3 从未入队），`koala ls`
   也偶尔抽风返回空。**判定标准一律是 S3 日志目录
   `axiomjin/.koala-logs/<job>-*/` 是否存在**

## 第 2 步（环境接入）验收：✅ 已通过（09-28，`verl-env-d5`）

`smoke_step2.py` 四层：D（projection 解码/非法降级）、E（Ray+Malmo `reset()`，
含"同组共享 task_config"校验）、F（`step()` 的 obs/reward/done/info 契约）、
G（`MinecraftEnvironmentManager`，verl 训练真正调用的入口）。

原先"已注册"的 minecraft 分支从未真正跑过一步（7 个致命 bug：import 断裂 /
调用契约不符 / 无限重试挂死 / reset 平均白等 90s / KeyError / Ray worker 里
留 breakpoint() / 组内各 worker 各自随机种子破坏 GRPO 可比性）；`env_manager.py`
直接复用 WebshopEnvironmentManager 也不成立（对图像观测做字符串切分）；
`projection.py` 把任何垃圾输出都判 valid=1。已全部重写，详细清单见
`verl_debug_progress.md` 第 2 层。

| 轮次 | 结果 | 原因 |
|---|---|---|
| d1 | ❌ D/E/F 全挂 | 跑在 base env（坑 5） |
| d2 | ❌ 没跑到 D | 装完 openjdk 后 nounset 退出（坑 6） |
| d4 | ❌ E 挂 | Malmo 引擎首次运行需交互确认下载，非交互 job 直接 EOFError |
| **d5** | **✅ D,E,F,G (4/4) 通过** | 补齐 setup 脚本里的 cuda-python 钉版本 + Malmo 引擎 S3 镜像下载 |

⚠️ 已知限制：`MinecraftEnvironmentManager` 每样本仅 1 张图，与评测的 h29
多图历史不一致，是后续必须解决的结构性差距。

## 第 3 步（GRPO 训练启动）：进行中

详见 `verl_debug_progress.md`。`colocate-p2` 已确认 torch2.10+transformers5.15+
vllm0.17 组合可行（HF↔vLLM 权重同步能走通）；`run_grpo_minecraft_smoke.sh`
六轮迭代已修复 5 个独立兼容性问题（Hydra struct 键校验、batch/GPU 整除断言、
flash_attn.bert_padding 无条件导入、AutoModelForVision2Seq 改名），`smoke6`
验证中。


## VERL_AGENT_COMMIT

（96MB → 3.3MB 最小快照）。vendor 内的本地补丁（相对上游 verl-agent）：

1. `verl/models/transformers/qwen3_5.py` —— 从上游 verl 拷入（661 行新版，含 packed seq / fla chunked Gated-DeltaNet / 多模态 embed）
2. `verl/models/transformers/monkey_patch.py` —— 移植 qwen3_5 两处注册分支（site-1 backend 选择 + site-2 完整 monkey patch；Ulysses SP 改为显式拒绝——比上游静默 patch 更安全）
3. `agent_system/environments/env_package/minecraft/` （verl_port 迁入）
4. `agent_system/environments/env_manager.py` —— 新增 minecraft elif 分支（Manager 暂复用 Webshop 实现）

测试脚本：`rl_train/verl_jobs/smoke_step1.py`（A import / B tiny 混合模型+适配器 / C 真实 checkpoint-520 原生前向+生成+patch 后 log_probs）

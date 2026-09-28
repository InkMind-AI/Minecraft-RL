# ReST-EM CoT 自举迭代：方法设计与实验记录

> 更新至 2026-09-23 | 状态：MVP 链路已验证（verify4 全流程跑通），首个正式规模
> iteration-1（50 任务 × 8 路）在跑 | 姊妹文档：`rl_grpo_design.md`（RL 基础设施）、
> `cot_annotation_schemes.md`（CoT 标注方案）

---

## 1. 方法定位：一句话与动机

**让策略模型自己生成 thought、用客观信号筛选好坏、再用筛出来的数据更新自己——
把 thought 当隐变量做 EM 式自举，摆脱对外部标注（Gemini）的依赖。**

三个动机（全部来自 P2 的实证发现）：

1. **成本与幻觉**：外部标注花钱且有不可验证断言（sheep 案例：v1 的 Combat 类
   7% 黑名单命中率），事后清洗只能堵漏不能治本
2. **分布错配**：策略模型在模仿一个不属于自己的表达分布——v2 训练分布 12%
   thought 密度，推理时自发发射率只有 4.1%。自举让模型学"自己会说的话"
3. **归因链已闭合**：ctrl 对照（同数据剥 thought）证明数据选择才是 +6.6pp 的
   增益主体；thought 若要产生净贡献，必须比 v2 更贴近策略自身分布

## 2. 与相关工作的关系

- 谱系：**STaR → ReST → ReST-EM**（Google 的迭代式自提升方法），但它们全部
  在纯文本域（数学题/代码），答案正确性可即时验证
- 本方法的差异化：**长时具身、部分可观测场景**下的自举。奖励是 episode 级
  稀疏二值（200 步一个成败标签），不存在"逐步可验证的答案"
- 核心难点与对策：信度分配（一条 episode 的标签怎么摊到 10+ 个决策点）——
  用 GRPO 组内相对优势 + 计划中的真值 verifier（见 §3.3 的四信号设计）

## 3. 方法设计

### 3.1 EM 形式化

把 thought z_t 当隐变量，动作序列与最终成败当观测：

- **E-step**：当前策略 π_θ 在每个决策点采样候选 thought，用多信号打分做
  后验加权（软筛选，不是硬过滤）
- **M-step**：用加权后的 (观测, thought, 动作) 更新 θ
- 迭代：新策略 → 新 rollout → 新 thought → 新权重，滚雪球式自提升

### 3.2 五步迭代循环（`trl_sft/rl_loop.sh`，`BOOTSTRAP=1`）

```
① Rollout     N_TASKS 任务 × GROUP_SIZE 路（复用 eval harness：vLLM + Malmo）
              FORCE_THOUGHT=1：每步强制以 "Thought: " 前缀续写
              ——解耦"要不要想"和"想得好不好"，不依赖模型自发触发意愿
② 适配        rl_build_batch.py：episode → parquet + rewards + groups
              决策点感知剥离：只有 decision_points.py 检出的决策点保留
              thought（对齐人工标注 12.4% 密度，不引入分布偏移），
              其余剥回纯 "Action: ..."
③ 打分        信号2（言行一致）× 信号3（组内优势）→ 合成权重
④ 更新        train_grpo.py：加权 GRPO（DeepSpeed ZeRO-2，8 卡）
⑤ 发布        新权重 → 下一轮 ①
```

### 3.3 打分筛选：四信号设计（两层组合，非加权求和）

打分只回答三个问题：**说的是不是真的（信号1）、说了有没有用（信号2）、
结果好不好（信号3）**；信号4 补稀疏信号的洞。

| 信号 | 内容 | 状态 | 说明 |
|---|---|---|---|
| 1. 真话检验 | 用 MineStudio meta_info 逐帧真值（位置/朝向/inventory/血量）核对 thought 断言；查不出的不打分（防误杀主观判断） | **待实现** | 硬淘汰项：实锤说谎一票否决 |
| 2. 言行一致 | thought 陈述的意图 vs 实际动作的语义比对（攻击/交互/移动/停止四类，词根正则匹配，查不出关键词返回 None 不参与打分） | ✅ MVP 已实现 | `thought_consistency.py` |
| 3. 组内优势 | GRPO：同任务 G 路的成败做组内中心化；全成/全败的零方差组 advantage=0（宁可不学，不要瞎学） | ✅ MVP 已实现 | `grpo_core.py`（复用） |
| 4. 局部进展 | potential-based reward shaping：用真值算"离任务目标更近了吗"作局部代理奖励（Ng et al. 1999，理论上不改变最优策略） | **待实现** | 只做权重微调，防被钻空子（绕远路刷距离变化） |

**两层组合原则**：
- 第一层（硬淘汰）：信号1 查实说谎、信号2 严重矛盾 → 一票否决
- 第二层（软加权）：过了第一层的用信号3+4 决定训练权重，不非黑即白

**MVP 简化**：只上信号2×信号3。信号2 的分数（中性 1.0）直接乘进 advantage
作为窗口权重，裁剪到 [0.5, 1.5] 防极端值。

### 3.4 防坑设计（评审过风险点）

| 风险 | 对策 |
|---|---|
| 模型学会说讨好打分器的套话（模式坍缩） | 监控 thought 多样性/n-gram 去重率（待做）；打分器不偏好特定表述风格 |
| 信号4 被钻空子（刷距离变化） | 信号4 只做微调，信号3（真实结果）兜底 |
| 多信号互相矛盾 | 校准集（已人工分析的 good/bad case）定期抽查打分器准确率 |

**分阶段落地顺序**（每加一层看边际收益，天然构成消融）：
信号2+3（MVP）→ 信号1 → 信号4。

## 4. 实现细节

### 4.1 组件与代码

| 组件 | 文件 | 要点 |
|---|---|---|
| 前缀强制 | `openagents/agents/openha.py` | `enforce_format` 扩展支持 online 模式：vLLM OpenAI 兼容服务的 `continue_final_message=True` + `add_generation_prompt=False`，把末尾 assistant 消息当续写前缀（与本地 vllm 模式的 prompt 拼接等价） |
| 信号2 打分 | `trl_sft/thought_consistency.py` | `split_response`（三格式匹配）/ `action_consistency_score` / `strip_and_score`（决策点感知剥离+打分） |
| 适配器扩展 | `trl_sft/rl_build_batch.py` | `--strip-non-decision-thought --thought-signal`：非决策点剥 thought、信号2 逐窗口打分写旁车 npy |
| 循环编排 | `trl_sft/rl_loop.sh` | `BOOTSTRAP=1` 分支：FORCE_THOUGHT 导出、信号合成（adv × clip(sig,0.5,1.5)）、GPU 硬清理 |
| 训练器 | `trl_sft/train_grpo.py` | **零改动**——合成权重走现有 advantages 数组机制 |

### 4.2 关键格式发现（写代码前不可能知道的）

FORCE_THOUGHT rollout 下，`continue_final_message` 使 vLLM **只回传续写的新
token，不含被注入的 "Thought: " 前缀字面文本**——真实响应是
`" <感知>\|<状态>\|<决策>\nAction: ..."`。`split_response` 必须三格式匹配：
显式 `Thought:`（人工标注数据）/ 隐式正文+`\nAction:`（强制触发 rollout）/
纯 `Action:`（非决策点正常输出）。

## 5. 实验记录（按时间序，含全部排障）

### 5.0 前置：RL trainer 冒烟（test3-10，详见 rl_grpo_design.md §4）

自举依赖的 `train_grpo.py` 先后排除四层 OOM 根因：flash-attn 未装 →
sdpa O(N²) 注意力 → reentrant checkpoint 静默失效（冻结 ViT 后输入链无 grad）→
**fla naive fallback（真凶：`LINEAR_ATTN_KERNELS=1` 从未装线性注意力内核）**。
最终 test10 配置（flash-attn + 内核齐全 + non-reentrant checkpoint + fused CE
logprob）通过。结论：**30 帧长序列 VLM 训练，flash_attention_2 与 fla 内核是
硬性要求，不是可选项**。

### 5.1 本地单测（09-22，零 GPU）

对 `thought_consistency.py` 纯逻辑单测，实抓两个 bug：
1. 词根正则：`\battack\b` 匹配不上 "attacking"（ing/ed 语态变形），改 `\w*` 词根
2. 停止判定：`parse_action.is_noop` 只认字面 "no_op"，真实空动作是
   `"move(0, 0) and press()"`（空 keys），改按 keys/click 皆空判定

### 5.2 bootstrap-iter1（首次提交）：秒退

`source common.sh` 内部 `cd` 到自身目录，后续 `cd trl_sft` 找不到目录。
修正命令顺序（在 trl_sft 里 source，完事 `cd ..`）。

### 5.3 bootstrap-iter1b：静默全灭（重要事故）

每个 episode 的 agent 构造都触发 assert：`enforce_format` 只支持本地 vllm
模式，而评测链路全部用 online HTTP 模式。**危害放大链**：Ray actor 内的异常
被捕获只打日志 → bash `|| true` 又吞掉非零退出 → 任务状态显示正常跑完，
实际零有效 rollout。发现方式：日志里 "use enforce_format, only support vllm"
后仍见任务推进。
**修复**：§4.1 的 continue_final_message 方案。

### 5.4 bootstrap-verify（第一次小规模）：任务数失控

设了 `N_TASKS=5` 却跑了远超预期的任务——排查发现 **`N_TASKS` 是死代码**，
真正控制变量是 `EASY_NUM_TASKS`（默认 300！）。手动删除任务止损。
**修复**：rl_loop.sh 里 `export EASY_NUM_TASKS="$N_TASKS"` 转译。

### 5.5 bootstrap-verify2（5 任务×2 路）：②环节路径错误

Rollout 5 任务全部成功（~30 分钟），`rl_build_batch.py` 报 FileNotFoundError：
`--eval-output` 拼的是一层目录，真实结构是
`eval_output/<MODEL_LOCAL_NAME>/<SERVED_MODEL_NAME>-text_action/`（两层）——
09-20 原始设计的死代码，从未被端到端跑过。
**离线补救**：用 verify2 已上传 S3 的真实数据做离线验证（不烧 GPU）：
- 200 步 episode 检出 55 个决策点（27.5%，高于训练分布 12.4%——视觉搜索类
  episode 方向变化频繁，属正常）
- 信号2 可判定 39/55（71%），可判定样本初检全部 1.0（言行一致）
- 同时发现 §4.2 的隐式格式问题并修复

### 5.6 bootstrap-verify3（5 任务×2 路）：④环节 OOM

①②③全部正常：**357 个决策点、信号2 可判定 268 个（75.1%）、可判定均分
0.791**；组利用率 20%（5 组仅 1 组非零方差——2 路采样下全成/全败组占多数，
属小规模固有限制）；信号2 因子 mean=0.837 / min=0.500 / max=1.000。
④训练 OOM，报错关键行：`Process 4299 has 126.32 GiB memory in use`——
**rollout 阶段的 vLLM 服务进程没被杀干净**（`kill $VLLM_PID` 只杀了
`conda run` 包装进程顶层，引擎 worker 子进程存活）。
**修复**：rl_loop.sh 在①与②之间加硬清理（pkill vllm 模式 + nvidia-smi
compute-apps 全扫 + 显存回落确认）。

### 5.7 bootstrap-verify4（5 任务×2 路）：**全链路首次跑通** ✅

```
① 10 episodes 全部 200 步完成，强制 thought 全程生成
② 449 个决策点检出，68 行训练样本
③ 信号2 可判定 368 个（82.0%），均分 0.709；合成因子正常
④ 8 步训练正常（零方差组 loss=0 符合预期），无 OOM
⑤ 新权重 18.8GB 落盘 S3: minecraft-rl-policy/iterverify4/final/
```
遗留一个无害收尾 bug：`ITER` 为字符串（"verify4"）时 `$((ITER+1))` 报
unbound variable——已修（纯数字才提示 +1）。

### 5.8 iteration-1c（当前在跑，首个正式规模）

```
起点: cot-pilot-v2/checkpoint-520（29.2% headline）
规模: N_TASKS=50 × GROUP_SIZE=8（400 episodes，对齐 rl_loop.sh 默认）
提交: 09-23 13:18，8 GPU
```

## 6. 实验结论汇总（截至 09-23）

### 6.1 机制验证结论

1. **强制触发有效**：continue_final_message 路径下模型每步产出结构良好的
   三段式 thought（感知|状态|决策），零格式失败——"什么时候想"与"想什么"
   成功解耦
2. **自产 thought 质量不差**：信号2 可判定比例 71-82%，可判定均分 0.709-0.791
   ——模型自己说的话大部分与实际动作一致，初步支持"自举有原料可用"
3. **决策点检测在自产数据上工作正常**：55/200（单 episode）到 449/68行
   （5 任务合并）的密度波动反映任务类型差异，剥离机制运转正确
4. **小规模组利用率低是结构性问题**：GROUP_SIZE=2 时仅 20% 组有非零方差
   ——正式规模用 G=8 正是为解决此问题

### 6.2 工程结论（踩坑清单，按严重度）

| 坑 | 教训 |
|---|---|
| Ray + `|| true` 双重吞错 | 静默失败比崩溃危险得多——verify3 若不查 "Process 4299" 这种报错细节就归因到训练本身 |
| 死代码变量（N_TASKS） | 从未被运行过的编排代码 = 未定义行为；冒烟测试必须校验"实际做了什么"而非"没报错" |
| 多进程服务的 kill 语义 | kill 顶层包装 PID ≠ 杀进程树；跨阶段共享 GPU 的作业必须有显式清理+显存确认 |
| koala CLI 的 git 缓存 | "初始化失败"三种根因：缓存损坏（搬家修）、陈旧锁（删锁修）、**git ref 损坏**（`echo <hash> > .git/refs/heads/master` 手动重建——09-23 新模式） |

### 6.3 尚未回答的问题（等 iter1c 及后续）

1. **自举是否提升成绩**：iter1c 权重 vs v2 基线 29.2%——方法成立与否的
   第一判据
2. **强制训练能否迁移到自由触发**：迭代后（撤掉 FORCE_THOUGHT）发射率是否
   高于 v2 的 4.1%
3. **多轮收敛性**：groundedness/多样性随迭代的变化（模式坍缩监控）
4. **信号1/4 的边际收益**：按分阶段计划逐个加

## 7. 下一步计划

1. iter1c 出结果 → 评测权重 vs 29.2%（同协议 easy-ng × h29 × 3 rollouts）
2. 若 ≥ 基线：跑 iter2/iter3 看多轮趋势 + 监控发射率与 thought 多样性
3. 若 < 基线：上信号1（meta_info 真值 verifier，设计见 §3.3）再迭代
4. 并行：把信号1 verifier 开发排期（与 Phase 1 标注升级共享组件）

## ⚠️ iteration-1 结果（09-28）：权重崩溃，不能评测

iter1c 评测跑了 23h 仍在第 1 个 episode 的第 9 步（70-81 s/step，SFT 权重是
1.04 s/step），已删除。诊断探针 `rl_train/diag/probe_policy.py`（vLLM 离线，
同 prompt，n=8，max_tokens=256）直接比对自举前后：

| 指标 | v2-e4（起点） | iter1/final（自举后） |
|---|---|---|
| 生成 token 中位数 | 13 | **256（=上限）** |
| 触顶 max_tokens | 0% | **100%** |
| 含 `Action:` | 100% | 50% |
| 典型输出 | `Action: move(0, 0) and press()` | `<think>\n\n</think>\n\nAction: Walk forward across the terrain\nReason: ...`（无限续写） |

**这不只是 EOS 丢失，是格式整体崩溃**，三个症状：
1. **不再停止**：100% 触顶，每步都生成满 256 token → 评测慢 70 倍的直接原因
2. **动作语法丢失**：`Action: Walk forward...` 是自然语言，不是 `move(dx,dy) and
   press(...)` 语法，projection 解析不了
3. **Qwen3 思维模板泄漏**：`<think>\n\n</think>` 反复出现——训练数据里没有这个，
   是基座 chat template 的 thinking 块。说明 GRPO 训练把模型推离了 SFT 格式，
   回落到了基座的先验

**根因候选**（按可能性排序，下一步逐一核对 `train_grpo.py` / `rl_build_batch.py`）：
- **chat template 不一致**：训练时渲染的 prompt 是否带了 `enable_thinking=False`？
  SFT collator 从 processor 解析并注入 `chat_template_kwargs`（见 collators.py
  `_resolve_chat_template_kwargs`），自研 GRPO 很可能没做 → 训练序列里出现空的
  `<think></think>` 块，模型学会了生成它
- **`<|im_end|>` 未进 loss**：thought 加权 loss 若只覆盖 thought/action 文本 token，
  EOS 就失去监督
- **无 KL 约束 / LR 过大**：`verl_migration.md` 已记录 iteration-1 "pg 失控"，
  首个 verl 配置因此加了 low_var_kl coef 0.01

**结论**：自研线 iteration-1 作废。这个问题**不依赖迁移**——哪怕换到 verl，
chat template 对不上也会同样崩。所以修复点要在 rollout→训练样本的渲染一致性上，
两条线共用。

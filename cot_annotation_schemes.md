# CoT 标注方案：设计、实验与结论

> 更新至 2026-09-22 | 状态：**v2 配方定稿**，D 方案曲线未见顶（可补跑 e6-e8）

## 1. 标注链路总览

```
原始训练集 parquet（noop-filtered，363 shard，每行 = 30步切段轨迹）
  ① 决策点检测      decision_points.py（v2 冻结）→ "哪些步值得想"（实测 12.4%）
  ② Payload 组装    keyframes（决策点前帧）+ 动作日志（决策点行标 >>）
  ③ API 标注        annotator.py（gemini-3.8-flash via Venus 网关，可插拔）
  ④ 校验           step 对齐 + 非空 + ≤45词 + 黑名单（v2 新增）
  ⑤ 渲染插入       "Thought: xxx\nAction: ..." 进 assistant turn
  ⑥ 训练集构建     多批去重 + 20% 回放混入（防同轨迹矛盾监督）
  ⑦ continue-SFT   从 headline 模型续训（LR 1e-6，5 epochs，每 epoch 存档）
  ⑧ 评测           easy-ng × h29 × 3 rollouts（与基线严格同协议）
```

## 2. 三代标注方案对比

| | **v1（初版）** | **v2（定稿）** 🥇 | D 方案 🥈 |
|---|---|---|---|
| Thought 定位 | 三段式感知\|状态\|决策，允许自由因果叙述 | **只写单帧可验证感知** | **任务进度追踪**（时序进展型） |
| 核心规则 | 无约束 | 禁 14 类不可验证断言（killed/died/defeated/task complete/taking damage/within range 等）+ 状态段仅 HUD 可见且只报变化 | thought 记录"离目标还有多远"的进展 |
| 校验 | 对齐+非空+词数 | + **黑名单正则**（命中丢该 thought） | 同 v2 |
| Combat 配比 | 4%（失误配比） | **23%**（定向补标 1,600 条） | 同 v2 基础上调整 |
| 数据规模 | 6,661 行 / 18,495 thoughts | 8,261 行 / 22,777 thoughts（剥离 1,452 病灶） | ~8,261 行 |
| **评测最优** | **25.3%**（e4） | **29.2%**（e4） | **28.8%**（e5，曲线未见顶） |
| Embodied / Combat | 27.5% / 18.4% | 31.2% / 23.1% | 31.0% / 21.8% |
| thought 发射率 | ~1% | **4.1%** | — |

**共同起点**：基线（headline 模型，无 continue-SFT）= 24.9%。

## 3. v1 → v2 的四处改进（全部对症）

| # | 改动 | 依据（实证） | 效果 |
|---|---|---|---|
| ① | Prompt 重写：禁不可验证断言 | sheep 失败标本——thought 幻觉"剪毛完成""羊已死亡"→ 写进历史 → h29 回放自我强化 → 挥空 170 次锁死 | Combat 类黑名单命中 7% → 0.3% |
| ② | 校验加黑名单 | 同上（prompt 绕过时兜底） | 脏 thought 进不了训练集 |
| ③ | 存量清洗 1,452 条 | 全量扫描 v1 的 18,495 thoughts：幻觉断言 ~357 + 等待型低信息 ~1,095 | thought 级手术，轨迹保留 |
| ④ | Combat 配比 4%→23% | 扫描发现 Combat 轨迹仅 268/6,661（评测集 24% Combat）——Combat -7pp 主因 | Combat 成绩 18.4%→23.1% |

## 4. 关键实验结论

### 4.1 案例分析：两类 thought 的可验证性不对称

| | Embodied 感知 thought | Combat 状态 thought |
|---|---|---|
| 断言类型 | "目标在视野内/选中框下"——**单帧可验证** | "在射程内/正在掉血/已消灭"——隐藏状态 |
| 错误后果 | 看不见就是看不见，不误导 | 错误断言 + 历史回放自我强化 → 锁死错误行为 |
| 净效应 | 全部增益来源（allium 0/3→3/3） | 全部退化来源（sheep 2/3→0/3） |

### 4.2 归因对照（顺带发现，供背景）

同数据剥光 thought 的对照组达到 **31.5%**（全场景最高）——数据选择（决策点检测器筛轨迹）才是增益主体（+6.6pp），thought 文本净贡献为负（v1 -6.2pp / v2 -2.3pp）。**在"必须带 thought"的标注方案赛道内**，v2 处方已把 thought 的伤害降到最低并转化为发射率/类别平衡的改善。

### 4.3 Epoch 曲线（三方案）

```
v1:  24.5 → 24.5 → 25.2 → 25.3 → 23.3   （e4 峰值，e5 过拟合）
v2:  26.1 → 27.5 → 27.3 → 29.2 →  (e5 存档丢失)   （e4 峰值）
D:   21.3 → 21.9 → 27.3 → 27.6 → 28.8   （未见顶，e5 是最后存档）
```

## 5. 终判：v2 配方定稿为扩量标注标准

```
✅ V2 prompt：只写单帧可验证感知
   - 感知段必须具体（实体名/方块类型/相对位置）
   - 状态段仅 HUD 可见且只报变化，无变化整段省略
   - 决策段只许引用可见证据（"crosshair 对准 X"）
   - 明文禁止：killed/died/defeated/eliminated/task complete/
     taking damage/dealing damage/within (melee/attack/striking) range/within reach
✅ 黑名单校验：UNVERIFIABLE_RE 正则进 validate_thoughts（命中丢 thought 不丢轨迹）
✅ Combat 配比 23%（标注时定向分层采样）
✅ 决策点检测器 v2（R1 按键变化 + R2 点击切换，触发率 12.4%）
```

**D 方案的遗留悬念**：e5=28.8% 仍在上行，补跑 e6-e8 是低成本可选实验；若反超则切换。

## 6. 标注器工程要点（可复用）

- **可插拔三后端**：stub / openai（兼容网关）/ gemini，环境变量切换零改码
- **稳健性**：429 指数退避（10s→320s × 8）、400 渐进降级（先摘 response_format 再摘 temperature）、JSON 容错解析
- **断点续跑**：done_ids.txt + 50 条原子 chunk + 每 500 条增量落盘
- **事故教训**：Venus 配额耗尽时 API 静默失败 → 存 4,805 行无 thought 空壳。**失败率超阈值必须硬报错**（修复项）
- 速率参考：gemini-3.8-flash ~24 轨迹/分钟，1.6k 条 Combat 定向批 21 分钟跑完

## 7. 相关文件索引

| 模块 | 路径 |
|---|---|
| 决策点检测 | `cot_annotation/decision_points.py` |
| Prompt（V2 版） | `cot_annotation/prompts.py` |
| 标注器 | `cot_annotation/annotator.py` |
| 管线（含黑名单） | `cot_annotation/pipeline.py` |
| Combat 定向批 | `cot_annotation/combat_batch.py` |
| 训练集构建 | `~/cot_build/build_dataset.py` / `build_v2.py` |
| 训练脚本（参数化） | `trl_sft/train_stage3.sh` |
| 实验全记录 | `experiment_summary_20260905.md` §7 |

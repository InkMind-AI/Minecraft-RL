# RL 代码：最终结构与调试记录


## 1. 最终代码结构

### 1.1 文件清单

| 文件 | 职责 |
|---|---|
| `trl_sft/rl_loop.sh` | 迭代编排：rollout → 适配 → 优势 → 训练 → 发布；含 `BOOTSTRAP=1` 自举模式分支（FORCE_THOUGHT 导出、信号合成、GPU 硬清理——分支设计见 rest_em 文档，此处不重复） |
| `trl_sft/grpo_core.py` | 组内优势计算（纯 numpy，无框架依赖，可单测） |
| `trl_sft/rl_build_batch.py` | episode → 训练样本适配器（与 SFT 数据同构） |
| `trl_sft/train_grpo.py` | GRPO trainer（复用 `train_sft.py` 的模型加载/数据管线骨架） |
| `trl_sft/thought_consistency.py` | 信号2打分 + 决策点感知剥离（仅自举模式使用） |

### 1.2 数据流

```
eval_output/<MODEL_LOCAL_NAME>/<SERVED_MODEL_NAME>-text_action/   ← rollout 产物（注意两层目录）
  ├── episode.jsonl        逐帧 PNG（base64，360×640）
  └── raw_action.jsonl     逐步模型原始输出
          │
          ▼  rl_build_batch.py
batch.parquet（conversations + image_bytes，SFT 同构）
rewards.npy（帧数<180 → 1，否则 0）      groups.json（任务名列表）
          │
          ▼  grpo_core.compute_group_advantages（std 或 dr 模式）
adv.npy（零方差组 = 0）──[BOOTSTRAP=1 时 × clip(信号2, 0.5, 1.5)]──▶ 最终优势
          │
          ▼  train_grpo.py（torchrun --nproc_per_node=8，DeepSpeed ZeRO-2）
checkpoint / final 权重 ──▶ S3 ──▶ 下一轮 rollout
```

### 1.3 模块要点

**`grpo_core.py`**
```python
"std"  A = (r - mean) / (std + eps)   # 原版 GRPO
"dr"   A = r - mean                    # Dr.GRPO（去长度偏差）
# 零方差组（全成功/全失败）advantage = 0 —— 无相对信号不学
# group_coverage_report()：报告多少组有非零信号（指导 GROUP_SIZE 配置）
```

**`rl_build_batch.py`**
- 切窗：29 步不重叠窗口（对齐 h29 推理历史）。原因：200 步整段会被 `_exceeds_max_length` 丢弃，且成功 episode 更短（早停），整段喂入会系统性丢失败样本、破坏组内对比结构；窗口继承整条 episode 的奖励与组标签
- 不足 5 步的残余窗口丢弃
- 指令保真（两个防回归点）：system_prompt 文件**不可 `.strip()`**（结尾恰好一个 `\n`，strip 导致标题与指令粘连，权威拼接见 `openha.py:325`）；`window_start > 0` 的窗口指令附加 `(continuing, step N)`

**`train_grpo.py`**
```python
per_sample_pg = -(logp_sum / n_tok)        # 每样本按生成 token 数归一
loss = (per_sample_pg * adv).mean()        # adv 从旁车 .npy 按行号取，不进数据管线
# 可选：--kl_beta > 0 加载冻结参考模型算 KL（显存 +18GB/卡）
# focal 默认关闭（RL 损失自带 advantage 权重）
```
- `token_logprobs_from_logits`：**fused cross_entropy 逐 chunk**（不是 log_softmax+gather，原因见 §2）
- non-reentrant gradient checkpointing + `enable_input_require_grads()`（原因见 §2）
- `save_16bit_model` 存纯权重 + `aws s3 sync` 回传 S3

### 1.4 运行硬性要求（缺一不可，全部来自 §2 的实测教训）

1. `ATTN_IMPL=flash_attention_2` —— 触发 `bootstrap_env` 源码编译 flash-attn（预编译 wheel 与 koala torch ABI 不兼容）
2. `LINEAR_ATTN_KERNELS=1` —— 装 fla + causal-conv1d（模型带线性注意力层，缺内核会静默 fallback 到 naive 实现，~130GB OOM）
3. `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` + `model.enable_input_require_grads()` —— 冻结 vision tower 的组合下 reentrant 模式静默失效
4. logprob 用 fused `F.cross_entropy` —— `log_softmax+gather` 会 materialize [chunk, V] fp32 中间量（19k token 序列下 +23GB/卡 常驻计算图）

### 1.5 启动模板（与生产一致，冒烟测试也不许偏离）

```bash
# 环境准备（koala job 内）
cd trl_sft && source common.sh          # ⚠ common.sh 会 cd 到自身目录
export ATTN_IMPL=flash_attention_2 LINEAR_ATTN_KERNELS=1
bootstrap_env                            # 装 flash-attn/fla/causal-conv1d（幂等）
cd ..
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 训练
torchrun --nproc_per_node=8 train_grpo.py \
    --model_path s3://... --data_path ... --advantages_file ... \
    --output_dir ... --attn_implementation flash_attention_2 ...
```

---

## 2. 调试过程（test3-10：四层洋葱排障史，最终全通过）

| 测试 | 配置 | 结果 | 根因与教训 |
|---|---|---|---|
| test3 | 1 卡 + sdpa | ❌ ImportError | 裸调 `train_grpo.py` 跳过了 `bootstrap_env`——flash-attn 没装 |
| test4 | 2 卡 ZeRO2 + sdpa | ❌ OOM（首次前向） | 单/双卡下 ZeRO2 静态占用 ~63GB，激活预算不足 |
| test5 | 2 卡 + 优化器 CPU offload + sdpa | ❌ OOM | offload 后静态仅 ~27GB 仍爆——当时误判为 sdpa materialize O(N²) 注意力矩阵 |
| test6 | 2 卡 + flash-attn | ❌ OOM | flash-attn 确认装上（2.7.4.post1）仍爆 135GB——排除注意力实现，指向代码层 |
| test7 | **8 卡 + flash-attn（=生产配置）** | ❌ OOM 134GB | 决定性一击：与生产 SFT 完全相同配置也爆 → 问题在 train_grpo.py 自身。归因 log_softmax+gather materialize [chunk,V] fp32 中间量（~+35GB），改 fused cross_entropy |
| test8 | 8 卡 + fused CE + offload | ❌ OOM 135GB | CE 重构（真优化但非主犯）无效——135GB 与 test7 一字不差，说明大头另有其人 |
| test9 | 8 卡 + non-reentrant checkpoint + enable_input_require_grads | ❌ OOM 135GB | 修复 reentrant checkpoint 静默失效（冻结 ViT 后输入链无 grad）——仍无效，但排除了第三层 |
| **test10** | 8 卡 + **`LINEAR_ATTN_KERNELS=1`**（装 fla/causal-conv1d） | ✅ **通过** | **真凶**：模型带线性注意力层，fla 内核从未安装，旧版 fla 在 Triton 不兼容时静默 fallback 到纯 PyTorch naive 实现（O(N²) 中间量 ≈130GB，与卡数无关——解释了 2 卡/8 卡内存一字不差的所有"反常"） |

### 调试结论

1. **四个硬性要求**（已固化为 §1.4，写进代码注释与启动模板）
2. **冒烟测试必须与生产启动命令逐旗标对齐**，不能"看起来等价"——fla 内核缺失在 SFT 侧从未出问题（生产脚本每次都带 `--linear-attn-kernels`），只有脱离生产脚本的裸调用才踩中
3. **教训（test5-6 的错误归因过程本身）**：排除法得出的"最合理解释"可能是错的——sdpa O(N²) 假说看似完美解释了 test5 的数据，却在 test7（8 卡也爆）后崩塌。内存问题要穷尽到"哪个张量"级别的证据（调用栈、与卡数的关系、修复前后字节数对比）
4. rl_loop.sh 集成阶段的调试（enforce_format online 支持、N_TASKS→EASY_NUM_TASKS、eval-output 两层路径、vLLM 遗留进程清理、ITER 算术展开）共 5 个 bug，属自举链路，记录在 `rest_em_bootstrap.md` §5，此处不重复

---

## 3. 关联文档

| 文档 | 内容 |
|---|---|
| `rest_em_bootstrap.md` | 自举方法设计（信号体系/FORCE_THOUGHT）+ 集成调试 + 自举实验 |
| `cot_annotation_schemes.md` | CoT 标注方案（v1/v2/D）与终判 |
| `experiment_summary_20260905.md` | 全部实验的原始记录（含 RL 基础设施事故档案） |

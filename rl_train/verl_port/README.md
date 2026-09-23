# verl 移植资产（从 CrossAgent/MTRL 抢救，09-23）

> 来源：CrossAgent 项目（2025-09）的 verl fork，随 CrossAgent 整体删除而抢救至此。
> git 历史中完整原件仍在（删除前最后一个 commit）。移植目标：**上游 verl**
>（volcengine/verl 最新版）——本目录文件是待移植素材，不能直接 import
>（仍带 fork 的 `verl.` 前缀依赖）。

## 资产清单

| 目录/文件 | 内容 | 移植要点 |
|---|---|---|
| `model_adapter/qwen3_5.py`（127 行） | **Qwen3.5 hybrid 模型 verl 适配器**：Gated-DeltaNet 线性注意力 × full-attn 混合架构的 `forward_for_ppo` monkey patch，含 log_probs/entropy 输出、`apply_monkey_patch` 注册 | 我们模型（qwen35-9b-nf2 族）在 verl 里跑起来的关键。Ulysses SP 不支持（会显式报错）——FSDP 路径 |
| `model_adapter/test_qwen3_5_adapter.py` | 适配器结构冒烟测试：tiny 随机初始化模型 + 伪造多模态输入，验证 hybrid 层前向产出 log_probs/entropy 形状 | **本地可跑（CPU）**，是移植后的第一个验收步骤 |
| `env_minecraft/minecraft/`（3 文件） | Malmo 环境的 verl env 集成（envs.py + projection.py 动作投影） | 需对接我们当前 eval harness 的 Malmo 部署方式（MineStudio/minestudio 路径，见 envs.py 引用） |
| `gigpo/core_gigpo.py`（303 行） | **GiGPO**：组内嵌套组优势——episode 级 + step 级双层分组，step-level 组按观测哈希聚合 | 正中我们"episode 稀疏奖励下信度分配"软肋；是 EM 自举信号4（局部进展塑形）的 principled 替代 |
| `ref_configs/mc-mix_coa/*.sh` | 当年 Minecraft GRPO 的完整超参参考（qwen2-vl-7b 时代：lr 5e-6、KL 0.01 low_var_kl、dynamic_rollouts、TP=2、gpu_mem_util 0.4 等） | 迁移时换模型路径/环境配置，超参作起点 |

## 移植时的已知坑（从旧脚本和我们的经验推断）

1. **fla 内核**：我们的 koala 环境需 `LINEAR_ATTN_KERNELS=1` 装 flash-linear-attention + causal-conv1d（见 `rl_grpo_design.md` §1.4）；FSDP × fla 组合在 koala 环境从未验证过——这是移植的最大不确定点
2. **transformers 版本**：`Qwen3_5ForConditionalGeneration` 需较新 transformers；fork 的 requirements 与上游 verl 的版本约束需对齐
3. **旧脚本的集群路径**（`/share/hkc/...`、`MC-verl-agent/...`）全部失效，参考时只看超参

## 移植步骤（Phase 1 计划）

```
① pip install 上游 verl → 版本/API 摸底
② qwen3_5.py 移植到上游的 verl/models/transformers/ 注册路径 → 跑冒烟测试
③ minecraft env 对接我们的 Malmo（minestudio）部署
④ reward 函数：帧数 < 180 = 1（沿用现有 SUCCESS_MAX_FRAMES 判定）
⑤ 首个 GRPO：从 cot-pilot-v2/checkpoint-520 起步（与自研线同起点，可对照）
⑥ GiGPO 替换 GRPO（解决信度分配）
```

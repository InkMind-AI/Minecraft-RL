# verl 移植资产（从 CrossAgent/MTRL 抢救，09-23）

> 来源：CrossAgent 项目（2025-09）的 verl fork，随 CrossAgent 整体删除而抢救至此。
> git 历史中完整原件仍在（删除前最后一个 commit）。移植目标：**上游 verl**
>（verl-project/verl，23.6k★，现为 verl-project 组织维护）。

## ⚠ 09-23 移植评估结论（实测 diff 后更新，改变下列资产的定位）

对比上游 `verl/models/transformers/qwen3_5.py`（661 行）与抢救版（127 行）：

| 资产 | 上游现状 | 移植结论 |
|---|---|---|
| **qwen3_5 适配器** | ✅ **已被上游大幅超越**：packed sequence（remove-padding）、fla chunked Gated-DeltaNet 前向（`_packed_chunk_gated_delta_rule`——正是我们 test3-10 踩坑的内核路径）、多模态 embed、fused kernel 后端选择 | **抢救版作废，直接用上游**——模型后端（原以为最难的部分）移植成本归零 |
| minecraft env 包 | 上游无 Minecraft；社区 **verl-agent** 项目是 gym 风格 env + GiGPO 的正式新家 | 仍需移植，本目录 envs.py 作参考 |
| GiGPO core | ✅ verl-agent 项目收录 | 从 verl-agent 取，抢救版作算法阅读材料 |
| mc-mix_coa 参考配置 | 无对应 | 仍有用（超参起点） |

## 资产清单（保留作参考）

| 目录/文件 | 内容 |
|---|---|
| `model_adapter/qwen3_5.py`（127 行，**已被上游超越**） | fork 时代的 Qwen3.5 hybrid 适配器：Gated-DeltaNet × full-attn 的 `forward_for_ppo`。上游版本功能是其超集 |
| `model_adapter/test_qwen3_5_adapter.py` | 结构冒烟测试（tiny 模型 + 伪造多模态输入）——验收思路仍可复用 |
| `env_minecraft/minecraft/`（3 文件） | Malmo 的 verl env 集成（envs.py + projection.py 动作投影） |
| `gigpo/core_gigpo.py`（303 行，**上游已有**） | GiGPO：episode 级 + step 级嵌套组优势 |
| `ref_configs/mc-mix_coa/*.sh` | 当年 Minecraft GRPO 超参参考（lr 5e-6 / KL 0.01 low_var_kl / TP=2 / gpu_mem_util 0.4 等） |
| `ref_configs/mc-mix_coa/*.sh` | 当年 Minecraft GRPO 的完整超参参考（qwen2-vl-7b 时代：lr 5e-6、KL 0.01 low_var_kl、dynamic_rollouts、TP=2、gpu_mem_util 0.4 等） | 迁移时换模型路径/环境配置，超参作起点 |

## 移植时的已知坑（从旧脚本和我们的经验推断）

1. **fla 内核**：我们的 koala 环境需 `LINEAR_ATTN_KERNELS=1` 装 flash-linear-attention + causal-conv1d（见 `rl_grpo_design.md` §1.4）；FSDP × fla 组合在 koala 环境从未验证过——这是移植的最大不确定点
2. **transformers 版本**：`Qwen3_5ForConditionalGeneration` 需较新 transformers；fork 的 requirements 与上游 verl 的版本约束需对齐
3. **旧脚本的集群路径**（`/share/hkc/...`、`MC-verl-agent/...`）全部失效，参考时只看超参

## 移植步骤（Phase 1 计划，按上游评估结论更新）

```
① pip install 上游 verl → 验证 qwen3_5 适配器加载我们的 checkpoint（真模型冒烟，集群跑）
② minecraft env：参考本目录 envs.py + verl-agent 的 env 接口，对接我们的 Malmo 部署
③ reward 函数：帧数 < 180 = 1（沿用现有 SUCCESS_MAX_FRAMES 判定）
④ 首个 GRPO：从 cot-pilot-v2/checkpoint-520 起步（与自研线同起点，可对照）
⑤ GiGPO 替换 GRPO（从 verl-agent 取，解决信度分配）
```

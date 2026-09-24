# verl 迁移抢救资产（从 CrossAgent/MTRL 抢救，09-23；09-24 瘦身）

> 来源：CrossAgent 项目（2025-09）的 verl fork，随 CrossAgent 整体删除而抢救。
> git 历史中完整原件仍在（删除前最后一个 commit）。
> **09-24 更新**：迁移基座定为 **verl-agent**（GiGPO 官方库，内嵌完整 verl +
> agent_system 多轮环境框架），抢救资产中已被上游取代的已从工作区删除，
> 只留上游不存在的两样。

## 当前保留（均为上游不存在、不可 clone 的私有资产）

| 目录 | 内容 | 用途 |
|---|---|---|
| `env_minecraft/minecraft/`（3 文件） | Malmo 的 env 集成：envs.py（minestudio 对接）+ projection.py（动作投影） | **verl-agent 的 env_package/ 布局与此同源**——直接放进其 `agent_system/environments/env_package/` 即可 |
| `ref_configs/mc-mix_coa/*.sh` | 当年 Minecraft GRPO 真实超参（lr 5e-6 / KL 0.01 low_var_kl / TP=2 / gpu_mem_util 0.4 / dynamic_rollouts） | 超参起点 |

## 已删除（上游有更好的，git 历史可找回）

| 原资产 | 上游位置 |
|---|---|
| `model_adapter/qwen3_5.py`（127 行） | verl-project/verl 的 `verl/models/transformers/qwen3_5.py`（661 行，含 packed seq / fla chunked Gated-DeltaNet / 多模态 embed） |
| `model_adapter/test_qwen3_5_adapter.py` | 上游 verl 自带测试 |
| `gigpo/core_gigpo.py`（303 行） | verl-agent 官方仓库根目录 `gigpo/`（NeurIPS 2025 论文官方实现） |

## verl-agent 基座的关键情报（09-24 查证）

- 完整 vendor verl（2025-06 合并最新版特性）+ `agent_system/` 多轮环境框架 + GiGPO
- 环境清单：ALFWorld / WebShop / Search / Sokoban / Gym Cards / AppWorld——**无 Minecraft**（我们的 env_minecraft 是唯一现成实现）
- **待验证坑**：其内嵌 verl 是 2025-06 版，可能不含较新的 qwen3_5 适配器——
  若缺，从上游 verl 拷 `qwen3_5.py` 单文件进 vendor（迁移第 1 步冒烟时确认）

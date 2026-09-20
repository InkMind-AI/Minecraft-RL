"""GRPO 组内优势计算（RL 核心数学，纯 numpy，无框架依赖，可单测）。

设计（对应 09-20 的自研 RL 决策）：
- 无 value model（选 GRPO 而非 PPO 的核心理由）
- episode 级二值奖励（任务成功 1/0），组 = 同一任务的 G 个 rollout
- advantage 在数据集构建阶段离线算好（写进 parquet 旁车文件），trainer 只读
  ——trainer 极简、advantage 逻辑独立可测

两种归一化：
- std 归一化（原版 GRPO）：A = (r - mean) / (std + eps)
- Dr.GRPO（去长度偏差）：A = r - mean，token 级损失按 token 数归一
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-6


def compute_group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
    mode: str = "std",
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """按组计算 GRPO 优势。

    参数:
        rewards: 每条 episode 的奖励（0/1 或连续值），与 group_ids 一一对应
        group_ids: 每条 episode 的组标识（通常是任务名）
        mode: "std"（原版 GRPO）或 "dr"（Dr.GRPO，无 std 除法）

    返回:
        (advantages, group_stats)
        零方差组的成员 advantage = 0（GRPO 惯例：全成功/全失败的组无学习信号）
    """
    if len(rewards) != len(group_ids):
        raise ValueError(f"rewards({len(rewards)}) 与 group_ids({len(group_ids)}) 长度不一致")

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[str(g)].append(i)

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    advantages = np.zeros(len(rewards), dtype=np.float64)
    stats: Dict[str, dict] = {}

    for g, idxs in groups.items():
        r = rewards_arr[idxs]
        mean = float(r.mean())
        std = float(r.std())
        stats[g] = dict(n=len(idxs), mean=mean, std=std,
                        n_success=int((r > 0).sum()) if set(np.unique(r)) <= {0.0, 1.0} else None)
        if std < EPS:
            # 全成功 / 全失败：无相对信号
            continue
        if mode == "std":
            a = (r - mean) / (std + EPS)
        elif mode == "dr":
            a = r - mean
        else:
            raise ValueError(f"未知 mode: {mode}")
        advantages[idxs] = a

    return advantages, stats


def group_coverage_report(stats: Dict[str, dict]) -> Dict[str, float]:
    """数据利用效率报告：多少组有非零信号（可指导 rollout 配置——G 太小则大多组零方差）。"""
    if not stats:
        return dict(n_groups=0, usable_frac=0.0)
    usable = [s for s in stats.values() if s["std"] >= EPS]
    return dict(
        n_groups=len(stats),
        usable_frac=round(len(usable) / len(stats), 3),
        mean_group_size=round(np.mean([s["n"] for s in stats.values()]), 2),
    )

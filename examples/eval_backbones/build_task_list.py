#!/usr/bin/env python3
"""
生成与论文(OpenHA, arXiv:2509.13347)Embodied / GUI / Combat 三大类任务体系对齐的
评测任务列表。

任务类别与本仓库任务生成器的对应关系(见 openagents/envs/tasks/task_manager.py)：
    Embodied = mine_block:*                                  (导航+挖掘/砍伐)
    Combat   = kill_entity:*  (仅取 spawn 配置里 label 含 "normal" 的条目)
    GUI      = craft_item:* / smelt_item:* / interact_with_*  (工作台/炼炉等GUI交互)

用法：
    python build_task_list.py --scope mini            # 30个代表性任务：每类10个，
                                                        # 按 easy/middle/hard 均分
    python build_task_list.py --scope full             # 全部 800+ 个任务(单一难度)
    python build_task_list.py --scope easy             # 非 GUI 任务(Embodied 153 +
                                                        # Combat 49，共202个)全量，
                                                        # 统一 difficulty=easy，可用
                                                        # --num_tasks/--easy_difficulty 调整

输出（stdout）：空格分隔的 "task_name@difficulty" token 序列，可直接赋给
run_backbone_eval.sh 的 TASK_DIFFICULTY_LIST 环境变量。若设置了环境变量
TASK_LIST_MANIFEST，还会额外写一份 JSON 明细（含 category 字段）方便留档/复现。

注意：仅依赖标准库 + 读取 openagents/assets/spawns/*.json，不需要 conda env /
项目其他依赖，可以在 run_backbone_eval.sh 里 conda 环境准备好之前就调用。
"""
import argparse
import json
import os
import random

SPAWN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "openagents", "assets", "spawns"
)


def load_tasks_by_category():
    def _load(name):
        with open(os.path.join(SPAWN_DIR, name), encoding="utf-8") as f:
            return json.load(f)

    mine_block = list(_load("mine_block.json").keys())
    kill_entity_raw = _load("kill_entity.json")
    kill_entity = [k for k, v in kill_entity_raw.items() if "normal" in v.get("label", "")]
    craft_item = list(_load("craft_item.json").keys())
    smelt_item = list(_load("smelt_item.json").keys())
    interact_block = list(_load("interact_block.json").keys())

    return {
        "Embodied": sorted(mine_block),
        "Combat": sorted(kill_entity),
        "GUI": sorted(set(craft_item) | set(smelt_item) | set(interact_block)),
    }


def pick_mini(tasks_by_cat, per_category, seed):
    rng = random.Random(seed)
    levels = ["easy", "middle", "hard"]
    picked = []
    for cat, tasks in tasks_by_cat.items():
        n = min(per_category, len(tasks))
        chosen = rng.sample(tasks, n)
        for i, t in enumerate(chosen):
            picked.append((cat, t, levels[i % len(levels)]))
    return picked


def pick_full(tasks_by_cat, difficulty):
    return [(cat, t, difficulty) for cat, tasks in tasks_by_cat.items() for t in tasks]


def pick_easy(tasks_by_cat, num_tasks, difficulty, seed):
    """从非 GUI 类别(Embodied/Combat)采样 num_tasks 个任务，统一使用同一难度。

    用于"大量简单任务"型高统计功效评测（如 300 任务 x 3 rollout 全 easy）。
    GUI 类完全排除：当前所有 checkpoint 的 GUI 成功率均为 0%，纳入只会浪费
    rollout 且不产生区分信号。非 GUI 池共 202 个任务(Embodied 153 + Combat 49)，
    num_tasks 超过池大小时自动取全量。若未来 GUI 能力不再为 0，应恢复纳入。
    """
    rng = random.Random(seed)
    pool = [(cat, t) for cat, tasks in tasks_by_cat.items() if cat != "GUI" for t in tasks]
    chosen = rng.sample(pool, min(num_tasks, len(pool)))
    return [(cat, t, difficulty) for cat, t in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["mini", "full", "easy"], default="mini")
    ap.add_argument("--per_category", type=int, default=10, help="mini scope: 每类任务数")
    ap.add_argument("--seed", type=int, default=42, help="mini/easy scope: 任务采样随机种子(保证可复现)")
    ap.add_argument("--full_difficulty", default="normal", help="full scope: 所有任务统一难度")
    ap.add_argument("--num_tasks", type=int, default=300, help="easy scope: 采样任务总数")
    ap.add_argument("--easy_difficulty", default="easy", help="easy scope: 所有任务统一难度")
    args = ap.parse_args()

    tasks_by_cat = load_tasks_by_category()

    if args.scope == "mini":
        picked = pick_mini(tasks_by_cat, args.per_category, args.seed)
    elif args.scope == "easy":
        picked = pick_easy(tasks_by_cat, args.num_tasks, args.easy_difficulty, args.seed)
    else:
        picked = pick_full(tasks_by_cat, args.full_difficulty)

    print(" ".join(f"{t}@{d}" for _, t, d in picked))

    manifest_path = os.environ.get("TASK_LIST_MANIFEST")
    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"category": c, "task": t, "difficulty": d} for c, t, d in picked],
                f,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()

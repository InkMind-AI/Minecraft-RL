"""episode → RL 训练批次适配器（阶段 2，rl_loop.sh 的 ② 环节）。

数据源（每个 episode 目录）：
  episode.jsonl   逐帧 PNG（base64，360x640）——观测流
  raw_action.jsonl 逐步模型输出（"Thought: ...\nAction: ..." 或纯 "Action: ..."）
  帧数 < 早停阈值 → 成功（评测 harness 在任务完成时提前终止 episode）

输出：
  --out     parquet（与 SFT 训练格式同构：conversations + image_bytes + id）
  --rewards npy（成功 1 / 失败 0，与 parquet 行序对齐）
  --groups  json（任务名列表，与 parquet 行序对齐）

⚠ 指令文本保真：训练 parquet 的首 user turn = 系统提示 + "## User Instruction\n\n<指令>"。
评测时的指令由 openagents 从任务名生成。本适配器从评测 manifest 取任务名并用同一
模板重建；上线前必须抽样对比重建指令与训练数据中同任务指令的逐字一致性（格式失配
会让 RL 学到错误条件分布——见 train_stage3.sh 头部关于 byte-identical 的教训）。
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 早停判定：评测 harness 任务完成即终止（MAX_STEPS_NUM=200）
SUCCESS_MAX_FRAMES = 180

INSTRUCTION_TEMPLATES = {
    "kill_entity": "Kill the {name}.",
    "mine_block": "Mine the {name}.",
    # ⚠ 待办：对照 openagents/assets/instructions.json 补全模板并逐字校验
}


def task_instruction(task_name: str) -> str:
    kind, _, name = task_name.partition(":")
    name = name.replace("_", " ")
    tpl = INSTRUCTION_TEMPLATES.get(kind, "{name}.")
    return tpl.format(name=name)


def load_episode(episode_dir: str):
    """返回 (frames[bytes], responses[str], n_frames)。"""
    frames = []
    ep_path = os.path.join(episode_dir, "episode.jsonl")
    with open(ep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(base64.b64decode(d["base64"]))
    responses = []
    with open(os.path.join(episode_dir, "raw_action.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ra = json.loads(line).get("raw_action", "")
            if ra:
                responses.append(ra)
    return frames, responses


def build_row(task_name: str, episode_id: str, frames, responses, system_prompt: str):
    """构建与 SFT 训练数据同构的行：交替 user(帧)/assistant(输出)。"""
    instruction = task_instruction(task_name)
    first_user_text = f"{system_prompt}\n\n## User Instruction\n\n{instruction}"
    conversations = [{
        "role": "user",
        "content": [{"type": "text", "text": first_user_text},
                    {"type": "image"}],
    }]
    for i, resp in enumerate(responses):
        if i > 0:
            conversations.append({
                "role": "user",
                "content": [{"type": "image"}],
            })
        conversations.append({
            "role": "assistant",
            "content": [{"type": "text", "text": resp}],
        })
    return dict(id=f"rl_{episode_id}", label=["text_action"],
                conversations=conversations,
                image_bytes=[f for f in frames[:len(responses)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-output", required=True,
                    help="eval_output/<MODEL_LOCAL_NAME>-text_action 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--system-prompt",
                    default=os.path.join(os.path.dirname(__file__),
                                         "..", "openagents", "assets",
                                         "system_prompt", "text_action.txt"))
    args = ap.parse_args()

    system_prompt = open(args.system_prompt).read().strip()
    rows, rewards, groups = [], [], []
    for task_dir in sorted(os.listdir(args.eval_output)):
        task_path = os.path.join(args.eval_output, task_dir)
        if not os.path.isdir(task_path):
            continue
        for ep in sorted(os.listdir(task_path)):
            ep_dir = os.path.join(task_path, ep)
            if not os.path.isdir(ep_dir):
                continue
            try:
                frames, responses = load_episode(ep_dir)
            except FileNotFoundError:
                continue
            if len(frames) < 5 or not responses:
                continue
            rows.append(build_row(task_dir, ep, frames, responses, system_prompt))
            rewards.append(1.0 if len(frames) < SUCCESS_MAX_FRAMES else 0.0)
            groups.append(task_dir)

    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, args.out, row_group_size=64)
    np.save(args.rewards, np.array(rewards, dtype=np.float64))
    json.dump(groups, open(args.groups, "w"))
    r = np.array(rewards)
    print(f"批次: {len(rows)} episodes | 成功率 {100*r.mean():.1f}% | "
          f"任务数 {len(set(groups))}")


if __name__ == "__main__":
    main()

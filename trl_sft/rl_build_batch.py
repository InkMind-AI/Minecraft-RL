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


def build_row(task_name: str, episode_id: str, frames, responses, system_prompt: str,
             window_start: int = 0):
    """构建与 SFT 训练数据同构的行：交替 user(帧)/assistant(输出)。

    ⚠ system_prompt 文件本身已以 "## User Instruction\n\n" 结尾（占位符），
    此处直接续接指令文本，不可重复拼接该标题——曾在冒烟测试中实测捕获重复
    bug（见 09-20 测试记录），保留本注释防止回归。

    window_start > 0 表示这是 episode 内某个窗口（见 windows_from_episode），首帧
    不再是任务真正的起点，指令后附加位置说明，避免模型误以为每个窗口都是新任务。
    """
    instruction = task_instruction(task_name)
    if window_start > 0:
        instruction += f" (continuing, step {window_start})"
    first_user_text = f"{system_prompt}{instruction}"
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
    return dict(id=f"rl_{episode_id}_w{window_start}", label=["text_action"],
                conversations=conversations,
                image_bytes=[f for f in frames[:len(responses)]])


def windows_from_episode(frames, responses, window: int, min_window: int = 5):
    """把长 episode 切成 <=window 步的不重叠窗口（默认对齐 h29 推理历史长度，
    见 09-20 冒烟测试：200 步整段轨迹会被 dataset.py 的 _exceeds_max_length
    安全过滤器丢弃，且成功/失败 episode 天然长度不同，整段喂入会系统性偏向
    保留成功样本、丢弃失败样本，破坏 GRPO 组内对比结构——切窗后每个窗口独立
    通过长度检查，且继承整个 episode 的奖励，成败样本存活率恢复一致）。

    最后一个不足 min_window 步的残余窗口丢弃（信息量太低，且过短窗口的
    prefix-cache 收益也低）。
    """
    n = min(len(frames), len(responses))
    out = []
    for s in range(0, n, window):
        e = min(s + window, n)
        if e - s < min_window:
            break
        out.append((s, frames[s:e], responses[s:e]))
    return out


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
    ap.add_argument("--window", type=int, default=29,
                    help="切窗步数，对齐 MAXIMUM_HISTORY_LENGTH（h29）——见"
                         " windows_from_episode 的长度过滤事故记录")
    args = ap.parse_args()

    # ⚠ 不可 .strip()：权威拼接方式见 openha.py:325 `self.system_message + instruction`
    # （self.system_message = 原始 f.read()，未 strip）。文件本身以恰好一个 "\n" 结尾
    # （"...## User Instruction\n"），指令直接续接、中间无额外分隔——strip() 会吃掉这个
    # 换行导致标题与指令粘连。09-20 冒烟测试曾捕获此 bug，务必保留本注释防止回归。
    system_prompt = open(args.system_prompt).read()
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
            episode_reward = 1.0 if len(frames) < SUCCESS_MAX_FRAMES else 0.0
            # 窗口继承整个 episode 的奖励与组标签——切窗只为绕开序列长度限制，
            # 不改变 GRPO 的信度分配粒度（仍是 episode 级，见设计文档）
            for wstart, wframes, wresponses in windows_from_episode(
                    frames, responses, args.window):
                rows.append(build_row(task_dir, ep, wframes, wresponses,
                                      system_prompt, window_start=wstart))
                rewards.append(episode_reward)
                groups.append(task_dir)

    tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, args.out, row_group_size=64)
    np.save(args.rewards, np.array(rewards, dtype=np.float64))
    json.dump(groups, open(args.groups, "w"))
    r = np.array(rewards)
    n_episodes = len(set(f"{g}_{rid.rsplit('_w',1)[0]}" for g, rid in
                        zip(groups, [row["id"] for row in rows])))
    print(f"批次: {len(rows)} 窗口样本（来自 ~{n_episodes} episodes）| "
          f"窗口级成功率 {100*r.mean():.1f}% | 任务数 {len(set(groups))}")


if __name__ == "__main__":
    main()

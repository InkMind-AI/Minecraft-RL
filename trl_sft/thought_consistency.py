"""ReST-EM 自举 MVP：信号2（言行一致）打分 + 决策点感知的 thought 剥离。

对应设计讨论《打分筛选应该怎么设计》的落地（先只做信号2+信号3，信号1真值
verifier 和信号4局部进展留待后续版本）：

- **决策点感知剥离**：rollout 时用 FORCE_THOUGHT=1 让模型每一步都被强制生成
  "Thought: ..."（见 run_backbone_eval.sh），但只有回溯用 decision_points.py
  （对真实发生的动作序列做检测，v2 冻结算法，触发率 12.4%）判定为决策点的
  步骤才保留 Thought；非决策点即使被强制生成了也剥回纯 "Action: ..."——
  这样构造的训练分布密度与人工标注一致，不引入新的分布偏移。
- **信号2打分（言行一致）**：只在能从 thought 明确解析出意图类型（攻击/交互/
  移动/停止）的样本上打分，查不出关键词的返回 None、不参与打分——宁可不判，
  不可错判（对应设计讨论里"查不出来的不打分"的原则）。
- 信号2的分数最终会在 rl_loop.sh 里与信号3（GRPO 组内 episode 优势）相乘，
  作为该训练窗口的最终权重，train_grpo.py 不需要任何改动（复用现有的单一
  advantages 数组机制）。
"""
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cot_annotation.decision_points import thought_points, parse_action  # noqa: E402

# 两种真实格式都要兼容：
# ① 人工标注训练数据（v2 等）：字面 "Thought: xxx\nAction: yyy"
# ② FORCE_THOUGHT=1 的 rollout 输出（09-22 实测确认）：continue_final_message
#    机制下 vLLM 只回传续写的新 token，不包含被注入的 "Thought: " 前缀字面文本
#    ——所以真实拿到的是 "xxx\nAction: yyy"（没有开头的 "Thought:"）。
RESPONSE_RE = re.compile(r"^Thought:\s*(?P<thought>.*?)\s*\n(?P<action>Action:.*)$",
                         re.DOTALL)
IMPLICIT_RESPONSE_RE = re.compile(r"^(?P<thought>.+?)\s*\n(?P<action>Action:.*)$", re.DOTALL)
ACTION_ONLY_RE = re.compile(r"^(?P<action>Action:.*)$", re.DOTALL)

# 词根匹配（\w* 而非结尾 \b）：需覆盖 "attacking"/"mined"/"approaching" 等语态
# 变形，纯 \b...\b 会因为末尾紧跟 ing/ed 而匹配失败（09-22 单测实测踩过这个坑）。
ATTACK_WORDS = re.compile(r"\b(attack|hit|strik|kill|min(?!or)|break|chop)\w*", re.I)
INTERACT_WORDS = re.compile(r"\b(us(?:e|ing)|place|open|craft|interact|eat|drink|equip)\w*", re.I)
FORWARD_WORDS = re.compile(
    r"\b(forward|approach|closer|advanc|chas|follow)\w*|walk toward|move toward", re.I)
STOP_WORDS = re.compile(r"\b(stop|stand still|wait|paus|remain still|hold position)\w*", re.I)


def split_response(resp: str) -> Tuple[Optional[str], str]:
    """把一步的原始输出拆成 (thought_text_or_None, action_text)。

    依次尝试：显式 "Thought:" 前缀 → 隐式（"Action:" 前有非空正文，但没有
    字面 "Thought:"，即 FORCE_THOUGHT rollout 的真实格式）→ 纯 "Action: ..."
    （非决策点、模型没被强制想的正常单行输出）。三种都不匹配（异常/空响应）
    时退化为无 thought，整段当 action 文本（parse_action 内部用 .search 不
    是 .match，仍能在异常文本里找到 move/press/click，不会崩）。
    """
    resp = resp.strip()
    m = RESPONSE_RE.match(resp)
    if m:
        return m.group("thought") or None, m.group("action").strip()
    m = IMPLICIT_RESPONSE_RE.match(resp)
    if m and m.group("thought").strip():
        return m.group("thought").strip(), m.group("action").strip()
    m = ACTION_ONLY_RE.match(resp)
    if m:
        return None, m.group("action").strip()
    return None, resp


def action_consistency_score(thought: str, action_text: str) -> Optional[float]:
    """信号2：言行一致性打分，返回 [0,1] 或 None（无法判定时不参与打分）。

    两个独立轴（缺一不影响另一个）：
    - click 轴：thought 提到攻击类词 → 期望 left-click；提到交互类词 → 期望
      right-click
    - 移动轴：thought 提到"前进/靠近/追" → 期望 w 键；提到"停/等" → 期望 no-op
    命中数/检查数 = 分数；一个关键词都没命中时返回 None。
    """
    sem = parse_action(action_text)
    checks, hits = 0, 0

    wants_attack = bool(ATTACK_WORDS.search(thought))
    wants_interact = bool(INTERACT_WORDS.search(thought))
    if wants_attack != wants_interact:  # 恰好命中一类，语义明确才判
        checks += 1
        if wants_attack and sem.click == "L":
            hits += 1
        elif wants_interact and sem.click == "R":
            hits += 1

    wants_forward = bool(FORWARD_WORDS.search(thought))
    wants_stop = bool(STOP_WORDS.search(thought))
    if wants_forward != wants_stop:
        checks += 1
        # 注意：真实空动作文本是 "move(0,0) and press()"（空 keys），不是字面
        # "no_op"（那是 parse_action.is_noop 认的另一种格式，此仓库训练数据
        # 不用），所以停止判定看 keys 和 click 是否都空，不能用 sem.is_noop。
        no_op = (not sem.keys) and sem.click is None
        if wants_forward and "w" in sem.keys:
            hits += 1
        elif wants_stop and no_op:
            hits += 1

    if checks == 0:
        return None
    return hits / checks


def strip_and_score(responses: List[str]) -> Tuple[List[str], Dict[int, Optional[float]]]:
    """对一整段 episode 的原始输出做决策点感知剥离 + 信号2打分。

    返回 (cleaned_responses, decision_step_scores)：
    - cleaned_responses: 与输入等长；非决策点已剥成纯 Action（即便原文有 Thought）
    - decision_step_scores: {episode 内步索引 -> score_or_None}，只含决策点
    """
    parsed = [split_response(r) for r in responses]
    action_texts = [a for _, a in parsed]
    dpts = set(thought_points(action_texts))

    cleaned: List[str] = []
    scores: Dict[int, Optional[float]] = {}
    for i, (thought, action) in enumerate(parsed):
        if i in dpts and thought:
            cleaned.append(f"Thought: {thought}\n{action}")
            scores[i] = action_consistency_score(thought, action)
        else:
            cleaned.append(action)
    return cleaned, scores

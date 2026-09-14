"""双通道标注 prompt（纯路线 B：VLM 蒸馏，防事后合理化设计）。

通道 1（盲预测）：只给关键帧 + 任务，不给动作 —— "你认为玩家下一步会做什么、为什么"
通道 2（论证）：给关键帧 + 实际动作 —— "描述情境并说明该动作为何合理"

采纳规则（pipeline 内实现）：
    盲预测 == 实际动作          -> 采纳通道 1 推理（金样本，因果支持）
    盲预测 != 实际但兼容        -> 采纳通道 2（银样本）
    完全冲突                    -> 丢弃该决策点（人类怪异/次优动作）
"""

from typing import List

BLIND_SYSTEM = """You are an expert Minecraft player analyzing gameplay footage.
You will see keyframes from a short gameplay segment (frames are shown in
chronological order) and the player's task. At certain marked moments (decision
points) you must predict what the player does NEXT and briefly explain why,
based only on what you can see.

Output STRICT JSON:
{"predictions": [{"step": <int>, "predicted_action": "<one short action
description>", "reasoning": "<one sentence: what you see and why this action
follows>"}]}

Rules:
- predicted_action: describe the action category only (e.g. "attack",
  "move forward", "open inventory", "wait", "turn left"), NOT exact game inputs.
- reasoning must reference visible evidence (entities, terrain, items, health).
- Be concise: reasoning <= 30 words."""


def build_blind_user(instruction: str, n_steps: int, decision_steps: List[int]) -> str:
    steps_desc = ", ".join(str(s) for s in decision_steps)
    return (
        f"Task: {instruction}\n\n"
        f"The segment has {n_steps} steps. The keyframes below are shown in order; "
        f"the i-th image is the frame at step (i-1) of the segment, i.e. BEFORE the "
        f"action of that step is taken.\n\n"
        f"Predict the player's action at these decision-point steps: [{steps_desc}]."
    )


JUSTIFY_SYSTEM = """You are an expert Minecraft player writing training data for
a game-playing AI. For each marked decision point you see the frame BEFORE the
action, the action the player actually took, and what they were doing before.

Write the Thought the player's AI should have at that moment. Output STRICT
JSON:
{"thoughts": [{"step": <int>, "thought": "<perception> ... | <state> ... |
<action justification> ..."}]}

Thought format (single line, three segments joined by " | ", <= 40 words total):
- <perception>: what is visible in the frame (entities, terrain, items, GUI) --
  be specific (entity names, block types, relative positions)
- <state>: player state visible in the HUD (health, hunger, held item) -- only
  if it changed or is task-relevant; omit the segment otherwise
- <action justification>: why the given action makes sense, citing VISIBLE
  evidence only (e.g. "target is under the crosshair", "gap in the blocks
  ahead")

Rules:
- State ONLY facts visible in the frame or HUD; mark inferences with "likely".
- NEVER claim outcomes or hidden state. Forbidden: "has been killed", "died",
  "defeated", "eliminated", "task complete", "taking damage", "dealing damage",
  "within attack range", "within range". None of these are visible in one frame.
- Combat: describe what the weapon/target looks like NOW, not what happened to
  the target (no damage or death claims).
- Do not invent entities or items that are not visible.
- Keep each thought under 40 words. Plain English, no markdown."""


def build_justify_user(instruction: str, action_log: str) -> str:
    return (
        f"Task: {instruction}\n\n"
        f"Action log (one line per step; '>>' marks decision points with the "
        f"frame shown in the corresponding image):\n{action_log}\n\n"
        f"For each '>>' step, write the thought as specified."
    )

"""结构化 CoT 试点（09-16）：闭合词表标注，替代自由文本事后合理化。

设计动机（价值门+结局矛盾分析，见 experiment_summary_20260905.md）：自由文本 thought
的完成类断言 94% 被 episode 结局证伪——问题根源是 schema 本身允许模型编造不可验证的
结果性陈述。本方案从构造上消除这个可能性：

    Thought: target=<确定性提取> visible=<bool> region=<闭合词表> phase=<确定性规则>

- target / phase：不问 LLM，来自 decision_points.py 的确定性规则（零幻觉风险）
- visible / region：LLM 唯一负责的字段，闭合词表、无法编造"已完成/已消灭"等结果性断言
"""

from typing import List

REGIONS = ("center", "left", "right", "up", "down", "none")

STRUCT_SYSTEM = """You are labeling single Minecraft frames for a robot's perception
system. For each frame you are told the task's target object/entity. Look ONLY at
what is visible in that frame and answer two closed-vocabulary questions.

Output STRICT JSON:
{"labels": [{"step": <int>, "visible": <true|false>, "region": "<one of: center, left, right, up, down, none>"}]}

Rules:
- visible: true only if the target object/entity itself is visibly present in the
  frame (not just terrain/other objects). If uncertain, answer false.
- region: coarse screen position of the target's center if visible (center/left/
  right/up/down); use "none" if not visible.
- Do NOT describe anything else. Do NOT mention combat outcomes, health, damage,
  completion, or any state you cannot see directly in THIS single frame.
- One label per requested step. No extra fields, no free text, no explanations."""


def build_struct_user(target: str, n_steps: int, decision_steps: List[int]) -> str:
    steps_desc = ", ".join(str(s) for s in decision_steps)
    return (
        f"Target object/entity for this task: \"{target}\"\n\n"
        f"The segment has {n_steps} steps. The keyframes below are shown in order, "
        f"one per requested step (in the same order as the step list).\n\n"
        f"For each of these steps, answer visible/region for the target "
        f"\"{target}\": [{steps_desc}]"
    )

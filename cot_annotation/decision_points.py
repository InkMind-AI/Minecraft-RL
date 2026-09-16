"""决策点检测（冻结 v2 算法，2026-09-07 用 400 条真实轨迹标定，触发率 12.4%）。

规则（任一满足即触发，另有突发合并与首步恒标注）：
    R1 按键集合变化：keys(t) != keys(t-1)（frozenset 比较，顺序无关）
    R2 点击状态变化：click(t) != click(t-1) ∈ {None, 'L', 'R'}
    R3 主动性翻转：  is_noop(t) != is_noop(t-1)
刻意不触发（噪声）：
    - 完全相同的动作延续
    - 纯鼠标幅度微调（同桶内的 dx/dy 抖动，实测占文本切换的 63.5%）
可选（pilot 消融开关，默认关）：
    R4 鼠标幅度跨 >=2 级跳变（捕捉"发现目标猛转头"）
"""

import re
from typing import List, Tuple, FrozenSet, Optional, NamedTuple

ACT_RE = re.compile(r"move\((-?\d+),\s*(-?\d+)\)\s+and\s+press\(([^)]*)\)")


class ActionSem(NamedTuple):
    """动作的规范语义元组。"""
    is_noop: bool
    keys: FrozenSet[str]
    click: Optional[str]        # None | 'L' | 'R'
    bucket: int                 # 鼠标幅度桶 0/1/2/3（0=不动,1=<10,2=10-40,3=>40）


def parse_action(s: str) -> ActionSem:
    """把动作文本解析成 ActionSem。no_op 返回特殊态。"""
    s = s.strip()
    if s in ("no_op", "Action: no_op"):
        return ActionSem(True, frozenset(), None, 0)
    m = ACT_RE.search(s)
    dx, dy = 0, 0
    keys: FrozenSet[str] = frozenset()
    if m:
        dx, dy = int(m.group(1)), int(m.group(2))
        keys = frozenset(k.strip() for k in m.group(3).split(",") if k.strip())
    click = None
    if "click" in s:
        click = "L" if "left" in s else "R"
    mag = max(abs(dx), abs(dy))
    bucket = 0 if mag == 0 else (1 if mag < 10 else (2 if mag <= 40 else 3))
    return ActionSem(False, keys, click, bucket)


def _triggers(p: ActionSem, c: ActionSem, use_r4: bool) -> bool:
    """R1/R2/R3（可选 R4）判定：前一动作 p -> 当前动作 c 是否构成决策点。"""
    if p == c:
        return False
    keys_ch = p.keys != c.keys                      # R1（含 R3：noop 的 keys 为空集，
    click_ch = p.click != c.click                   #  noop<->act 必然 keys 不同）
    r4 = use_r4 and abs(p.bucket - c.bucket) >= 2   # R4（默认关）
    return keys_ch or click_ch or r4


def thought_points(actions: List[str], min_gap: int = 2, use_r4: bool = False) -> List[int]:
    """返回需要标注 thought 的步索引（在最终可见序列上检测）。

    - 首步恒标注（初始决策）
    - min_gap：距上一个触发点 <= min_gap 步的新触发被合并（战斗突发抑制）
    """
    if not actions:
        return []
    pts = [0]
    last = 0
    sems = [parse_action(a) for a in actions]
    for t in range(1, len(sems)):
        if _triggers(sems[t - 1], sems[t], use_r4) and t - last > min_gap:
            pts.append(t)
            last = t
    return pts


# ─── 确定性 phase 分类（09-16，结构化 CoT 试点：零 LLM、零幻觉风险）───────────
# 只依赖动作语义本身，不依赖任何视觉/结果判断，因此 100% 可复现、不存在标注对错。
PHASE_ORDER = ("engage", "collect", "interact", "approach", "reposition", "wait")


def action_phase(sem: ActionSem) -> str:
    """把单步动作语义映射到闭合词表的 phase 标签（确定性规则，非 LLM）。"""
    if sem.is_noop:
        return "wait"
    if sem.click == "L":
        return "engage"       # 左键：攻击/挖掘
    if sem.click == "R":
        return "interact"     # 右键：使用/放置/交互
    moving = bool(sem.keys & {"w", "a", "s", "d"})
    if moving and sem.bucket >= 2:
        return "approach"     # 移动 + 较大视角调整：朝目标靠近/搜索
    if moving:
        return "reposition"   # 移动但视角变化小：走位微调
    return "reposition"       # 纯视角调整、无移动无点击


def extract_target(instruction: str) -> str:
    """从任务指令确定性提取目标名词短语（正则，非 LLM，零幻觉风险）。

    真实指令句式多样（"Break the brewing stand to collect it."/"Mine the jungle
    log in the rainforest biome."），核心名词短语夹在动词短语与从句/介词短语之间。
    策略：动词+冠词后，抓 1-4 个词，遇到从句连接词（to/for/from/in/at/using/by/
    into/on/with/back/at）或句末即停止。匹配失败时退化为整句下划线化（宁可噪声
    大，不可静默失败——下游训练侧目标不精确不影响 phase/visible 两个核心字段）。
    """
    s = instruction.strip().rstrip(".")
    m = re.search(
        r"(?:kill|mine|break|attack|collect|gather|get|harvest|recycle|craft|"
        r"create|open|use|smelt|find|approach)\w*\s+"
        r"(?:the\s+|a\s+|an\s+|some\s+)?"
        r"([a-zA-Z_]+(?:\s+[a-zA-Z_]+){0,3}?)"
        r"(?=\s+(?:to|for|from|in|at|using|by|into|on|with|back|and)\b|$)",
        s, re.I)
    if m and m.group(1).strip().lower() not in ("it", "them", "one"):
        return m.group(1).strip().lower().replace(" ", "_")
    return re.sub(r"\s+", "_", s.lower())


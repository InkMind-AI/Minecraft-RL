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

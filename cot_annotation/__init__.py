"""CoT 标注管线（v1，2026-09-07）。

链路：parquet 轨迹 → 决策点检测（R1/R2/R3，冻结 v2 算法）
    → per-chunk 双通道 payload（盲预测 + 论证） → 可插拔标注器（Stub/OpenAI/Gemini）
    → 校验渲染 → 输出带 Thought 的训练数据。

决策点判定（已在真实数据上标定，触发率 12.4%）：
    R1 按键集合变化 | R2 点击状态变化 | R3 no-op↔动作转换
    + 首步恒标注 + min_gap=2 突发合并。R4（鼠标跨级跳变）留作 pilot 消融开关。
"""
from .decision_points import parse_action, thought_points
from .pipeline import run_pipeline

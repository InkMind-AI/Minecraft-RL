"""grpo_core 单元测试（纯 numpy，本地可跑）。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

from grpo_core import compute_group_advantages, group_coverage_report  # noqa: E402


def test_basic_group_centering():
    """组内均值应为 0（std 模式），全成功/全失败组 advantage=0。"""
    rewards = [1, 0, 1, 1, 0, 0]
    groups = ["A", "A", "A", "B", "B", "B"]
    adv, stats = compute_group_advantages(rewards, groups, mode="std")
    # 组 A: [1,0,1] mean=2/3 std≈0.471 → 非零
    assert not np.allclose(adv[:3], 0)
    assert abs(adv[:3].mean()) < 1e-9, "组内均值应为0"
    # 组 B: [1,0,0] 同样非零
    assert not np.allclose(adv[3:], 0)
    assert abs(adv[3:].mean()) < 1e-9
    assert len(stats) == 2


def test_zero_variance_groups():
    """全成功/全失败的组无信号。"""
    rewards = [1, 1, 0, 0, 1, 0]
    groups = ["all_success", "all_success", "all_fail", "all_fail", "mixed", "mixed"]
    adv, _ = compute_group_advantages(rewards, groups, mode="std")
    assert np.allclose(adv[:4], 0), "零方差组成员 advantage 应为 0"
    assert adv[4] > 0 and adv[5] < 0, "混合组成员应有非零信号"


def test_std_mode_bounds():
    """std 模式下二值奖励组的 advantage 应有界。"""
    rewards = [1, 0, 1]
    groups = ["g"] * 3
    adv, _ = compute_group_advantages(rewards, groups, mode="std")
    assert adv[0] == adv[2] > 0 > adv[1]


def test_dr_mode():
    """Dr.GRPO 模式：A = r - mean，无 std 除法。"""
    rewards = [1, 0, 0, 0]
    groups = ["g"] * 4
    adv, _ = compute_group_advantages(rewards, groups, mode="dr")
    assert np.allclose(adv, [0.75, -0.25, -0.25, -0.25])


def test_coverage_report():
    """利用率报告：全零方差组 + 混合组 + 单成员组 → usable_frac=1/3。"""
    rewards = [1, 1, 1, 0, 1, 0]
    groups = ["a", "a", "a", "b", "b", "c"]
    _, stats = compute_group_advantages(rewards, groups)
    rep = group_coverage_report(stats)
    assert rep["n_groups"] == 3
    assert abs(rep["usable_frac"] - 1 / 3) < 1e-3  # 只有混合组 b 有信号（round到3位）


def test_empty_and_mismatch():
    try:
        compute_group_advantages([1, 0], ["a"])
        assert False, "应抛长度不一致异常"
    except ValueError:
        pass
    adv, stats = compute_group_advantages([], [], mode="std")
    assert len(adv) == 0 and stats == {}


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  OK {name}")
    print("ALL_GRPO_CORE_TESTS_PASSED")

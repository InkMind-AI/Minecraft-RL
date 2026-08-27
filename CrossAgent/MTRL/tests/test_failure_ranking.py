import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "agent_system" / "multi_turn_rollout" / "failure_ranking.py"
SPEC = importlib.util.spec_from_file_location("failure_ranking_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_trajectory(uid, task="mine_block:oak_log", target_count=0, pickup=None, valid=True, noop=False, infra=False):
    pickup = pickup or {}
    infos = [
        {
            "uid": uid,
            "task_name": task,
            "won": False,
            "mine_block": {"oak_log": 0},
            "pickup": pickup,
            "inventory": [{"type": "oak_log", "quantity": 0}],
            "health": 20,
            "food_level": 20,
            "is_action_valid": valid,
            "action_is_noop": noop,
            "infra_error": infra,
        },
        {
            "uid": uid,
            "task_name": task,
            "won": False,
            "mine_block": {"oak_log": target_count},
            "pickup": pickup,
            "inventory": [{"type": "oak_log", "quantity": target_count}],
            "health": 20,
            "food_level": 20,
            "is_action_valid": valid,
            "action_is_noop": noop,
            "infra_error": infra,
        },
    ]
    steps = [
        {"uid": uid, "is_action_valid": valid, "action_is_noop": noop, "info": info}
        for info in infos
    ]
    return steps, infos


def run(trajectories, rewards, successes=None, mode="sfr", max_steps=20):
    successes = successes or [0] * len(trajectories)
    return MODULE.relabel_episode_rewards(
        trajectory_steps=[item[0] for item in trajectories],
        episode_rewards=np.asarray(rewards, dtype=np.float32),
        episode_lengths=[10] * len(trajectories),
        success=successes,
        max_steps=max_steps,
        mode=mode,
        beta=0.2,
        quality_epsilon=0.02,
        trajectory_infos=[item[1] for item in trajectories],
    )


def test_sfr_ranks_all_failed_trajectories():
    trajectories = [
        make_trajectory("g", target_count=0, noop=True),
        make_trajectory("g", target_count=0),
        make_trajectory("g", target_count=1),
        make_trajectory("g", target_count=2),
    ]
    rewards, result = run(trajectories, [0, 0, 0, 0])
    assert np.allclose(sorted(rewards.tolist()), [-0.2, -0.06666667, 0.06666667, 0.2], atol=1e-5)
    assert result["metrics"]["sfr_all_failed_groups"] == 1
    assert result["metrics"]["sfr_ranked_groups"] == 1
    assert result["metrics"]["sfr_rank_coverage"] == 1.0


def test_sfr_abstains_when_group_has_no_difference():
    trajectories = [make_trajectory("g"), make_trajectory("g"), make_trajectory("g"), make_trajectory("g")]
    rewards, result = run(trajectories, [0, 0, 0, 0])
    assert np.allclose(rewards, 0.0)
    assert result["metrics"]["sfr_abstain_groups"] == 1


def test_success_group_is_unchanged():
    trajectories = [make_trajectory("g", target_count=2), make_trajectory("g")]
    rewards, result = run(trajectories, [1.0, 0.0], successes=[1, 0])
    assert np.allclose(rewards, [1.0, 0.0])
    assert result["metrics"]["sfr_all_failed_groups"] == 0


def test_unrelated_pickup_is_not_task_progress():
    trajectories = [
        make_trajectory("g", pickup={"dirt": 10}),
        make_trajectory("g", pickup={"dirt": 10}),
    ]
    rewards, result = run(trajectories, [0, 0])
    assert np.allclose(rewards, 0.0)
    assert result["metrics"]["sfr_abstain_groups"] == 1


def test_infrastructure_error_is_excluded_from_ranking():
    trajectories = [make_trajectory("g", target_count=1, infra=True), make_trajectory("g", target_count=2)]
    rewards, result = run(trajectories, [0, 0])
    assert np.allclose(rewards, [0, 0])
    assert result["metrics"]["sfr_infra_error_groups"] == 1


def test_comparison_modes_are_available():
    trajectories = [make_trajectory("g", target_count=i) for i in range(4)]
    for mode in ("sparse", "random_rank", "absolute_q", "length_rank"):
        rewards, _ = run(trajectories, [0, 0, 0, 0], mode=mode)
        assert len(rewards) == 4


def test_sparse_mode_preserves_original_rewards():
    trajectories = [make_trajectory("g", target_count=0), make_trajectory("g", target_count=2)]
    rewards, result = run(trajectories, [-0.1, 0.0], mode="sparse")
    assert np.allclose(rewards, [-0.1, 0.0])
    assert result["metrics"]["sfr_ranked_groups"] == 0


def test_config_hash_is_order_invariant():
    left = {"seed": 7, "spawn": {"x": 1, "z": [2, 3]}}
    right = {"spawn": {"z": [2, 3], "x": 1}, "seed": 7}
    assert MODULE.stable_config_hash(left) == MODULE.stable_config_hash(right)


def test_short_success_and_length_inputs_are_padded_safely():
    trajectories = [make_trajectory("g"), make_trajectory("g")]
    rewards, result = MODULE.relabel_episode_rewards(
        trajectory_steps=[item[0] for item in trajectories],
        episode_rewards=[0.0, 0.0],
        episode_lengths=[10],
        success=[0.0],
        max_steps=20,
        trajectory_infos=[item[1] for item in trajectories],
        mode="sfr",
    )
    assert np.allclose(rewards, 0.0)
    assert result["metrics"]["sfr_all_failed_groups"] == 1

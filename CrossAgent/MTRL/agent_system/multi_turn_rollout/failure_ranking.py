"""Simple Failure Ranking (SFR) for Minecraft trajectory-level rewards.

The implementation intentionally stays trajectory-level: a complete rollout
gets one scalar reward after the episode ends. No per-step reward is emitted.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_FEATURE_WEIGHTS = {
    "event": 0.30,
    "inventory": 0.25,
    "validity": 0.15,
    "survival": 0.15,
    "noop": 0.05,
    "loop": 0.05,
    "death": 0.10,
    "timeout": 0.05,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.reshape(-1)[0]
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_config_hash(value: Any) -> str:
    """Build a deterministic short hash for a task configuration."""
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(k): normalize(v)
                for k, v in sorted(item.items(), key=lambda x: str(x[0]))
            }
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item

    payload = json.dumps(normalize(value), sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _flatten_count_mapping(value: Any) -> dict[str, float]:
    """Normalize common Minecraft inventory/event structures into counts."""
    result: dict[str, float] = defaultdict(float)
    if value is None:
        return dict(result)

    if isinstance(value, Mapping):
        for key, item in value.items():
            key = str(key)
            if isinstance(item, Mapping):
                if "type" in item:
                    item_type = str(item["type"])
                    quantity = item.get("quantity", item.get("count", item.get("amount", 1)))
                    result[item_type] += _as_float(quantity, 1.0)
                elif any(k in item for k in ("count", "quantity", "amount", "value")):
                    quantity = item.get(
                        "count",
                        item.get("quantity", item.get("amount", item.get("value", 0))),
                    )
                    result[key] += _as_float(quantity)
                else:
                    for nested_key, nested_value in _flatten_count_mapping(item).items():
                        result[nested_key] += nested_value
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], str):
                result[str(item[0])] += _as_float(item[1], 1.0)
            else:
                result[key] += _as_float(item)
        return dict(result)

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping):
                item_type = item.get("type", item.get("name"))
                if item_type is None:
                    for nested_key, nested_value in _flatten_count_mapping(item).items():
                        result[nested_key] += nested_value
                else:
                    quantity = item.get("quantity", item.get("count", item.get("amount", 1)))
                    result[str(item_type)] += _as_float(quantity, 1.0)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result[str(item[0])] += _as_float(item[1], 1.0)
        return dict(result)

    return dict(result)


def _event_counts(info: Mapping[str, Any]) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    event_types = {
        "pickup",
        "break_item",
        "craft_item",
        "mine_block",
        "kill_entity",
        "use_item",
        "drop",
        "entity_killed_by",
        "custom",
    }
    for event_type in event_types:
        raw = info.get(event_type)
        if isinstance(raw, Mapping):
            for obj, count in raw.items():
                name = str(obj)
                if event_type == "custom" and "interact_with" in name:
                    name = name.split("interact_with", 1)[-1].lstrip(":_")
                    event_key = f"interact_with:{name}"
                else:
                    event_key = f"{event_type}:{name}"
                counts[event_key] += _as_float(count)

    stats = info.get("stats")
    if isinstance(stats, Mapping):
        for key, value in stats.items():
            counts[str(key)] = max(counts[str(key)], _as_float(value))
    return dict(counts)


def _task_parts(task_name: Any) -> tuple[str, str]:
    text = str(task_name or "")
    if ":" in text:
        task_type, target = text.split(":", 1)
    else:
        task_type, target = text, ""
    return task_type.strip(), target.strip()


def _position_key(info: Mapping[str, Any]) -> tuple[Any, ...] | None:
    location = info.get("location_stats")
    if not isinstance(location, Mapping):
        location = info.get("player_pos")
    if not isinstance(location, Mapping):
        return None
    x = location.get("xpos", location.get("x"))
    y = location.get("ypos", location.get("y"))
    z = location.get("zpos", location.get("z"))
    if x is None or y is None or z is None:
        return None
    return (
        round(_as_float(x), 1),
        round(_as_float(y), 1),
        round(_as_float(z), 1),
        round(_as_float(location.get("yaw", 0)), 0),
        round(_as_float(location.get("pitch", 0)), 0),
        _as_bool(info.get("is_gui_open", info.get("isGuiOpen", False))),
    )


def _loop_ratio(infos: Sequence[Mapping[str, Any]]) -> float:
    explicit = [
        _as_bool(info.get("action_is_loop", info.get("is_loop", False)))
        for info in infos
        if "action_is_loop" in info or "is_loop" in info
    ]
    if explicit:
        return float(np.mean(explicit))

    state_keys = [_position_key(info) for info in infos]
    if len([key for key in state_keys if key is not None]) < 8:
        return 0.0

    repeated = 0
    total = 0
    for end in range(7, len(state_keys)):
        window = [key for key in state_keys[end - 7 : end + 1] if key is not None]
        if len(window) < 8:
            continue
        total += 1
        if len(set(window)) <= 2:
            has_event = any(
                any(value > 0 for value in _event_counts(info).values())
                for info in infos[max(0, end - 7) : end + 1]
            )
            if not has_event:
                repeated += 1
    return repeated / max(total, 1)


def _target_event_keys(task_type: str, target: str) -> set[str]:
    if not target:
        return set()
    keys = {f"{task_type}:{target}", f"pickup:{target}"}
    if task_type == "smelt_item":
        keys.add(f"craft_item:{target}")
    return keys


def _bounded_progress(value: float) -> float:
    """Keep progress in [0, 1] while preserving differences above one event."""
    value = max(float(value), 0.0)
    return value / (1.0 + value)


def _trajectory_quality(
    infos: Sequence[Mapping[str, Any]],
    trajectory_steps: Sequence[Mapping[str, Any]] | None,
    episode_length: int,
    max_steps: int,
    feature_weights: Mapping[str, float],
) -> dict[str, Any]:
    infos = [info for info in infos if isinstance(info, Mapping)]
    last = infos[-1] if infos else {}
    task_name = last.get("task_name", infos[0].get("task_name", "") if infos else "")
    task_type, target = _task_parts(task_name)
    target_keys = _target_event_keys(task_type, target)

    event_history = [_event_counts(info) for info in infos]
    event_final: dict[str, float] = defaultdict(float)
    event_initial: dict[str, float] = defaultdict(float)
    for event_map in event_history:
        for key, value in event_map.items():
            event_final[key] = max(event_final[key], value)
    if event_history:
        explicit_initial = infos[0].get("initial_event_counts")
        if isinstance(explicit_initial, Mapping):
            event_initial.update(_event_counts({"stats": explicit_initial}))
        else:
            # Rollout logs start after reset; cumulative counters include
            # events from the first executed action.
            event_initial = defaultdict(float)
    useful_delta = {
        key: max(event_final[key] - event_initial.get(key, 0.0), 0.0)
        for key in event_final
    }
    target_progress = sum(useful_delta.get(key, 0.0) for key in target_keys)
    useful_event_count = sum(
        useful_delta.get(key, 0.0) > 0 for key in target_keys
    )
    event_score = (
        _bounded_progress(target_progress)
        if target_progress > 0
        else min(useful_event_count / 3.0, 1.0)
    )

    final_inventory = {}
    initial_inventory = {}
    for info in reversed(infos):
        if not final_inventory and "inventory" in info:
            final_inventory = _flatten_count_mapping(info["inventory"])
        if not final_inventory and "inventory_stats" in info:
            final_inventory = _flatten_count_mapping(info["inventory_stats"])
    for info in infos:
        if "initial_inventory" in info:
            initial_inventory = _flatten_count_mapping(info["initial_inventory"])
            break
        if "inventory" in info:
            initial_inventory = _flatten_count_mapping(info["inventory"])
            break
    inventory_target = final_inventory.get(target, 0.0) - initial_inventory.get(target, 0.0)
    inventory_score = _bounded_progress(inventory_target)

    valid_values = [
        _scalar(step["is_action_valid"])
        for step in trajectory_steps or []
        if isinstance(step, Mapping) and "is_action_valid" in step
    ]
    if not valid_values:
        valid_values = [
            info["is_action_valid"]
            for info in infos
            if "is_action_valid" in info
        ]
    validity_available = bool(valid_values)
    validity_score = (
        float(np.mean([_as_bool(value, True) for value in valid_values]))
        if valid_values
        else 0.0
    )

    noop_values = [
        _scalar(step["action_is_noop"])
        for step in trajectory_steps or []
        if isinstance(step, Mapping) and "action_is_noop" in step
    ]
    if not noop_values:
        noop_values = [
            info["action_is_noop"]
            for info in infos
            if "action_is_noop" in info
        ]
    noop_available = bool(noop_values)
    noop_ratio = float(np.mean([_as_bool(value) for value in noop_values])) if noop_values else 0.0

    health_values = [
        _as_float(info.get("health"), np.nan)
        for info in infos
        if "health" in info
    ]
    food_values = [
        _as_float(info.get("food_level"), np.nan)
        for info in infos
        if "food_level" in info
    ]
    health_values = [value for value in health_values if np.isfinite(value)]
    food_values = [value for value in food_values if np.isfinite(value)]
    final_health = health_values[-1] if health_values else 0.0
    final_food = food_values[-1] if food_values else 0.0
    survival_available = bool(health_values or food_values)
    health_score = min(max(final_health / 20.0, 0.0), 1.0) if health_values else 0.0
    food_score = min(max(final_food / 20.0, 0.0), 1.0) if food_values else health_score
    survival_score = 0.7 * health_score + 0.3 * food_score

    death = any(
        _as_bool(info.get("death_detected", info.get("dead", False)))
        or _as_float(info.get("health"), 1.0) <= 0
        for info in infos
    )
    death = death or any(_as_bool(info.get("respawn_detected", False)) for info in infos)
    timeout = bool(max_steps > 0 and episode_length >= max_steps)
    loop_ratio = _loop_ratio(infos)

    values = {
        "event": event_score,
        "inventory": inventory_score,
        "validity": validity_score,
        "survival": survival_score,
        "noop": noop_ratio,
        "loop": loop_ratio,
        "death": float(death),
        "timeout": float(timeout),
    }
    availability = {
        "event": bool(target_keys),
        "inventory": bool(final_inventory or initial_inventory),
        "validity": validity_available,
        "survival": survival_available,
        "noop": noop_available,
        "loop": bool(loop_ratio > 0 or len(infos) >= 8),
        "death": True,
        "timeout": True,
    }

    numerator = 0.0
    denominator = 0.0
    positive = {"event", "inventory", "validity", "survival"}
    for name, weight in feature_weights.items():
        if availability.get(name, False):
            denominator += abs(float(weight))
            sign = 1.0 if name in positive else -1.0
            numerator += sign * float(weight) * values[name]
    quality = numerator / denominator if denominator > 0 else 0.0
    evidence = int(
        target_progress > 0
        or useful_event_count > 0
        or inventory_target > 0
        or (validity_available and validity_score < 1.0)
        or (noop_available and noop_ratio > 0)
        or loop_ratio > 0
        or death
        or timeout
    )
    return {
        "quality": float(quality),
        "quality_components": {key: float(value) for key, value in values.items()},
        "availability": availability,
        "evidence": evidence,
        "infra_error": any(_as_bool(info.get("infra_error", False)) for info in infos),
        "task_type": task_type,
        "target": target,
        "useful_event_count": int(useful_event_count),
        "target_progress": float(target_progress),
        "inventory_target_gain": float(max(inventory_target, 0.0)),
        "validity_score": float(validity_score),
        "noop_ratio": float(noop_ratio),
        "loop_ratio": float(loop_ratio),
        "final_health": float(final_health),
        "final_food": float(final_food),
        "death": bool(death),
        "timeout": bool(timeout),
    }


def _midranks(values: Sequence[float], eps: float) -> tuple[list[float], bool]:
    if not values:
        return [], False
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    for end in range(1, len(order) + 1):
        if end == len(order) or abs(float(values[order[end]]) - float(values[order[start]])) > eps:
            midpoint = (start + end - 1) / 2.0
            for position in range(start, end):
                ranks[order[position]] = midpoint
            start = end
    return ranks, max(values) - min(values) >= eps


def _group_indices(
    trajectory_steps: Sequence[Sequence[Mapping[str, Any]]],
    fallback_uids: Sequence[Any] | None = None,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, steps in enumerate(trajectory_steps):
        uid = None
        if steps and isinstance(steps[0], Mapping):
            uid = steps[0].get("uid")
        if uid is None and fallback_uids is not None and index < len(fallback_uids):
            uid = fallback_uids[index]
        if uid is None:
            uid = f"trajectory-{index}"
        groups[str(_scalar(uid))].append(index)
    return dict(groups)


def _attach_metadata(
    trajectory_steps: Sequence[Sequence[Mapping[str, Any]]],
    index: int,
    metadata: Mapping[str, Any],
) -> None:
    for step in trajectory_steps[index]:
        if isinstance(step, dict):
            step.update(metadata)


def relabel_episode_rewards(
    trajectory_steps: Sequence[Sequence[Mapping[str, Any]]],
    episode_rewards: Sequence[float],
    episode_lengths: Sequence[int],
    success: Sequence[float],
    max_steps: int,
    trajectory_infos: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    mode: str = "sparse",
    beta: float = 0.2,
    quality_epsilon: float = 0.02,
    random_seed: int = 0,
    fallback_uids: Sequence[Any] | None = None,
    feature_weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Relabel complete-episode rewards and annotate trajectory steps."""
    rewards = np.asarray(episode_rewards, dtype=np.float32).copy()
    lengths = np.zeros(len(rewards), dtype=np.int64)
    raw_lengths = np.asarray(episode_lengths, dtype=np.int64).reshape(-1)
    lengths[: min(len(lengths), len(raw_lengths))] = raw_lengths[: len(lengths)]
    successes = np.zeros(len(rewards), dtype=np.float32)
    raw_successes = np.asarray(success, dtype=np.float32).reshape(-1)
    successes[: min(len(successes), len(raw_successes))] = raw_successes[: len(successes)]
    mode = str(mode or "sparse").lower()
    supported_modes = {"sparse", "sfr", "random_rank", "absolute_q", "length_rank"}
    if mode not in supported_modes:
        raise ValueError(
            f"Unsupported SFR mode: {mode}. Expected one of {sorted(supported_modes)}"
        )
    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if feature_weights:
        weights.update({str(key): float(value) for key, value in feature_weights.items()})

    n = len(rewards)
    qualities = np.zeros(n, dtype=np.float32)
    ranks = np.full(n, -1.0, dtype=np.float32)
    abstain = np.zeros(n, dtype=np.bool_)
    all_failed = np.zeros(n, dtype=np.bool_)
    summaries: list[dict[str, Any]] = []
    for index in range(n):
        if trajectory_infos is not None and index < len(trajectory_infos):
            infos = [
                info for info in trajectory_infos[index]
                if isinstance(info, Mapping)
            ]
        else:
            infos = [
                step.get("info")
                for step in trajectory_steps[index]
                if isinstance(step, Mapping) and isinstance(step.get("info"), Mapping)
            ]
        summary = _trajectory_quality(
            infos=infos,
            trajectory_steps=trajectory_steps[index],
            episode_length=int(lengths[index]) if index < len(lengths) else 0,
            max_steps=max_steps,
            feature_weights=weights,
        )
        summaries.append(summary)
        qualities[index] = summary["quality"]

    metrics: dict[str, Any] = {
        "sfr_mode": mode,
        "sfr_total_trajectories": int(n),
        "sfr_all_failed_groups": 0,
        "sfr_ranked_groups": 0,
        "sfr_abstain_groups": 0,
        "sfr_rank_coverage": 0.0,
        "sfr_mean_quality": float(np.mean(qualities)) if n else 0.0,
        "sfr_quality_std": float(np.std(qualities)) if n else 0.0,
        "sfr_infra_error_groups": 0,
    }

    groups = _group_indices(trajectory_steps, fallback_uids=fallback_uids)
    rng = np.random.default_rng(random_seed)
    for _, indices in groups.items():
        if any(summaries[index]["infra_error"] for index in indices):
            metrics["sfr_infra_error_groups"] += 1
            for index in indices:
                _attach_metadata(
                    trajectory_steps,
                    index,
                    {
                        "sfr_reward": float(rewards[index]),
                        "sfr_quality": float(qualities[index]),
                        "sfr_rank": -1.0,
                        "sfr_abstain": True,
                        "sfr_all_failed": False,
                        "sfr_infra_error": True,
                    },
                )
            continue
        group_failed = not bool(np.any(successes[indices] > 0.5))
        if not group_failed:
            continue
        all_failed[indices] = True
        metrics["sfr_all_failed_groups"] += 1
        if mode == "sparse":
            for index in indices:
                _attach_metadata(
                    trajectory_steps,
                    index,
                    {
                        "sfr_reward": float(rewards[index]),
                        "sfr_quality": float(qualities[index]),
                        "sfr_rank": -1.0,
                        "sfr_abstain": False,
                        "sfr_all_failed": True,
                        "sfr_infra_error": False,
                    },
                )
            continue

        if mode == "length_rank":
            group_values = [float(lengths[index]) for index in indices]
        elif mode == "random_rank":
            group_values = list(rng.random(len(indices)))
        else:
            group_values = [float(qualities[index]) for index in indices]

        group_ranks, has_difference = _midranks(group_values, quality_epsilon)
        has_evidence = any(summaries[index]["evidence"] > 0 for index in indices)
        if mode == "sfr" and (not has_difference or not has_evidence):
            abstain[indices] = True
            metrics["sfr_abstain_groups"] += 1
            rewards[indices] = 0.0
            for index in indices:
                _attach_metadata(
                    trajectory_steps,
                    index,
                    {
                        "sfr_reward": 0.0,
                        "sfr_quality": float(qualities[index]),
                        "sfr_rank": -1.0,
                        "sfr_abstain": True,
                        "sfr_all_failed": True,
                        "sfr_infra_error": False,
                    },
                )
            continue

        denom = max(len(indices) - 1, 1)
        low = float(np.min(group_values))
        high = float(np.max(group_values))
        for local_rank, index in enumerate(indices):
            normalized_rank = float(group_ranks[local_rank] / denom)
            ranks[index] = normalized_rank
            if mode == "absolute_q":
                reward = beta * (
                    (group_values[local_rank] - low) / (high - low)
                    if high > low
                    else 0.0
                )
            else:
                reward = beta * (2.0 * normalized_rank - 1.0)
            rewards[index] = float(reward)
            _attach_metadata(
                trajectory_steps,
                index,
                {
                    "sfr_reward": float(reward),
                    "sfr_quality": float(qualities[index]),
                    "sfr_rank": normalized_rank,
                    "sfr_abstain": False,
                    "sfr_all_failed": True,
                    "sfr_infra_error": False,
                },
            )
        metrics["sfr_ranked_groups"] += 1

    for index in range(n):
        if not all_failed[index]:
            _attach_metadata(
                trajectory_steps,
                index,
                {
                    "sfr_reward": float(rewards[index]),
                    "sfr_quality": float(qualities[index]),
                    "sfr_rank": -1.0,
                    "sfr_abstain": False,
                    "sfr_all_failed": False,
                    "sfr_infra_error": False,
                },
            )

    if metrics["sfr_all_failed_groups"]:
        metrics["sfr_rank_coverage"] = (
            metrics["sfr_ranked_groups"] / metrics["sfr_all_failed_groups"]
        )
    metrics["sfr_reward_mean"] = float(np.mean(rewards)) if n else 0.0
    metrics["sfr_reward_std"] = float(np.std(rewards)) if n else 0.0
    metrics["sfr_abstain_ratio"] = float(np.mean(abstain)) if n else 0.0
    return rewards, {
        "metrics": metrics,
        "qualities": qualities,
        "ranks": ranks,
        "abstain": abstain,
        "all_failed": all_failed,
        "summaries": summaries,
    }


__all__ = [
    "DEFAULT_FEATURE_WEIGHTS",
    "relabel_episode_rewards",
    "stable_config_hash",
]

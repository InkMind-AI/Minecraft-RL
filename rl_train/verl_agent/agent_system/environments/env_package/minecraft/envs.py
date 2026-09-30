"""Minecraft 环境包（verl-agent 多轮 rollout 用）。

09-28 重写。原版来自 CrossAgent 抢救资产（rl_train/verl_port/env_minecraft/），
是对着另一版 openagents 写的，对当前仓库而言**从未跑通过**，问题清单：

  1. import 断裂：render_video_enable / InitialActionCallback / craft_item 的
     三个符号在当前 openagents 里都不存在 → 模块一导入就 ImportError
  2. env_init 调用契约不符：旧代码 `self.env, extra_info = env_init(**kwargs)`，
     传 group_n/action_type/task_name 等关键字；当前 env_init 签名是
     (task_config, rollout_path, args, ...) 且只返回 env → 必然 TypeError
  3. 上面的 TypeError 被 `while True: try/except: retry` 包着 → **无限重试挂死**
  4. reset 里 `time.sleep(random.randint(0, 180))` → 每次 reset 平均白等 90s
  5. `self.tasks = env_kwargs["tasks"]`，但 env_manager 从不传 tasks → KeyError
  6. save_render_videos 里有 `breakpoint()` → Ray worker 里会永久挂起
  7. 同组（GRPO group）各 worker 各自调 choose_available_task → 各自随机 seed，
     组内初始条件不同，组内相对优势失去可比性

本版做法：
  - 不经 env_init（它强制要求 rollout_path + args.fps 并写 raw_action.jsonl），
    直接按 rollout_openha.py / env.py 的同一套 callback 组装 MinecraftSim，
    录像变为可选
  - 任务配置在 **组级** 采样一次，同组 group_n 个 worker 共享同一个 task_config
    （同 seed、同出生点、同初始物品）——GRPO 组内对比的前提
  - 所有重试有上限，超限直接抛错，不静默挂死
  - 观测取 info["pov"]（与 OpenHA.gen_response 的 real_obs 一致），而非
    obs["image"]（224×224 的低清副本）
  - 成功判定与评测一致：reward > 0 即成功并终止（rollout_openha.py 同款）
"""
import copy
import random
import time
import traceback
from typing import Any, Dict, List, Optional

import gym
import numpy as np
import ray

# ⚠ 09-28：numpy 2.0 移除了 `np.unicode_`（改名 `np.str_`），而 minestudio 内置
# 的 minerl fork（herobraine/hero/spaces.py 的 Text space）还在用旧名——不是
# 我们的代码问题，是第三方库对 numpy 2.x 的兼容缺口。verl 训练 env（vllmtrain，
# vllm 0.17/torch 2.10 要求 numpy>=2）不能像评测 openha env（numpy 1.26.4）
# 那样靠装老版本 numpy 绕开：vllm/transformers 那一侧才是真正需要 numpy 2 的一方。
# 官方 numpy 2.0 迁移指南给的标准做法就是这个别名 shim；必须在 `from
# minestudio.simulator import MinecraftSim` 之前打上（MinecraftSim() 实例化时
# 才会真正触发 herobraine 的 space 构造，见 verl-trainenv10 实测的完整调用栈：
# envs.py::_build_sim -> MinecraftSim.__init__ -> HumanSurvival.__init__ ->
# env_spec.reset() -> create_observables() -> spaces.Text.__init__ ->
# AttributeError: `np.unicode_` was removed in the NumPy 2.0 release）。
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

# 与评测 rollout 完全相同的 callback 来源（openagents/envs/env.py）
from openagents.envs.callbacks import (CommandsCallback, InitInventoryCallback,
                                       RecordCallback, SummonMobsCallback)
from openagents.envs.tasks.task_manager import choose_available_task
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RewardsCallback

MAX_SIM_CREATE_ATTEMPTS = 3
# worker 启动错峰：避免 N 个 JVM 同时拉起抢资源。原版是 0-180s，这里压到 0-5s。
STARTUP_JITTER_S = 5.0
# 与评测 harness（run_backbone_eval.sh MAX_STEPS_NUM）一致
DEFAULT_MAX_STEPS = 200


def _info_for_transport(info: Dict[str, Any]) -> Dict[str, Any]:
    """从 MineStudio 的 info 里挑出可序列化的小字段。

    原始 info 里有 pov（640×360×3）等大数组，每步经 Ray 回传会成为吞吐瓶颈；
    图像已作为观测单独返回，这里只保留标量/短字段。
    """
    keep = {}
    for k, v in (info or {}).items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            keep[k] = v
    return keep


def _build_sim(task_config: Dict[str, Any], record_path: Optional[str], fps: int = 20) -> MinecraftSim:
    """按 openagents/envs/env.py::env_init 的同一套 callback 组装模拟器（录像可选）。"""
    cb_cfg = task_config["callback"]
    inv_cfg = cb_cfg["init_inventory"]
    callbacks = [
        InitInventoryCallback(
            inv_cfg.get("init_inventory", []),
            inventory_distraction_level=inv_cfg.get("inventory_distraction_level", [0]),
            equip_distraction_level=inv_cfg.get("equip_distraction_level", [0]),
            forbidden_slots=inv_cfg.get("forbidden_slots", []),
        ),
        RewardsCallback(task_config["rewards"]),
        CommandsCallback(cb_cfg.get("commands", [])),
    ]
    if cb_cfg.get("mobs"):
        callbacks.append(SummonMobsCallback(cb_cfg["mobs"]))
    if record_path:
        callbacks.insert(0, RecordCallback(record_path=record_path, fps=fps, frame_type="pov"))
    return MinecraftSim(
        action_type="env",
        obs_size=(224, 224),
        render_size=(640, 360),
        seed=task_config["seed"],
        preferred_spawn_biome=None,
        camera_config=None,
        callbacks=callbacks,
    )


# -----------------------------------------------------------------------------
# Ray remote worker ------------------------------------------------------------
# -----------------------------------------------------------------------------

class MinecraftWorker:
    """一个 Ray actor 托管一个 MinecraftSim。

    ⚠ 09-30：这个类**不带 `@ray.remote` 装饰器**，由 `MinecraftMultiProcessEnv`
    用 `ray.remote(**resources_per_worker)(MinecraftWorker)` 动态包装——与
    `sokoban/envs.py:85` 同一写法。原先写死 `@ray.remote(num_cpus=2)`，导致
    启动脚本里的 `env.resources_per_worker.num_cpus` 是死参数（env_manager 的
    minecraft 分支当时也没往下传，见同日修复），且 CPU 预留随
    `train_batch_size × group_n` 线性膨胀：worker 数 = 组数 × 组大小 + 验证集，
    每个固定吃 2 CPU，很容易超过 koala 的每卡 CPU 配额（-g 1 默认 20 CPU），
    超了以后 Ray actor 永久 PENDING，最终在 `reset()` 的
    `ray.get(..., timeout=reset_timeout_s)` 抛 GetTimeoutError——表现为"卡住
    15 分钟后超时"，很难一眼看出是 CPU 预留问题。
    """

    def __init__(self, env_id: int = 0, max_steps: int = DEFAULT_MAX_STEPS):
        self.env_id = env_id
        self.max_steps = max_steps
        self.env: Optional[MinecraftSim] = None
        self.task_config: Optional[Dict[str, Any]] = None
        self.cur_step = 0
        self.won = False
        self.done = False
        self._last_pov: Optional[np.ndarray] = None
        self._noop = None

    # ------------------------------------------------------------------
    def _close_sim(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                traceback.print_exc()
            self.env = None

    def _info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = {
            "won": self.won,
            "step_count": self.cur_step,
            "task_name": self.task_config["task_name"],
            "task_description": self.task_config["task_description"],
        }
        if extra:
            info.update(extra)
        return info

    def reset(self, task_config: Dict[str, Any], record_path: Optional[str] = None):
        """用给定 task_config 起一局；同组 worker 收到的是同一个 task_config。"""
        self._close_sim()
        self.task_config = copy.deepcopy(task_config)
        self.cur_step = 0
        self.won = False
        self.done = False
        time.sleep(random.random() * STARTUP_JITTER_S)

        last_err = None
        for attempt in range(1, MAX_SIM_CREATE_ATTEMPTS + 1):
            try:
                self.env = _build_sim(self.task_config, record_path)
                obs, info = self.env.reset()
                for action in self.task_config.get("init_actions", []):
                    obs, _r, _t, _tr, info = self.env.step(action)
                # rollout_openha.py 在交给 agent 前先走一个 noop，这里保持一致
                self._noop = self.env.noop_action()
                obs, _r, _t, _tr, info = self.env.step(self._noop)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[MinecraftWorker {self.env_id}] sim 创建失败 "
                      f"{attempt}/{MAX_SIM_CREATE_ATTEMPTS}: {e!r}", flush=True)
                traceback.print_exc()
                self._close_sim()
                time.sleep(2 + random.random() * 3)
        else:
            raise RuntimeError(f"MinecraftWorker {self.env_id}: sim 创建连续失败") from last_err

        self._last_pov = np.asarray(info.get("pov", obs["image"]), dtype=np.uint8)
        return self._last_pov, self._info(_info_for_transport(info))

    def step(self, action: Dict[str, Any]):
        """action = {"raw_action": env_action_dict 或 None, "thought": str}。

        已结束的 env 不再推进模拟器，直接回放终态（verl-agent 的 rollout 循环
        会对整批调用 step，已 done 的样本也会收到动作）。
        """
        if self.done:
            return self._last_pov, 0.0, True, self._info({"already_done": True})

        env_action = action.get("raw_action") if isinstance(action, dict) else None
        if env_action is None:
            env_action = self._noop  # 非法动作按 noop 执行，valid 标志由 projection 负责

        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.cur_step += 1
        reward = float(reward)
        if reward > 0:  # 评测同款成功判定：拿到任务奖励即成功并终止
            self.won = True
        done = bool(self.won or terminated or truncated or self.cur_step >= self.max_steps)
        self.done = done
        self._last_pov = np.asarray(info.get("pov", obs["image"]), dtype=np.uint8)
        return self._last_pov, reward, done, self._info(_info_for_transport(info))

    def close(self):
        self._close_sim()
        return True

    def ready(self):
        return True


# -----------------------------------------------------------------------------
# Vectorised Ray environment ---------------------------------------------------
# -----------------------------------------------------------------------------

class MinecraftMultiProcessEnv(gym.Env):
    """env_num 个任务组 × 每组 group_n 个 worker（同组共享同一 task_config）。"""

    def __init__(self, env_num: int = 1, group_n: int = 1, is_train: bool = True,
                 resources_per_worker: Optional[dict] = None,
                 env_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        env_kwargs = dict(env_kwargs or {})
        # 每个 worker 的 Ray 资源预留。默认 0.5 CPU（不是原先写死的 2）：Malmo 的
        # 真实计算在独立的 JVM 进程里，Ray actor 本身只做 RPC 转发，预留过多会
        # 在 worker 数增长时直接撞上 koala 的每卡 CPU 配额（详见 MinecraftWorker
        # 的文档串）。sokoban 同理用 0.1。
        resources_per_worker = dict(resources_per_worker or {"num_cpus": 0.5})
        if not is_train:
            assert group_n == 1, "验证集不分组"
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.is_train = is_train

        tasks = env_kwargs.get("tasks")
        if not tasks:
            # 兼容 env_manager 只给单个 task_name 的旧调用方式
            tasks = [env_kwargs["task_name"]]
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]
        self.tasks: List[str] = list(tasks)
        self.difficulty = env_kwargs.get("difficulty", "easy")
        self.max_steps = int(env_kwargs.get("max_steps") or DEFAULT_MAX_STEPS)
        self.record_path = env_kwargs.get("record_path")
        self.reset_timeout_s = int(env_kwargs.get("reset_timeout_s", 900))
        self.step_timeout_s = int(env_kwargs.get("step_timeout_s", 300))
        self._rng = random.Random(env_kwargs.get("seed", 0) + (0 if is_train else 1000))

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        env_worker = ray.remote(**resources_per_worker)(MinecraftWorker)
        self._workers = [env_worker.remote(i, self.max_steps) for i in range(self.num_processes)]
        self._closed = False

    # ------------------------------------------------------------------
    def _sample_group_configs(self) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(self.env_num):
            task = self._rng.choice(self.tasks)
            configs.append(choose_available_task(task, difficulty=self.difficulty))
        return configs

    def reset(self):
        group_configs = self._sample_group_configs()
        futures = []
        for idx, worker in enumerate(self._workers):
            cfg = group_configs[idx // self.group_n]
            rec = None
            if self.record_path:
                rec = f"{self.record_path}/{cfg['task_name'].replace(':', '_')}/env{idx:03d}_{int(time.time())}"
            futures.append(worker.reset.remote(cfg, rec))
        results = ray.get(futures, timeout=self.reset_timeout_s)
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def step(self, actions: List[Dict[str, Any]]):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")
        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures, timeout=self.step_timeout_s)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", True):
            return
        try:
            ray.get([w.close.remote() for w in self._workers], timeout=60)
        except Exception:
            traceback.print_exc()
        for w in self._workers:
            try:
                ray.kill(w, no_restart=True)
            except Exception:
                pass
        self._closed = True

    def __del__(self):  # noqa: D401
        try:
            self.close()
        except Exception:
            pass


def build_minecraft_envs(env_num: int = 1, group_n: int = 1, is_train: bool = True,
                         resources_per_worker: Optional[dict] = None,
                         env_kwargs: Optional[dict] = None):
    """与 build_sokoban_envs 同形的工厂函数（含 resources_per_worker 透传）。"""
    return MinecraftMultiProcessEnv(env_num=env_num, group_n=group_n,
                                    is_train=is_train,
                                    resources_per_worker=resources_per_worker,
                                    env_kwargs=env_kwargs)

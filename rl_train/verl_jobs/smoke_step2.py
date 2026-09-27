"""verl-agent 迁移第 2 步冒烟：Minecraft 环境接入验收（koala，需 xvfb + Malmo）。

第 1 步（smoke_step1.py）验的是"模型侧"——适配器 + 真实 9B 权重能前向。
本脚本验"环境侧"——verl-agent 的多轮 rollout 框架能否真的驱动 Malmo：

  D. import + projection：env 包可导入；minecraft_projection 能把模型文本
     解码成 {raw_action, thought}，非法文本降级为 valid=0 而不抛异常
  E. Ray + Malmo 起环境：build_minecraft_envs(env_num=1, group_n=1) → reset()
     拿到首帧观测（这是最重的一步，Malmo/JVM 起不来就在这里暴露）
  F. step 闭环：projection 产出的动作喂进 env.step()，拿到 obs/reward/done/info
     —— 同时把 info 的键打出来，供 reward 契约对齐用（迁移第 2 步的已知工作项）

用法（koala job 内，openha env，已装 minestudio + xvfb）:
    xvfb-run -a python rl_train/verl_jobs/smoke_step2.py --steps 3
"""
import argparse
import sys
import traceback

PASS = []


def check_d_projection():
    print("=== D. import + projection ===", flush=True)
    from agent_system.environments.env_package.minecraft import (  # noqa: F401
        build_minecraft_envs, minecraft_projection)
    print("[OK] env_package.minecraft 可导入", flush=True)

    # 合法样例用训练/评测一致的文本动作语法（见 action_mapping.py）
    good = "Action: move(0, 0) and press(w)"
    bad = "这不是一个动作"
    actions, valids = minecraft_projection([good, bad])
    print(f"  valids={valids}", flush=True)
    assert valids[0] == 1, f"合法动作被判非法: {actions[0]}"
    assert valids[1] == 0, f"非法动作未被拦下: {actions[1]}"
    assert isinstance(actions[0], dict) and "raw_action" in actions[0]
    print(f"[OK] projection: 合法→{actions[0]['raw_action']!r}; 非法→降级 valid=0", flush=True)
    PASS.append("D")


def check_ef_env_loop(steps: int, task_name: str, task_description: str, max_steps: int):
    print("=== E. Ray + Malmo 起环境 ===", flush=True)
    import ray
    from agent_system.environments.env_package.minecraft import (
        build_minecraft_envs, minecraft_projection)

    if not ray.is_initialized():
        # 本地模式即可：只验契约，不验并发扩展性
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    env_kwargs = {
        "task_name": task_name,
        "task_description": task_description,
        "max_steps": max_steps,
    }
    envs = build_minecraft_envs(env_num=1, group_n=1, is_train=True, env_kwargs=env_kwargs)
    obs = envs.reset()
    print(f"[OK] reset() 返回 type={type(obs).__name__}", flush=True)
    if isinstance(obs, (list, tuple)):
        print(f"  batch={len(obs)}; first keys="
              f"{list(obs[0].keys()) if isinstance(obs[0], dict) else type(obs[0]).__name__}",
              flush=True)
    PASS.append("E")

    print(f"=== F. step 闭环 × {steps} ===", flush=True)
    action_text = "Action: move(0, 0) and press(w)"
    for i in range(steps):
        acts, valids = minecraft_projection([action_text])
        out = envs.step(acts)
        # verl-agent 的 env 约定返回 (obs, reward, done, info)
        assert isinstance(out, tuple) and len(out) == 4, f"step 返回形状异常: {type(out)}"
        _obs, reward, done, info = out
        info0 = info[0] if isinstance(info, (list, tuple)) and info else info
        print(f"  step{i}: valid={valids[0]} reward={reward} done={done} "
              f"info_keys={sorted(info0.keys()) if isinstance(info0, dict) else type(info0).__name__}",
              flush=True)
    print("[OK] step 闭环通过（obs/reward/done/info 契约成立）", flush=True)
    PASS.append("F")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--task-name", default="mine_block")
    ap.add_argument("--task-description", default="mine the oak log")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    run_d = not only or "d" in only
    run_ef = not only or "e" in only or "f" in only

    try:
        if run_d:
            check_d_projection()
        if run_ef:
            check_ef_env_loop(args.steps, args.task_name, args.task_description, args.max_steps)
    except Exception:
        traceback.print_exc()

    expected = (1 if run_d else 0) + (2 if run_ef else 0)
    ok = len(PASS) == expected and expected > 0
    print(f"\nSMOKE2_RESULT: {','.join(PASS) or '-'} "
          f"({len(PASS)}/{expected}) {'✅ 通过' if ok else '❌ 未全通过'}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""verl-agent 迁移第 2 步冒烟：Minecraft 环境接入验收（koala，需 xvfb + Malmo）。

第 1 步（smoke_step1.py）验"模型侧"——适配器 + 真实 9B 权重能前向。
本脚本验"环境侧"——verl-agent 的多轮 rollout 框架能否真的驱动 Malmo：

  D. projection：合法文本→env 动作；**有 Action: 前缀但无动作语法**的自然语言
     （iter1 崩溃样例）与垃圾文本都必须判 valid=0
  E. Ray + Malmo：build_minecraft_envs(env_num=1, group_n=2) → reset()，
     拿到首帧 pov；**同组两个 worker 必须拿到同一个 task_config**（GRPO 前提）
  F. step 闭环：projection 产出的动作喂 env.step()，obs/reward/done/info 契约成立
  G. MinecraftEnvironmentManager：verl 训练真正调用的入口，reset/step 产出
     {'text','image','anchor'}，text 以 system prompt 开头、以 <image> 结尾

用法（koala job 内，openha env）:
    xvfb-run -a python rl_train/verl_jobs/smoke_step2.py --steps 3
"""
import argparse
import sys
import time
import traceback

import numpy as np

PASS = []


def check_d_projection():
    print("=== D. projection ===", flush=True)
    from agent_system.environments.env_package.minecraft import minecraft_projection
    cases = [
        ("Action: move(0, 0) and press(w)", 1),
        ("Thought: sheep ahead | x | attack\nAction: move(5, -3) and click(left)", 1),
        ("Action: move(0, 0) and press()", 1),
        ("Action: Walk forward across the terrain", 0),   # iter1 崩溃样例
        ("<think>\n\n</think>\n\nThe sheep is above", 0),
        ("", 0),
    ]
    projected, valids = minecraft_projection([c for c, _ in cases])
    for (text, want), got, p in zip(cases, valids, projected):
        mark = "✓" if got == want else "✗"
        print(f"  {mark} valid={got} (期望 {want}) {text[:44]!r}", flush=True)
        assert got == want, f"projection 判定错误: {text!r}"
        assert (p["raw_action"] is None) == (want == 0)
    print("[OK] projection 合法/非法判定全部正确", flush=True)
    PASS.append("D")


def check_efg(steps: int, task: str, difficulty: str, max_steps: int):
    import ray
    from agent_system.environments.env_package.minecraft import (
        build_minecraft_envs, minecraft_projection)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    print("=== E. Ray + Malmo reset（1 组 × 2 worker） ===", flush=True)
    env_kwargs = {"tasks": [task], "difficulty": difficulty, "max_steps": max_steps, "seed": 0}
    t0 = time.time()
    envs = build_minecraft_envs(env_num=1, group_n=2, is_train=True, env_kwargs=env_kwargs)
    obs, infos = envs.reset()
    print(f"  reset 耗时 {time.time() - t0:.1f}s", flush=True)
    assert len(obs) == 2 and len(infos) == 2
    for i, (o, inf) in enumerate(zip(obs, infos)):
        o = np.asarray(o)
        print(f"  worker{i}: pov shape={o.shape} dtype={o.dtype} task={inf.get('task_name')} "
              f"desc={inf.get('task_description')!r}", flush=True)
        assert o.ndim == 3 and o.shape[2] == 3 and o.dtype == np.uint8
    assert infos[0]["task_name"] == infos[1]["task_name"], "同组 task 不一致"
    assert infos[0]["task_description"] == infos[1]["task_description"], "同组描述不一致"
    print("[OK] reset 通过，同组共享 task_config", flush=True)
    PASS.append("E")

    print(f"=== F. step 闭环 × {steps} ===", flush=True)
    texts = ["Action: move(0, 0) and press(w)", "Action: Walk forward"]  # 1 合法 + 1 非法
    for s in range(steps):
        acts, valids = minecraft_projection(list(texts))
        t0 = time.time()
        o2, rew, done, inf2 = envs.step(acts)
        print(f"  step{s}: valids={valids} reward={rew} done={done} "
              f"({time.time() - t0:.2f}s) info_keys={sorted(inf2[0].keys())}", flush=True)
        assert len(o2) == 2 and len(rew) == 2 and len(done) == 2 and len(inf2) == 2
        for inf in inf2:
            assert "won" in inf and "step_count" in inf
    print("[OK] step 契约成立（非法动作按 noop 执行、不崩溃）", flush=True)
    envs.close()
    PASS.append("F")

    print("=== G. MinecraftEnvironmentManager（verl 训练入口） ===", flush=True)
    from omegaconf import OmegaConf
    from agent_system.environments.env_manager import MinecraftEnvironmentManager
    cfg = OmegaConf.create({"env": {"history_length": 0, "max_steps": max_steps, "seed": 0,
                                    "minecraft": {"system_message_tag": "text_action"}}})
    envs2 = build_minecraft_envs(env_num=1, group_n=1, is_train=True, env_kwargs=env_kwargs)
    mgr = MinecraftEnvironmentManager(envs2, minecraft_projection, cfg)
    ob, _ = mgr.reset({})
    assert ob["image"].shape[0] == 1 and ob["image"].ndim == 4, ob["image"].shape
    txt = ob["text"][0]
    assert txt.endswith("<image>") and txt.startswith(mgr.system_message), "prompt 结构不符"
    print(f"  prompt 长度={len(txt)} 结尾={txt[-60:]!r}", flush=True)
    ob2, r2, d2, i2 = mgr.step(["Action: move(0, 0) and press(w)"])
    assert int(i2[0]["is_action_valid"]) == 1
    print(f"  step: reward={r2} done={d2} is_action_valid={int(i2[0]['is_action_valid'])}", flush=True)
    mgr.close()
    print("[OK] manager reset/step 通过", flush=True)
    PASS.append("G")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--task", default="mine_block:oak_log")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    run_d = not only or "d" in only
    run_env = not only or bool(only & {"e", "f", "g"})

    try:
        if run_d:
            check_d_projection()
        if run_env:
            check_efg(args.steps, args.task, args.difficulty, args.max_steps)
    except Exception:
        traceback.print_exc()

    expected = (1 if run_d else 0) + (3 if run_env else 0)
    ok = len(PASS) == expected and expected > 0
    print(f"\nSMOKE2_RESULT: {','.join(PASS) or '-'} "
          f"({len(PASS)}/{expected}) {'✅ 通过' if ok else '❌ 未全通过'}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

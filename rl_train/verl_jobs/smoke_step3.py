#!/usr/bin/env python3
"""第 3 步冒烟：把 dataloader→actor 之间**所有静态读代码无法判定的契约**一次性验完。

动机（09-30）：GRPO 冒烟每轮要等 ~8 小时集群排队才能暴露一个问题，逐个试错的
成本无法接受。本脚本把剩余疑点压进**一个** 1-GPU 任务，每项独立 try/except、
互不阻塞，一次跑完拿到全部答案。

验的是 `rollout_loop._preprocess_single_sample`（多轮 rollout 每步都会走）与
`verl/utils/dataset/rl_dataset.__getitem__`（dataloader）**这两处对同一个
processor 的假设是否成立、产出是否一致**——两者都要喂进同一个 actor 前向，
形状约定不一致就会在 compute_log_prob 炸，而这正是 smoke10 之后的下一站。

各层含义：
  H. processor 类名 —— 决定两处 mrope 门控各自走哪个分支（这是形状一致性的根）
  I. processor 属性 —— rollout 侧硬编码了 `processor.image_token`、
     `image_processor.merge_size`、`image_inputs['image_grid_thw']`，
     以及 qwen3_vl.get_rope_index 需要的 image_token_id/video_token_id/
     vision_start_token_id；任一缺失都是 AttributeError/KeyError
  J. get_rope_index 实跑 —— 拿真实 processor + 真实图跑一次，看输出行数
  K. **4 行 mrope 能否进 Qwen3_5 前向** —— 全流程最大的未知：qwen2_vl 适配器有
     process_position_ids 做校验/裁剪，qwen3_5 适配器完全没有、直接透传，
     能否接受 (4,bs,seq) 取决于 transformers 5.15 内部实现，静态读不出来
  L. prompt 实际 token 数 vs max_prompt_length —— truncation='error' 时超限
     即抛异常，而 system_prompt(1.8KB) + 640×360 POV 的 vision token 估算已
     逼近 1024，需要实测确认余量

用法（C 层同理，必须独立进程，因为 apply_monkey_patch 是类级补丁）：
    python smoke_step3.py --model-path /local-ssd/model_cache
"""
import argparse
import sys
import traceback

import numpy as np
import torch

PASS = []
FAIL = []


def _section(name):
    print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}", flush=True)


def _ok(layer, msg):
    PASS.append(layer)
    print(f"[OK] {msg}", flush=True)


def _fail(layer, msg):
    FAIL.append(layer)
    print(f"[FAIL] {msg}", flush=True)


def check_h_processor_class(model_path):
    """H：两处 mrope 门控分别会走哪个分支——形状一致性的根因。"""
    _section("H. processor / image_processor 类名（决定 mrope 门控分支）")
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    pname = proc.__class__.__name__
    iname = proc.image_processor.__class__.__name__
    print(f"processor      = {pname}", flush=True)
    print(f"image_processor= {iname}", flush=True)

    # rl_dataset.py:235 的外层门控
    loader_mrope = "Qwen2VLImageProcessor" in iname
    # rollout_loop.py:141 没有外层门控，只要 obs 有图就恒为 True
    rollout_mrope = True
    # 两处内层都用这个子串选 qwen3_vl / qwen2_vl 版本的 get_rope_index
    use_qwen3 = "Qwen3VLProcessor" in pname

    print(f"dataloader 走 mrope: {loader_mrope}  (门控: 'Qwen2VLImageProcessor' in image_processor 类名)", flush=True)
    print(f"rollout    走 mrope: {rollout_mrope}  (无门控，恒 True)", flush=True)
    print(f"get_rope_index 版本: {'qwen3_vl' if use_qwen3 else 'qwen2_vl'}", flush=True)

    if loader_mrope == rollout_mrope:
        _ok("H", f"两处 position_ids 约定一致（都{'走' if loader_mrope else '不走'} mrope）")
    else:
        _fail("H", f"⚠ 形状不一致！dataloader mrope={loader_mrope} 但 rollout={rollout_mrope} "
                   f"→ 同一 actor 会收到两种形状的 position_ids，compute_log_prob 必炸")
    return proc, use_qwen3


def check_i_processor_attrs(proc, use_qwen3):
    """I：rollout 侧硬编码的属性/键是否都存在。"""
    _section("I. rollout_loop 与 get_rope_index 依赖的 processor 属性")
    from PIL import Image

    missing = []
    # rollout_loop.py:115 / 127
    for attr, owner in [("merge_size", proc.image_processor), ("image_token", proc)]:
        val = getattr(owner, attr, None)
        if val is None:
            missing.append(attr)
            print(f"  ✗ {attr} 缺失", flush=True)
        else:
            print(f"  ✓ {attr} = {val!r}", flush=True)

    # qwen3_vl.get_rope_index:45-48 需要的三个 id（qwen2_vl 版本改用 tokenizer 查表）
    if use_qwen3:
        for attr in ("image_token_id", "video_token_id", "vision_start_token_id"):
            val = getattr(proc, attr, None)
            if val is None:
                missing.append(attr)
                print(f"  ✗ {attr} 缺失（qwen3_vl.get_rope_index 需要）", flush=True)
            else:
                print(f"  ✓ {attr} = {val}", flush=True)

    # rollout_loop.py:112 硬编码的键名
    img = Image.new("RGB", (640, 360), (90, 140, 70))
    feats = proc.image_processor([img], return_tensors="pt")
    print(f"  image_processor 输出键: {sorted(feats.keys())}", flush=True)
    if "image_grid_thw" not in feats:
        missing.append("image_grid_thw")
        print("  ✗ image_grid_thw 缺失（rollout_loop.py:112 硬编码此键）", flush=True)
    else:
        print(f"  ✓ image_grid_thw = {feats['image_grid_thw'].tolist()}", flush=True)

    if missing:
        _fail("I", f"缺失 {missing}")
    else:
        _ok("I", "rollout 侧所有硬编码属性/键均存在")
    return feats


def check_j_rope_index(proc, use_qwen3, feats):
    """J：用真实 processor 实跑 get_rope_index，确认输出行数。"""
    _section("J. get_rope_index 实跑（输出行数 = mrope 维数）")
    if use_qwen3:
        from verl.models.transformers.qwen3_vl import get_rope_index
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index

    grid = feats["image_grid_thw"]
    merge = proc.image_processor.merge_size ** 2
    n_img_tok = int(grid[0].prod() // merge)
    # 复刻 rollout_loop 展开后的 token 序列形态
    text = ("<|im_start|>user\n<|vision_start|>" + proc.image_token * n_img_tok
            + "<|vision_end|>hi<|im_end|>\n<|im_start|>assistant\n")
    ids = proc.tokenizer(text, return_tensors="pt")["input_ids"]
    attn = torch.ones_like(ids)

    vision_pos = get_rope_index(proc, input_ids=ids[0], image_grid_thw=grid, attention_mask=attn[0])
    print(f"vision_position_ids shape = {tuple(vision_pos.shape)}  (图像 token 数={n_img_tok})", flush=True)

    # rollout_loop.py:155-157 / rl_dataset.py:251-253 的拼接方式
    valid = attn[0].bool()
    text_pos = torch.ones((1, len(ids[0])), dtype=torch.long)
    text_pos[0, valid] = torch.arange(int(valid.sum()))
    combined = torch.cat((text_pos, vision_pos), dim=0)
    print(f"拼接后 position_ids shape = {tuple(combined.shape)}  (verl 约定: (4, seq))", flush=True)

    if combined.shape[0] == 4:
        _ok("J", f"position_ids 为 4 行（1 文本 + 3 视觉），与 verl 约定一致")
    else:
        _fail("J", f"position_ids 为 {combined.shape[0]} 行，verl 下游按 4 行处理")
    return ids, attn, combined, feats


def check_k_mrope_forward(model_path, ids, attn, combined, feats):
    """K：★最关键★ 4 行 mrope position_ids 能否进 Qwen3_5 的 PPO 前向。

    qwen2_vl 适配器有 process_position_ids() 做 (4,bs,seq) 校验，qwen3_5 适配器
    没有、直接 **kwargs 透传给 language_model。能否接受取决于 transformers 5.15
    内部实现，只能实跑。dp_actor.py:131-132 会把 dim()==3 的 position_ids
    transpose 成 (4,bs,seq) 再喂进来，这里复刻同样的形态。
    """
    _section("K. ★ 4 行 mrope position_ids → Qwen3_5 PPO 前向（最关键未知）")
    from transformers import AutoConfig, AutoModelForImageTextToText
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"model_type={cfg.model_type}", flush=True)
    rope = getattr(getattr(cfg, "text_config", cfg), "rope_parameters", None)
    print(f"rope_parameters={rope}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, config=cfg, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").cuda().eval()
    apply_monkey_patch(model, use_remove_padding=False,
                       use_fused_kernels=True, fused_kernels_backend="torch")

    ids_c, attn_c = ids.cuda(), attn.cuda()
    # dp_actor.py: (bs,4,seq) --transpose--> (4,bs,seq)
    pos = combined.unsqueeze(0).cuda().transpose(0, 1)
    print(f"喂入 position_ids shape = {tuple(pos.shape)}  (dp_actor transpose 后的真实形态)", flush=True)

    kw = {"input_ids": ids_c, "attention_mask": attn_c, "position_ids": pos,
          "pixel_values": feats["pixel_values"].cuda().to(torch.bfloat16),
          "image_grid_thw": feats["image_grid_thw"].cuda(),
          "temperature": 1.0, "use_cache": False}
    with torch.no_grad():
        out = model(**kw)

    checked = []
    for f in ("log_probs", "entropy", "logits"):
        v = getattr(out, f, None)
        if v is not None:
            assert torch.isfinite(v).all(), f"{f} 含 NaN/Inf"
            checked.append(f"{f}{tuple(v.shape)}")
    assert checked, f"前向无可校验字段（返回 {type(out).__name__}）"
    _ok("K", f"4 行 mrope 前向通过: {', '.join(checked)}")
    del model
    torch.cuda.empty_cache()


def check_l_prompt_length(model_path, proc):
    """L：真实 prompt 的 token 数 vs max_prompt_length=1024（truncation='error'）。"""
    _section("L. 真实 prompt token 数 vs max_prompt_length")
    import os
    from PIL import Image

    # 与 MinecraftEnvironmentManager._texts 同源：system_prompt(text_action) + 任务描述
    repo = os.environ.get("REPO_ROOT", "/data/work/run_codes/Minecraft-CoT")
    sp_path = os.path.join(repo, "openagents/assets/system_prompt/text_action.txt")
    system_prompt = open(sp_path).read() if os.path.exists(sp_path) else ""
    if not system_prompt:
        print(f"[WARN] 找不到 {sp_path}，仅测 vision token 部分", flush=True)
    instruction = "Break the tree to get oak logs."

    img = Image.new("RGB", (640, 360), (90, 140, 70))  # 真实 POV 分辨率
    feats = proc.image_processor([img], return_tensors="pt")
    merge = proc.image_processor.merge_size ** 2
    n_img_tok = int(feats["image_grid_thw"][0].prod() // merge)

    msgs = [{"role": "user", "content": f"{system_prompt}\n{instruction}<image>"}]
    templated = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
    expanded = templated.replace(
        "<image>", "<|vision_start|>" + proc.image_token * n_img_tok + "<|vision_end|>")
    n_tok = len(proc.tokenizer(expanded, add_special_tokens=False)["input_ids"])

    print(f"system_prompt 字符数 = {len(system_prompt)}", flush=True)
    print(f"640x360 POV 展开后的图像 token 数 = {n_img_tok}", flush=True)
    print(f"**完整 prompt token 数 = {n_tok}**  (上限 max_prompt_length=1024)", flush=True)
    if n_tok <= 1024:
        _ok("L", f"未超限，余量 {1024 - n_tok} token（{100 * n_tok / 1024:.1f}% 占用）")
    else:
        _fail("L", f"⚠ 超限 {n_tok - 1024} token！truncation='error' 下每步必抛 "
                   f"RuntimeError，必须调大 max_prompt_length 或缩小图像")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/local-ssd/model_cache")
    ap.add_argument("--only", default="", help="只跑指定层（逗号分隔，如 'h,i'）")
    args = ap.parse_args()
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}

    def want(x):
        return not only or x in only

    proc = use_qwen3 = feats = None
    ids = attn = combined = None

    # 每层独立 try：一层失败不阻塞后面几层，一次任务拿到尽可能多的信息
    if want("h"):
        try:
            proc, use_qwen3 = check_h_processor_class(args.model_path)
        except Exception:
            _fail("H", "异常"); traceback.print_exc()
    if want("i") and proc is not None:
        try:
            feats = check_i_processor_attrs(proc, use_qwen3)
        except Exception:
            _fail("I", "异常"); traceback.print_exc()
    if want("j") and feats is not None:
        try:
            ids, attn, combined, feats = check_j_rope_index(proc, use_qwen3, feats)
        except Exception:
            _fail("J", "异常"); traceback.print_exc()
    if want("k") and combined is not None:
        try:
            check_k_mrope_forward(args.model_path, ids, attn, combined, feats)
        except Exception:
            _fail("K", "异常"); traceback.print_exc()
    if want("l") and proc is not None:
        try:
            check_l_prompt_length(args.model_path, proc)
        except Exception:
            _fail("L", "异常"); traceback.print_exc()

    print(f"\n{'=' * 70}", flush=True)
    print(f"SMOKE3_RESULT: 通过={','.join(PASS) or '-'} 失败={','.join(FAIL) or '-'}", flush=True)
    print(f"{'=' * 70}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

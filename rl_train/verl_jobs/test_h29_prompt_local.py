"""本地验证 h29 多轮上下文 prompt（不需要 GPU）。

1. 字符串 <image> 渲染 == 评测侧 [text, image] 列表渲染（逐字）
2. _preprocess_chat_sample：token 数、position_ids 形状、超长时丢弃最旧历史
3. _materialize_mm_inputs：pixel_values / image_grid_thw 与 prompt 里的图像 token 数一致
"""
import importlib.abc, importlib.machinery, sys, types
from collections import deque

REAL_MISSING_OK = ("omegaconf", "tensordict", "ray", "codetiming", "datasets", "hydra",
                   "pandas", "pyarrow", "gym", "gymnasium", "wandb", "torchdata", "peft",
                   "accelerate", "liger_kernel", "flash_attn", "vllm", "openagents", "minestudio")


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in REAL_MISSING_OK:
            try:
                importlib.machinery.PathFinder.find_spec(name.split(".")[0])
            except Exception:
                pass
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__getattr__ = lambda attr: type(attr, (), {})
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Stub())
sys.path.insert(0, "/Users/axiom/Desktop/code/code/Minecraft-RL/rl_train/verl_agent")

import numpy as np
import torch
from transformers import AutoProcessor

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.utils import process_image

proc = AutoProcessor.from_pretrained("/tmp/ckpt_proc")
tok = proc.tokenizer
print("processor:", proc.__class__.__name__, "/", proc.image_processor.__class__.__name__)


class Cfg(dict):
    __getattr__ = dict.__getitem__

    def get(self, k, d=None):
        return dict.get(self, k, d)


def make_cfg(max_len):
    return Cfg(data=Cfg(max_prompt_length=max_len, truncation="error", return_raw_chat=True,
                        apply_chat_template_kwargs={"enable_thinking": False}),
               env=Cfg(rollout=Cfg(train_steps_per_traj=8)))


SYSTEM = open("/Users/axiom/Desktop/code/code/Minecraft-RL/openagents/assets/system_prompt/text_action.txt").read()
INSTR = "Break the tree to get oak logs."
rng = np.random.default_rng(0)
frames = {f"f{i}": process_image(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)) for i in range(40)}
RESP = ["Action: move(0, 0) and press(w)", "Action: move(12, -3) and click(left)",
        "Thought: tree ahead | holding nothing | approach\nAction: move(0, 0) and press(w)"]


def struct(h):
    return {"prefix": SYSTEM + INSTR,
            "turns": deque([(f"f{i}", RESP[i % 3]) for i in range(h)], maxlen=29),
            "current": f"f{h}"}


# ---------- 1. 与评测列表形式逐字一致 ----------
def eval_style_messages(s):
    """复刻 openha.gen_response + create_message_vllm 的消息结构（content 为列表）。"""
    msgs = []
    turns = list(s["turns"])
    for hdx, (fid, resp) in enumerate(turns):
        p = s["prefix"] if hdx == 0 else ""
        msgs.append({"role": "user", "content": [{"type": "text", "text": p}, {"type": "image"}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": resp}]})
    p = "" if turns else s["prefix"]
    msgs.append({"role": "user", "content": [{"type": "text", "text": p}, {"type": "image"}]})
    return msgs


for h in (0, 1, 5, 29):
    s = struct(h)
    mine, _ = TrajectoryCollector._render_chat_messages(s, len(s["turns"]))
    a = tok.apply_chat_template(mine, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    a = a.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    b = proc.apply_chat_template(eval_style_messages(s), add_generation_prompt=True, tokenize=False,
                                 enable_thinking=False)
    assert a == b, f"h={h} 渲染不一致\n--- mine ---\n{a[-600:]}\n--- eval ---\n{b[-600:]}"
    print(f"[1] h={h:2d}: 与评测列表形式逐字一致（{len(a)} 字符，图像 {a.count('<|image_pad|>')} 张）")

# ---------- 2. 预处理 ----------
gen_batch = types.SimpleNamespace(non_tensor_batch={"data_source": np.array(["visual"], dtype=object)})
col = TrajectoryCollector(make_cfg(10240), tok, proc)
for h in (0, 29):
    obs = {"chat": [struct(h)], "get_frame": frames.__getitem__, "anchor": ["a"]}
    row = col.preprocess_single_sample(0, gen_batch, obs)
    n_valid = int(row["attention_mask"].sum())
    n_img_tok = int((row["input_ids"] == proc.image_token_id).sum())
    print(f"[2] h={h:2d}: 有效 token {n_valid} / 上限 10240，图像 token {n_img_tok}，"
          f"position_ids {tuple(row['position_ids'].shape)}，帧 {len(row['mm_frame_ids'].split('|'))}，"
          f"vLLM 图 {len(row['multi_modal_data']['image'])}，raw_prompt_ids {len(row['raw_prompt_ids'])}")
    assert row["position_ids"].shape == (4, 10240)
    assert len(row["multi_modal_data"]["image"]) == h + 1
    assert row["raw_prompt_ids"].count(proc.image_token_id) == h + 1

# 超长：上限设成只够放 ~10 张图
small = TrajectoryCollector(make_cfg(3500), tok, proc)
row = small.preprocess_single_sample(0, gen_batch, {"chat": [struct(29)], "get_frame": frames.__getitem__, "anchor": ["a"]})
kept = len(row["mm_frame_ids"].split("|"))
print(f"[2] 超长回退：上限 3500 时保留 {kept - 1} 条历史 + 当前帧，有效 token {int(row['attention_mask'].sum())}")
assert int(row["attention_mask"].sum()) <= 3500 and kept < 30
# 回退后首条 user 仍带 system prompt
first_user = row["raw_prompt"][0]["content"]
assert first_user.startswith(SYSTEM[:50]), "回退后丢了 system prompt"
print("[2] 回退后第一条 user 仍以 system prompt 开头 ✓")

# ---------- 3. 物化 ----------
row = col.preprocess_single_sample(0, gen_batch, {"chat": [struct(29)], "get_frame": frames.__getitem__, "anchor": ["a"]})
n_img_tok = int((row["input_ids"] == proc.image_token_id).sum())
col._materialize_mm_inputs(row, frames.__getitem__)
mmi = row["multi_modal_inputs"]
merge = proc.image_processor.merge_size ** 2
print(f"[3] pixel_values {tuple(mmi['pixel_values'].shape)} ({mmi['pixel_values'].element_size() * mmi['pixel_values'].nelement() / 2**20:.0f} MB)，"
      f"image_grid_thw {tuple(mmi['image_grid_thw'].shape)}，grid 推出的图像 token {int(mmi['image_grid_thw'].prod(-1).sum()) // merge} == prompt 里 {n_img_tok}")
assert int(mmi["image_grid_thw"].prod(-1).sum()) // merge == n_img_tok
assert "mm_frame_ids" not in row
print("\n全部通过")

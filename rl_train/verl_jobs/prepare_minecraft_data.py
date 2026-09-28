"""生成 verl-agent 需要的占位 parquet（Minecraft 用，不联网）。

verl-agent 的多轮 rollout 里，真正的 prompt 与图像全部来自环境（见
MinecraftEnvironmentManager），数据集只起两个作用：①告诉 trainer 模态（有无图像）
②决定每轮 batch 的"样本数"（= 任务组数）。官方 examples/data_preprocess/prepare.py
为此从 HuggingFace 下载 geometry3k——在 koala 上既慢又会撞 HF 匿名限流（评测侧
下载 MineStudio 引擎时真实撞过 429）。这里改为本地生成一张小图，行为等价。

用法:
    python prepare_minecraft_data.py --out /local-ssd/verl_data --train 8 --val 4
输出:
    <out>/train.parquet, <out>/test.parquet
"""
import argparse
import io
import os

import pandas as pd
from PIL import Image


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 140, 70)).save(buf, format="PNG")
    return buf.getvalue()


def _rows(n: int, split: str):
    img = _png_bytes()
    return [{
        "data_source": "visual",
        # 占位：rollout 阶段会被环境生成的 system_prompt+instruction+<image> 替换
        "prompt": [{"role": "user", "content": "<image>"}],
        "images": [{"bytes": img}],
        "ability": "agent",
        "extra_info": {"split": split, "index": i},
    } for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/local-ssd/verl_data")
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--val", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(_rows(args.train, "train")).to_parquet(os.path.join(args.out, "train.parquet"))
    pd.DataFrame(_rows(args.val, "test")).to_parquet(os.path.join(args.out, "test.parquet"))
    print(f"[prepare] wrote {args.train} train / {args.val} val rows to {args.out}")


if __name__ == "__main__":
    main()

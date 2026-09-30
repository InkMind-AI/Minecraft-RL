"""生成 verl-agent 需要的占位 parquet（Minecraft 用，不联网）。

verl-agent 的多轮 rollout 里，真正的 prompt 与图像全部来自环境（见
MinecraftEnvironmentManager），数据集只起两个作用：①告诉 trainer 模态（有无图像）
②决定每轮 batch 的"样本数"（= 任务组数）。官方 examples/data_preprocess/prepare.py
为此从 HuggingFace 下载 geometry3k——在 koala 上既慢又会撞 HF 匿名限流（评测侧
下载 MineStudio 引擎时真实撞过 429）。这里改为本地生成一张小图，行为等价。

⚠ 09-30 smoke10 实测踩到的坑：verl 的 `vision_utils.process_image` 对
`{"bytes": <raw png bytes>}` 格式的处理是 `image["image"] = BytesIO(image["bytes"])`
再调 `qwen_vl_utils.fetch_image`——但 `fetch_image` 只认 `PIL.Image` 或者
`str`（http(s):// / file:// / data:image base64 / 本地路径），**完全不接受
BytesIO**，一撞上就是 `image.startswith(...)` 报 `AttributeError`
（BytesIO 没有 `startswith`）。这是 verl 自身 `vision_utils.py` 与当前
`qwen_vl_utils` 版本之间的不匹配，不是我们能改的第三方代码。绕开方式：
不走 `bytes` 分支，改用 `file://` 本地路径——这是 `fetch_image` 明确支持、
也是 HF/Qwen 生态最常用的占位数据写法。

用法:
    python prepare_minecraft_data.py --out /local-ssd/verl_data --train 8 --val 4
输出:
    <out>/train.parquet, <out>/test.parquet, <out>/placeholder.png
"""
import argparse
import os

import pandas as pd
from PIL import Image


def _write_placeholder_png(out_dir: str) -> str:
    path = os.path.join(out_dir, "placeholder.png")
    Image.new("RGB", (64, 64), (90, 140, 70)).save(path, format="PNG")
    return path


def _rows(n: int, split: str, image_path: str):
    return [{
        "data_source": "visual",
        # 占位：rollout 阶段会被环境生成的 system_prompt+instruction+<image> 替换
        "prompt": [{"role": "user", "content": "<image>"}],
        "images": [{"image": f"file://{image_path}"}],
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
    image_path = _write_placeholder_png(args.out)
    pd.DataFrame(_rows(args.train, "train", image_path)).to_parquet(os.path.join(args.out, "train.parquet"))
    pd.DataFrame(_rows(args.val, "test", image_path)).to_parquet(os.path.join(args.out, "test.parquet"))
    print(f"[prepare] wrote {args.train} train / {args.val} val rows to {args.out}")


if __name__ == "__main__":
    main()

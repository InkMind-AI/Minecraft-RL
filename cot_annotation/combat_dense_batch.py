"""结构化 CoT：Combat 定向扫描 + 全步密度标注（09-18，优化目标：不追求归因干净，只追求最优配方）。

复用 combat_batch.py 的扫描逻辑（找 kill_entity 类轨迹）+ structured_pipeline.py 的
全步标注逻辑（density=every）。目标：同时解决"信号太稀"和"Combat 配比过低"两个已知
瓶颈，产出一份结构化+稠密+配平的训练集。
"""
import argparse
import json
import os
import pickle

from .combat_batch import scan_combat_rows
from .structured_pipeline import run_structured
from .annotator import get_annotator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000, help="目标 combat 轨迹数")
    ap.add_argument("--shard-lo", type=int, default=250)
    ap.add_argument("--shard-hi", type=int, default=363)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    rows, scanned = scan_combat_rows(args.shard_lo, args.shard_hi, args.target)
    if not rows:
        print("[combat-dense] 未找到 combat 轨迹")
        return

    annotator = get_annotator()
    annotated, stats = run_structured(
        rows, annotator=annotator, n_workers=args.workers,
        checkpoint_dir=os.path.join(args.out, "ckpt"), density="every")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "annotated.pkl"), "wb") as f:
        pickle.dump(annotated, f)
    meta = dict(stats, n_scanned=scanned, n_combat_rows=len(rows),
                shard_range=[args.shard_lo, args.shard_hi], n_annotated_out=len(annotated))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"输出: {args.out}/annotated.pkl ({len(annotated)} 行)")


if __name__ == "__main__":
    main()

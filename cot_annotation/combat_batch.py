"""Combat 定向标注批（09-14 配比重构）：从训练集 shard 10-99 扫描 kill 类轨迹，
定向标注，补足 CoT 数据的 Combat 配比（现有 4% → 目标 ~20%+）。

在集群上运行（避免本地图传流量）：
    COT_ANNOTATOR=gemini COT_API_KEY=xxx COT_MODEL=gemini-3.8-flash \
    python3 -m cot_annotation.combat_batch --target 1600 --out /local-ssd/combat_out
"""

import argparse
import json
import os
import re
import sys

from .pipeline import run_pipeline, INSTRUCTION_RE, extract_row_fields
from .annotator import get_annotator

SHARD_FMT = ("s3://arcwm-code-us-west-2/axiom/data/"
             "minecraft-text-action-dataset-noop-filtered/data/train-{:05d}-of-00363.parquet")


def is_combat_row(row) -> bool:
    try:
        instruction, _, _ = extract_row_fields(row)
    except Exception:
        return False
    return bool(re.search(r"\bkill|combat|attack\b", instruction, re.I))


def scan_combat_rows(shard_lo: int, shard_hi: int, target: int, verbose=True):
    """按 shard 顺序扫描，收集 kill 类轨迹直到 target 条。"""
    import s3fs
    import pyarrow.parquet as pq
    fs = s3fs.S3FileSystem()
    rows, scanned, combat_found = [], 0, 0
    for sid in range(shard_lo, shard_hi + 1):
        if combat_found >= target:
            break
        path = SHARD_FMT.format(sid)
        try:
            with fs.open(path.replace("s3://", ""), "rb") as handle:
                pf = pq.ParquetFile(handle)
                for batch in pf.iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        scanned += 1
                        if is_combat_row(row) and len(row.get("image_bytes") or []) >= 10:
                            rows.append(row)
                            combat_found += 1
                            if combat_found >= target:
                                break
                    if combat_found >= target:
                        break
        except FileNotFoundError:
            continue
        if verbose and sid % 10 == shard_lo % 10:
            print(f"[scan] 扫到 shard {sid}: 累计读 {scanned} 行, "
                  f"combat {combat_found}/{target}", flush=True)
    print(f"[scan] 完成: 读 {scanned} 行, combat {combat_found} 条 "
          f"(密度 {100*combat_found/max(scanned,1):.1f}%)", flush=True)
    return rows, scanned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1600, help="目标 combat 轨迹数")
    ap.add_argument("--shard-lo", type=int, default=10, help="起始 shard（含），避开已标注的 0-9")
    ap.add_argument("--shard-hi", type=int, default=99, help="结束 shard（含），避开回放源 100-104")
    ap.add_argument("--out", required=True, help="本地输出目录")
    ap.add_argument("--workers", type=int, default=16, help="并行标注线程数")
    args = ap.parse_args()

    rows, scanned = scan_combat_rows(args.shard_lo, args.shard_hi, args.target)
    if not rows:
        print("[combat] 未找到 combat 轨迹，检查指令格式", flush=True)
        sys.exit(1)

    annotator = get_annotator()
    annotated, stats = run_pipeline(
        rows, annotator=annotator, n_workers=args.workers,
        checkpoint_dir=os.path.join(args.out, "ckpt"))

    os.makedirs(args.out, exist_ok=True)
    import pickle
    with open(os.path.join(args.out, "annotated.pkl"), "wb") as f:
        pickle.dump(annotated, f)
    meta = dict(stats, n_scanned=scanned, n_combat_rows=len(rows),
                shard_range=[args.shard_lo, args.shard_hi],
                n_annotated_out=len(annotated))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1), flush=True)
    print(f"输出: {args.out}/annotated.pkl ({len(annotated)} 行)", flush=True)


if __name__ == "__main__":
    main()

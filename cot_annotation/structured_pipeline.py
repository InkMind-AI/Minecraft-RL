"""结构化 CoT 标注管线（09-16 试点）：决策点 -> 确定性 phase/target -> LLM 仅判 visible/region -> 渲染。

复用 pipeline.py 的决策点检测与 annotator 接口，只替换 payload/校验/渲染三处。

用法（集群/CVM 上跑）：
    COT_ANNOTATOR=openai COT_API_KEY=xxx python3 -m cot_annotation.structured_pipeline \
        --parquet s3://.../train-XXXXX-of-00363.parquet --out /local-ssd/struct_out --limit 300
"""
import argparse
import copy
import json
import os
import pickle
import sys
from typing import Any, Dict, List

from .decision_points import parse_action, thought_points, action_phase, extract_target
from .structured_prompts import STRUCT_SYSTEM, build_struct_user, REGIONS
from .pipeline import extract_row_fields, run_pipeline as _unused  # noqa: F401 (复用行提取)
from .annotator import get_annotator


def build_struct_payload(instruction: str, actions: List[str], pts: List[int],
                         image_bytes: List[bytes]) -> Dict[str, Any]:
    target = extract_target(instruction)
    kf_idx = sorted(set([0] + pts))
    images = [image_bytes[i] for i in kf_idx if i < len(image_bytes)]
    return {
        "target": target,
        "keyframe_idx": kf_idx,
        "images": images,
        "system_text": STRUCT_SYSTEM,
        "user_text": build_struct_user(target, len(actions), pts),
    }


def validate_labels(raw: Dict[str, Any], pts: List[int]) -> Dict[int, Dict[str, Any]]:
    """校验：step 对齐 + visible 是 bool + region 在闭合词表内。"""
    out = {}
    for lb in raw.get("labels", []):
        try:
            step = int(lb["step"])
            visible = bool(lb["visible"])
            region = str(lb.get("region", "none")).strip().lower()
        except (KeyError, ValueError, TypeError):
            continue
        if step not in pts or region not in REGIONS:
            continue
        if not visible and region != "none":
            region = "none"  # 一致性修正，不视为丢弃
        out[step] = {"visible": visible, "region": region}
    return out


def render_struct_row(row: Dict[str, Any], instruction: str, actions: List[str],
                      pts: List[int], labels: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """把结构化 Thought 插入 assistant turn：
    Thought: target=<X> visible=<bool> region=<enum> phase=<enum>
    """
    target = extract_target(instruction)
    new_row = copy.deepcopy(row)
    idx = 0
    for c in new_row["conversations"]:
        if c.get("role") != "assistant":
            continue
        if idx in labels and idx < len(actions):
            lb = labels[idx]
            phase = action_phase(parse_action(actions[idx]))
            struct_line = (f"Thought: target={target} visible={str(lb['visible']).lower()} "
                          f"region={lb['region']} phase={phase}")
            orig = "".join(it.get("text", "") for it in c["content"] if it.get("type") == "text")
            new_text = f"{struct_line}\n{orig}"
            for it in c["content"]:
                if it.get("type") == "text":
                    it["text"] = new_text
                    break
            else:
                c["content"].insert(0, {"type": "text", "text": new_text})
        idx += 1
    return new_row


def run_structured(rows: List[Dict[str, Any]], annotator=None, n_workers: int = 8,
                   checkpoint_dir: str = None):
    import time as _time
    annotator = annotator or get_annotator()
    stats = {"n_rows": 0, "n_decision_points": 0, "n_labels_ok": 0,
             "n_labels_dropped": 0, "n_visible_true": 0}
    annotated = []

    def _one(row):
        instruction, actions, convs = extract_row_fields(row)
        if len(actions) < 10 or not row.get("image_bytes"):
            return None, dict(n_pts=0, n_ok=0, n_vis=0)
        pts = thought_points(actions, min_gap=2, use_r4=False)
        if not pts:
            return None, dict(n_pts=0, n_ok=0, n_vis=0)
        payload = build_struct_payload(instruction, actions, pts, row["image_bytes"])
        try:
            raw = annotator.annotate(payload)
        except Exception as e:
            print(f"  [skip] row={row.get('id','?')}: {e}", file=sys.stderr, flush=True)
            raw = {"labels": []}
        labels = validate_labels(raw, pts)
        n_vis = sum(1 for v in labels.values() if v["visible"])
        return (render_struct_row(row, instruction, actions, pts, labels),
                dict(n_pts=len(pts), n_ok=len(labels), n_vis=n_vis))

    CHUNK = 50
    t0 = _time.time()
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start:start + CHUNK]
        if n_workers <= 1:
            results = [_one(r) for r in chunk]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_one, chunk))
        for new_row, s in results:
            stats["n_rows"] += 1
            stats["n_decision_points"] += s["n_pts"]
            stats["n_labels_ok"] += s["n_ok"]
            stats["n_labels_dropped"] += s["n_pts"] - s["n_ok"]
            stats["n_visible_true"] += s["n_vis"]
            if new_row is not None:
                annotated.append(new_row)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, "struct_partial.pkl"), "wb") as f:
                pickle.dump(annotated, f)
        rate = (start + len(chunk)) / max(_time.time() - t0, 1e-6)
        print(f"[prog] {start+len(chunk)}/{len(rows)} | {rate*60:.0f}条/分 | "
              f"labels_ok={stats['n_labels_ok']} visible={stats['n_visible_true']}", flush=True)
    return annotated, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    paths = []
    for p in args.parquet:
        if any(c in p for c in "*?["):
            import glob as globlib
            if p.startswith("s3://"):
                import s3fs
                fs = s3fs.S3FileSystem()
                paths.extend("s3://" + x for x in fs.glob(p.replace("s3://", "")))
            else:
                paths.extend(sorted(globlib.glob(p)))
        else:
            paths.append(p)
    rows = []
    for p in paths:
        if len(rows) >= args.limit:
            break
        if p.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem()
            handle = fs.open(p.replace("s3://", ""), "rb")
        else:
            handle = p
        pf = pq.ParquetFile(handle)
        for batch in pf.iter_batches(batch_size=64):
            rows.extend(batch.to_pylist())
            if len(rows) >= args.limit:
                break
    rows = rows[:args.limit]

    annotated, stats = run_structured(rows, n_workers=args.workers,
                                      checkpoint_dir=os.path.join(args.out, "ckpt"))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "annotated.pkl"), "wb") as f:
        pickle.dump(annotated, f)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"输出: {args.out}/annotated.pkl ({len(annotated)} 行)")


if __name__ == "__main__":
    main()

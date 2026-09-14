"""端到端标注管线：parquet 行 -> 决策点检测 -> payload -> 标注 -> 渲染插入 -> 输出。

用法（dry run，Stub 模式）:
    python3 -m cot_annotation.pipeline --sample /tmp/cot_sample.pkl --out /tmp/cot_out
真实标注（API 到位后，建议在集群上跑以避免本地图像上传流量）:
    COT_ANNOTATOR=openai COT_API_KEY=sk-... python3 -m cot_annotation.pipeline \
        --parquet s3://.../train-00000-of-00363.parquet --out s3://.../cot-pilot/
"""

import argparse
import copy
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List

from .decision_points import parse_action, thought_points
from . import prompts
from .annotator import get_annotator

INSTRUCTION_RE = re.compile(r"## User Instruction\s*\n\s*(.+?)\s*$", re.M)
MAX_THOUGHT_WORDS = 45  # 渲染前的词数上限（超限丢弃，回退无thought）

# 不可验证断言黑名单（09-14 案例分析定论：Combat 幻觉的直接来源——"已消灭/正在掉血/
# 在射程内/任务完成"均无法从单帧验证，错误断言 + 历史回放会锁死错误行为）。
# 命中即丢弃该条 thought（不是整条轨迹）。
UNVERIFIABLE_RE = re.compile(
    r"(has been killed|has died|is killed|killed the|was killed|eliminated|defeated|"
    r"task is complete|task complete|has been slain|"
    r"taking damage|dealing damage|dealt damage|"
    r"within (melee )?(attack |striking )?range|in attack range|within range|within reach)",
    re.I)


def extract_row_fields(row: Dict[str, Any]):
    """从 parquet 行提取 (instruction, actions, conversations)。"""
    convs = row["conversations"]
    first_text = "".join(
        it.get("text", "") for it in convs[0]["content"] if it.get("type") == "text")
    m = INSTRUCTION_RE.search(first_text)
    instruction = m.group(1).strip() if m else "(unknown task)"
    actions = [
        "".join(it.get("text", "") for it in c["content"] if it.get("type") == "text")
        for c in convs if c.get("role") == "assistant"
    ]
    return instruction, actions, convs


def build_action_log(actions: List[str], pts: List[int]) -> str:
    """per-chunk 动作日志：'>>' 标记决策点（该步的帧会作为 keyframe 提供）。"""
    lines = []
    for i, a in enumerate(actions):
        mark = ">>" if i in pts else "  "
        lines.append(f"{mark} step {i:02d}: {a.strip()}")
    return "\n".join(lines)


def build_payload(instruction: str, actions: List[str], pts: List[int],
                  image_bytes: List[bytes]) -> Dict[str, Any]:
    """组装单条轨迹的标注 payload（论证通道；盲预测通道在 pilot 脚本中启用）。"""
    # keyframes = 首帧 + 各决策点帧（帧 i 对应动作 i 之前的画面，
    # 与 parquet 的 image_bytes 顺序一致：第 i 个 assistant turn 的图在 image_bytes[i]）
    kf_idx = sorted(set([0] + pts))
    images = [image_bytes[i] for i in kf_idx if i < len(image_bytes)]
    return {
        "instruction": instruction,
        "n_steps": len(actions),
        "decision_points": [
            {"step": t, "prev_action": actions[t - 1] if t > 0 else "(start)",
             "action": actions[t]}
            for t in pts
        ],
        "keyframe_idx": kf_idx,
        "images": images,
        "system_text": prompts.JUSTIFY_SYSTEM,
        "user_text": prompts.build_justify_user(
            instruction, build_action_log(actions, pts)),
    }


def validate_thoughts(raw: Dict[str, Any], pts: List[int]) -> Dict[int, str]:
    """校验返回的 thoughts：step 对齐 + 非空 + 词数限制。返回 {step: thought}。"""
    out = {}
    for th in raw.get("thoughts", []):
        try:
            step = int(th["step"])
            text = str(th["thought"]).strip().replace("\n", " ")
        except (KeyError, ValueError, TypeError):
            continue
        if step not in pts or not text:
            continue
        if len(text.split()) > MAX_THOUGHT_WORDS:
            continue
        if UNVERIFIABLE_RE.search(text):
            continue
        out[step] = text
    return out


def render_annotated_row(row: Dict[str, Any], thoughts: Dict[int, str]) -> Dict[str, Any]:
    """把 Thought 插入 assistant turn（原行不动，返回新行）。

    训练格式（与现有 text_action 对齐，仅 assistant 文本前加一行）:
        Thought: <...>
        Action: move(...) and press(...)
    """
    new_row = copy.deepcopy(row)
    idx = 0
    for c in new_row["conversations"]:
        if c.get("role") != "assistant":
            continue
        if idx in thoughts:
            orig = "".join(it.get("text", "") for it in c["content"]
                           if it.get("type") == "text")
            new_text = f"Thought: {thoughts[idx]}\n{orig}"
            for it in c["content"]:
                if it.get("type") == "text":
                    it["text"] = new_text
                    break
            else:
                c["content"].insert(0, {"type": "text", "text": new_text})
        idx += 1
    return new_row


def run_pipeline(rows: List[Dict[str, Any]], annotator=None,
                 verbose: bool = True, n_workers: int = 1,
                 checkpoint_dir: str = None) -> Dict[str, Any]:
    """主流程。返回 (annotated_rows, stats)。

    - n_workers>1 时多线程并行标注
    - checkpoint_dir 非空时启用断点续跑：按轨迹id记录已完成，重跑自动跳过；
      每 50 条打印进度，每 500 条增量落盘 annotated_partial.pkl
    """
    import time as _time
    annotator = annotator or get_annotator()

    done_ids = set()
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "done_ids.txt")):
        with open(os.path.join(checkpoint_dir, "done_ids.txt")) as f:
            done_ids = set(line.strip() for line in f if line.strip())
        print(f"[ckpt] 断点续跑: 已完成 {len(done_ids)} 条，跳过", flush=True)

    todo = [r for r in rows if str(r.get("id")) not in done_ids]
    stats = {"n_rows": 0, "n_steps": 0, "n_decision_points": 0,
             "n_thoughts_ok": 0, "n_thoughts_dropped": 0}

    def _one(row):
        instruction, actions, convs = extract_row_fields(row)
        if len(actions) < 10 or not row.get("image_bytes"):
            return None, dict(n_steps=0, n_pts=0, n_ok=0)
        pts = thought_points(actions, min_gap=2, use_r4=False)
        if not pts:
            return None, dict(n_steps=len(actions), n_pts=0, n_ok=0)
        payload = build_payload(instruction, actions, pts, row["image_bytes"])
        try:
            raw = annotator.annotate(payload)
        except Exception as e:
            if verbose:
                print(f"  [skip] row={row.get('id','?')}: {e}", file=sys.stderr, flush=True)
            raw = {"thoughts": []}
        thoughts = validate_thoughts(raw, pts)
        return (render_annotated_row(row, thoughts),
                dict(n_steps=len(actions), n_pts=len(pts), n_ok=len(thoughts)))

    annotated = []
    n_todo = len(todo)
    t0 = _time.time()
    CHUNK = 50
    for start in range(0, n_todo, CHUNK):
        chunk = todo[start:start + CHUNK]
        if n_workers <= 1:
            results = [_one(r) for r in chunk]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_one, chunk))
        # 统计 + 断点记录（chunk 原子提交：本 chunk 全部完成后才记 done）
        ckpt_ids = []
        for row, (new_row, s) in zip(chunk, results):
            rid = str(row.get("id"))
            if new_row is None and s["n_pts"] == 0:
                # 无决策点/太短的行也算处理完，避免重跑重复扫
                ckpt_ids.append(rid)
                stats["n_steps"] += s["n_steps"]
                continue
            stats["n_rows"] += 1
            stats["n_steps"] += s["n_steps"]
            stats["n_decision_points"] += s["n_pts"]
            stats["n_thoughts_ok"] += s["n_ok"]
            stats["n_thoughts_dropped"] += s["n_pts"] - s["n_ok"]
            if new_row is not None:
                annotated.append(new_row)
            ckpt_ids.append(rid)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, "done_ids.txt"), "a") as f:
                f.write("\n".join(ckpt_ids) + "\n")
            if (start // CHUNK) % 10 == 9:  # 每500条增量落盘
                with open(os.path.join(checkpoint_dir, "annotated_partial.pkl"), "wb") as f:
                    pickle.dump(annotated, f)
        done_total = len(done_ids) + start + len(chunk)
        rate = (start + len(chunk)) / max(_time.time() - t0, 1e-6)
        eta = (n_todo - start - len(chunk)) / max(rate, 1e-6) / 60
        print(f"[prog] {done_total}/{len(done_ids)+n_todo} 完成 | "
              f"{rate*60:.0f}条/分 | ETA {eta:.0f}分 | "
              f"ok={stats['n_thoughts_ok']} drop={stats['n_thoughts_dropped']}", flush=True)
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "annotated_partial.pkl"), "wb") as f:
            pickle.dump(annotated, f)
    return annotated, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="pickle 缓存的行列表（dry run 用）")
    ap.add_argument("--parquet", nargs="+", help="parquet 路径（可多个/glob，本地或 s3://）")
    ap.add_argument("--out", required=True, help="输出目录（本地路径）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少行（0=全部）")
    ap.add_argument("--workers", type=int, default=1, help="并行标注线程数")
    args = ap.parse_args()

    if args.sample:
        with open(args.sample, "rb") as f:
            rows = pickle.load(f)
    elif args.parquet:
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
            if args.limit and len(rows) >= args.limit:
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
                if args.limit and len(rows) >= args.limit:
                    break
        if args.limit:
            rows = rows[:args.limit]
    else:
        ap.error("--sample 或 --parquet 必须提供一个")

    annotated, stats = run_pipeline(rows, n_workers=args.workers,
                                    checkpoint_dir=os.path.join(args.out, "ckpt"))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "annotated.pkl"), "wb") as f:
        pickle.dump(annotated, f)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"输出: {args.out}/annotated.pkl ({len(annotated)} 行)")


if __name__ == "__main__":
    main()

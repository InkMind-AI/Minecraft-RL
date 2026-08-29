"""Convert our `minecraft-text-action-dataset` parquet trajectories into the
`conversations` + `image` (on-disk file) jsonl layout that OpenHA/VeOmni's
`craftjarvis_sft_preprocess` / `train_qwen2_5_vl.py::process_sample` expect.

Used to train the OpenHA VeOmni baseline on the SAME data our own trl_sft Stage III
runs consume, so the two are directly comparable.

Differences vs. the earlier smoke-test-only converter that lived in the OpenHA tree
(`VeOmni/tools/convert_text_action_to_jsonl.py`):

  * Writes the ORIGINAL image bytes straight to disk instead of `PIL.Image.open(...)
    .convert("RGB").save(<png>)`. The parquet's `image_bytes` entries are already
    JPEG (verified: magic `ffd8ffe0`, 640x360, ~17.7KB each), so re-encoding to PNG
    inflated every frame 7.5x (~132KB) -- across the full dataset's 6.04M frames that
    is ~800GB of PNG vs. ~107GB written as-is, plus a full decode+encode round trip
    per frame. Copying bytes keeps the format the dataloader already handles (it just
    calls `Image.open(path)`) and is I/O-bound rather than CPU-bound.
  * Accepts MANY shards (glob or repeated --parquet_path) and streams them row-group
    by row-group, so peak memory stays flat instead of materializing an entire shard
    via `to_pylist()`.
  * `--limit` defaults to 0 = "convert everything" (the old default of 50 silently
    produced a 50-row smoke-test file).
  * Parallel across shards with `--num_workers`, each worker owning its own output
    jsonl shard (then the caller passes the whole directory / a glob to VeOmni).

Input row schema (parquet, minecraft-text-action-dataset):
  id: str
  conversations: list[{"role": "user"|"assistant",
                       "content": [{"type": "text", "text": ...} | {"type": "image"}]}]
  image_bytes: list[bytes]   # one entry per {"type": "image"} placeholder, in the
                             # order those placeholders appear across the whole
                             # conversation (flat index, NOT per-turn).

Output row schema (jsonl, VeOmni craftjarvis format):
  id: str
  conversations: list[...]   # unchanged -- VeOmni consumes the same content-item
                             # schema we already store.
  image: list[{"image_path": <path>, "resolution": [0, 0]}]   # same flat
                             # encounter-order convention as image_bytes.
"""

import argparse
import glob as globlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq

# `conversations`/`image_bytes` come back from Arrow as numpy arrays; json.dumps
# cannot serialize those, and VeOmni expects plain lists/dicts.
def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy scalar/array
        return _to_plain(obj.tolist())
    return obj


def convert_one_shard(parquet_path: str, out_dir: str, image_root: str, shard_tag: str, limit: int) -> tuple:
    """Convert a single parquet shard. Returns (n_written, n_skipped, jsonl_path).

    `out_dir` must contain ONLY the .jsonl files: VeOmni's `build_dataset` does a bare
    `os.listdir(data_path)` and treats every entry as a data file, so an `images/`
    subdir or a stray marker file sitting next to them makes it try to load those as
    datasets. Images therefore go under a separate `image_root`.
    """
    image_dir = os.path.join(image_root, shard_tag)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, f"text_action_{shard_tag}.jsonl")

    n_written, n_skipped = 0, 0
    pf = pq.ParquetFile(parquet_path)
    with open(jsonl_path, "w") as out_f:
        # Stream row groups so a shard never has to fit in memory all at once.
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["id", "conversations", "image_bytes"])
            for row in table.to_pylist():
                if limit and n_written >= limit:
                    return n_written, n_skipped, jsonl_path

                row_id = str(row.get("id"))
                conversations = row.get("conversations") or []
                image_bytes_list = row.get("image_bytes") or []
                if not conversations or not image_bytes_list:
                    n_skipped += 1
                    continue

                image_field = []
                ok = True
                for i, raw in enumerate(image_bytes_list):
                    if not raw:
                        ok = False
                        break
                    # Write the stored bytes verbatim: they are already JPEG, and the
                    # dataloader only ever does Image.open(path). No decode/encode.
                    img_path = os.path.join(image_dir, f"{row_id}_{i}.jpg")
                    with open(img_path, "wb") as img_f:
                        img_f.write(raw)
                    image_field.append({"image_path": img_path, "resolution": [0, 0]})
                if not ok:
                    n_skipped += 1
                    continue

                out_f.write(
                    json.dumps(
                        {
                            "id": row_id,
                            "conversations": _to_plain(conversations),
                            "image": image_field,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1
    return n_written, n_skipped, jsonl_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet_path",
        type=str,
        action="append",
        required=True,
        help="Parquet shard path or glob. Repeatable.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for the .jsonl shards ONLY -- pass this to VeOmni's "
        "--data.train_path. Nothing else may live here (VeOmni os.listdir()s it and "
        "treats every entry as a dataset file).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Directory for the extracted image files (default: <out_dir>_images). "
        "Kept OUTSIDE --out_dir on purpose, see above.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows PER SHARD (0 = no limit, convert everything).",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Shards converted in parallel.")
    args = parser.parse_args()

    image_root = args.image_root or (args.out_dir.rstrip("/") + "_images")

    shards = []
    for pattern in args.parquet_path:
        matched = sorted(globlib.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matched:
            raise SystemExit(f"No parquet files matched: {pattern!r}")
        shards.extend(matched)
    shards = sorted(dict.fromkeys(shards))  # de-dup, keep order

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(image_root, exist_ok=True)
    print(f"Converting {len(shards)} shard(s) with {args.num_workers} worker(s)")
    print(f"  jsonl  -> {args.out_dir}")
    print(f"  images -> {image_root}")

    total_written, total_skipped = 0, 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {}
        for shard in shards:
            tag = os.path.splitext(os.path.basename(shard))[0]
            futures[pool.submit(convert_one_shard, shard, args.out_dir, image_root, tag, args.limit)] = shard
        for done, fut in enumerate(as_completed(futures), 1):
            shard = futures[fut]
            n_written, n_skipped, jsonl_path = fut.result()
            total_written += n_written
            total_skipped += n_skipped
            print(
                f"[{done}/{len(shards)}] {os.path.basename(shard)}: "
                f"{n_written} rows ({n_skipped} skipped) -> {os.path.basename(jsonl_path)}",
                flush=True,
            )

    print(f"DONE: {total_written} rows written, {total_skipped} skipped, across {len(shards)} shard(s)")
    print(f"Pass this DIRECTORY to VeOmni --data.train_path: {args.out_dir}")


if __name__ == "__main__":
    main()

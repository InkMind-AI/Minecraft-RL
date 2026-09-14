#!/bin/bash
# CoT 标注 pilot 提交脚本（API 到位后使用）
#
# 用法（在集群上提交，避免本地图像上传流量）：
#   COT_ANNOTATOR=openai COT_API_KEY=sk-xxx bash cot_annotation/submit_pilot.sh
#   COT_ANNOTATOR=gemini COT_API_KEY=xxx  bash cot_annotation/submit_pilot.sh
#
# 可调参数（环境变量）：
#   COT_MODEL     模型名（默认 gpt-4o / gemini-2.0-flash）
#   COT_PILOT_N   处理轨迹数（默认 2000）
#   COT_SHARD     起始 parquet 分片号（默认 0）
set -euo pipefail

ANNOTATOR="${COT_ANNOTATOR:?需设置 COT_ANNOTATOR=openai|gemini}"
API_KEY="${COT_API_KEY:?需设置 COT_API_KEY}"
MODEL="${COT_MODEL:-}"
PILOT_N="${COT_PILOT_N:-2000}"
SHARD="${COT_SHARD:-0}"
TS=$(date +%Y%m%d-%H%M%S)
JOB="axiomjin-cot-pilot-${ANNOTATOR}-${TS}"

SHARD_IDX=$(printf "%05d" "$SHARD")
PARQUET="s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset-noop-filtered/data/train-${SHARD_IDX}-of-00363.parquet"
OUT="s3://arcwm-code-us-west-2/axiom/data/cot-annotated/pilot-${ANNOTATOR}-${TS}/"

echo "[pilot] annotator=${ANNOTATOR} model=${MODEL:-<默认>} n=${PILOT_N} shard=${SHARD}"
echo "[pilot] 输出: ${OUT}"

koala submit -m normal -j "${JOB}" -g 0 \
  -c "set -euo pipefail; cd /data/work/run_codes/Minecraft-CoT; pip install -q s3fs 2>&1 | tail -1; export COT_ANNOTATOR=${ANNOTATOR}; export COT_API_KEY='${API_KEY}'; ${MODEL:+export COT_MODEL=${MODEL};} python3 -m cot_annotation.pipeline --parquet ${PARQUET} --out /local-ssd/cot_out --limit ${PILOT_N} 2>&1 | tee /local-ssd/cot_stats.json; aws s3 sync /local-ssd/cot_out/ ${OUT} --only-show-errors; echo '[pilot] 完成'" \
  --code "s3://arcwm-code-us-west-2/axiom/code:/data/work/run_codes" \
  --large-ssd --s3-log -y

#!/bin/bash
# 标注器模型对比：同一批40条轨迹，6个模型逐一标注，输出统一对比报告
# 用法: nohup bash compare_models.sh > cmp.log 2>&1 &
set -uo pipefail
cd ~/work

export COT_ANNOTATOR=openai
export COT_BASE_URL=http://v2.open.venus.oa.com/llmproxy
export COT_API_KEY="${COT_API_KEY:?需先设置 COT_API_KEY 环境变量（Venus 网关密钥不入库）}"
export AWS_DEFAULT_REGION=us-west-2

MODELS="gemini-3.8-flash qwen3.8-flash-next deepseek-v4-flash-vision-official gpt-5.6-terra gpt-5.6-luna claude-sonnet-5"

# 1) 准备固定样本（分片0前40行，全模型共用）
if [ ! -f cot_cmp_sample.pkl ]; then
python3 - <<'PYEOF'
import s3fs, pyarrow.parquet as pq, pickle
fs = s3fs.S3FileSystem(anon=False)
path = 'arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset-noop-filtered/data/train-00000-of-00363.parquet'
with fs.open(path, 'rb') as f:
    rows = next(pq.ParquetFile(f).iter_batches(batch_size=40)).to_pylist()
pickle.dump(rows, open('cot_cmp_sample.pkl','wb'))
print(f'样本就绪: {len(rows)} 条')
PYEOF
fi

# 2) 逐模型标注
mkdir -p cot_cmp
for M in $MODELS; do
  echo "===== [$M] 开始: $(date +%H:%M:%S)"
  if [ -f "cot_cmp/$M/annotated.pkl" ]; then echo "已存在，跳过"; continue; fi
  mkdir -p "cot_cmp/$M"
  COT_MODEL="$M" timeout 3600 python3 -m cot_annotation.pipeline \
    --sample cot_cmp_sample.pkl --out "cot_cmp/$M" --workers 4 2>&1 | tail -12
  echo "===== [$M] 结束: $(date +%H:%M:%S)"
done

# 3) 对比报告
python3 - <<'PYEOF'
import pickle, re, statistics, os, json, time
MODELS = "gemini-3.8-flash qwen3.8-flash-next deepseek-v4-flash-vision-official gpt-5.6-terra gpt-5.6-luna claude-sonnet-5".split()
print(f"{'模型':<38}{'轨迹':>5}{'thought':>9}{'通过率':>8}{'词数均值':>9}{'三段率':>8}{'distinct2':>10}{'秒/条':>7}")
results = {}
for m in MODELS:
    p = f"cot_cmp/{m}/annotated.pkl"
    if not os.path.exists(p):
        print(f"{m:<38}{'失败/超时':>8}"); continue
    rows = pickle.load(open(p,'rb'))
    thoughts, per = [], []
    for r in rows:
        n = 0
        for c in r["conversations"]:
            if c.get("role")!="assistant": continue
            t = "".join(it.get("text","") for it in c["content"] if it.get("type")=="text")
            mm = re.match(r"Thought: (.+?)\n", t)
            if mm: thoughts.append(mm.group(1)); n += 1
        per.append(n)
    if not thoughts:
        print(f"{m:<38}{len(rows):>5}{0:>9}{'0%':>8}  (无有效thought)"); continue
    wl = [len(t.split()) for t in thoughts]
    seg = sum(1 for t in thoughts if t.count("|")>=2)/len(thoughts)
    def ng(s,n):
        w=s.lower().split(); return set(tuple(w[i:i+n]) for i in range(len(w)-n+1))
    d2 = len(set().union(*[ng(t,2) for t in thoughts]))
    # 通过率 = 有效thought数 / 该样本决策点总数(约40*3.7≈148)
    results[m] = dict(thoughts=thoughts, rows=len(rows))
    log = open(f"cot_cmp/{m}/../{m}_log.txt").read() if os.path.exists(f"cot_cmp/{m}_log.txt") else ""
    print(f"{m:<38}{len(rows):>5}{len(thoughts):>9}{'':>8}{statistics.mean(wl):>9.1f}{seg:>8.0%}{d2:>10}{0:>7}")

# 抽样对比
print("\n\n======== 每模型随机3条 thought 对比 ========")
import random
random.seed(42)
for m, r in results.items():
    print(f"\n--- {m} ---")
    for t in random.sample(r["thoughts"], min(3, len(r["thoughts"]))):
        print("  *", t[:170])
PYEOF
echo "===== 全部完成: $(date +%H:%M:%S)"

#!/usr/bin/env bash
# 组一个候选训练 env `vllmtrain` = vllm35 克隆（torch 2.10 / vllm 0.17）+ transformers 5.x
# （补上 Qwen3_5ForConditionalGeneration）+ 线性注意力内核，然后跑 probe_colocate.py。
#
# trainenv-p1 的结论（09-28）：
#   - vllm35: torch 2.10 / transformers 4.57.6 / vllm 0.17 —— transformers 里**没有**
#     Qwen3_5，FSDP 训练侧加载不了模型（vLLM 自带模型实现所以推理没问题）
#   - sft env: torch 2.6 / transformers 5.15 —— 有 Qwen3_5，但 vllm 0.17 要 torch 2.10
#   - vendored verl 的 vllm glue 按 0.8-0.11 写：vllm_async_server
#     (vllm.entrypoints.openai.protocol) 和 fsdp_vllm (vllm.lora.models) 在 0.17 下
#     import 失败
# 所以唯一可能的 colocated 组合是 "torch 2.10 + vllm 0.17 + transformers 5"，本探针
# 验证它能否成立。
set +e
source /opt/conda/etc/profile.d/conda.sh
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

if ! conda env list | grep -qE '^vllm35 '; then
  conda create -n vllm35 python=3.11 -y >/dev/null 2>&1
  conda activate vllm35 && pip install -q --no-cache-dir 'vllm==0.17.0'
  conda deactivate
fi
if ! conda env list | grep -qE '^vllmtrain '; then
  echo "[probe] cloning vllm35 -> vllmtrain"
  conda create -n vllmtrain --clone vllm35 -y >/dev/null 2>&1
fi
conda activate vllmtrain

echo "[probe] installing transformers 5.15.0 (pip 可能报 vllm 的版本约束冲突——记录下来，不中断)"
pip install -q 'transformers==5.15.0' 2>&1 | tail -4
pip check 2>&1 | grep -iE "vllm|transformers|torch" | head -8

echo "[probe] 线性注意力内核"
python -c 'import fla' 2>/dev/null || pip install -q flash-linear-attention 2>&1 | tail -2
python -c 'import causal_conv1d' 2>/dev/null || \
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d 2>&1 | tail -3

echo "[probe] 下载权重"
aws s3 cp s3://arcwm-code-us-west-2/axiom/model/qwen35-9b-nf2-c3000-slim/ /local-ssd/model_cache/ \
  --recursive --exclude 'global_step*' --only-show-errors

python rl_train/verl_jobs/probe_colocate.py --model /local-ssd/model_cache
echo COLOCATE_PROBE_DONE

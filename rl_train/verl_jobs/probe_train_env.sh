#!/usr/bin/env bash
# 探针：以 vllm35 env（vllm 0.17，评测已验证能服务 Qwen3.5）为基座，能否组出
# verl 训练 env。回答四个问题，每条独立 try，不因前一条失败而中断：
#
#   Q1 vllm35 里 torch / transformers / vllm 版本？transformers 有没有 Qwen3_5？
#   Q2 vendored verl 的 vllm_rollout 模块在 vllm 0.17 下能否 import？
#      （它按 vllm 0.8-0.11 写；0.17 挪走/删掉的内部 API 会在这里暴露）
#   Q3 fla / causal-conv1d / flash-attn 能否装进这个 torch？（线性注意力必需）
#   Q4 verl 训练侧依赖（ray/tensordict/codetiming/...）装完后 verl 能否 import
#
# 背景：sft env 是 torch 2.6 + transformers 5.15（适配器 C 层已在其中验过），但
# vllm 0.17 不兼容 torch 2.6；verl 的 hf_rollout 又不传 pixel_values（视觉瞎跑）。
# 所以统一训练 env 只能往 vllm35 这一侧靠。
set +e
source /opt/conda/etc/profile.d/conda.sh
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

if ! conda env list | grep -qE '^vllm35 '; then
  conda create -n vllm35 python=3.11 -y >/dev/null 2>&1
fi
conda activate vllm35
python -c 'import vllm' 2>/dev/null || pip install -q --no-cache-dir 'vllm==0.17.0'

echo "########## Q1 版本矩阵"
python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    print("Qwen3_5ForConditionalGeneration: OK")
except Exception as e:
    print("Qwen3_5ForConditionalGeneration: MISSING", repr(e))
PY

echo "########## Q4 verl 训练侧依赖"
pip install -e rl_train/verl_agent/ --no-deps -q 2>&1 | tail -1
pip install -q accelerate codetiming datasets dill pandas peft pylatexenc 'ray[default]' \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' torchdata wandb 'qwen-vl-utils' pybind11 omegaconf hydra-core 2>&1 | tail -2
python -c "import verl, agent_system, gigpo; from verl import DataProto; print('verl import OK')" 2>&1 | tail -3
python -c "import torch, transformers, vllm; print('复查版本(装依赖后):', torch.__version__, transformers.__version__, vllm.__version__)"

echo "########## Q2 vllm_rollout 模块 import（逐个）"
for m in \
  verl.workers.rollout.vllm_rollout \
  verl.workers.rollout.vllm_rollout.vllm_rollout_spmd \
  verl.workers.rollout.vllm_rollout.vllm_async_server \
  verl.workers.sharding_manager.fsdp_vllm \
  verl.workers.fsdp_workers ; do
  python -c "import importlib; importlib.import_module('$m'); print('OK  $m')" 2>&1 | tail -1
done

echo "########## Q3 线性注意力 / flash-attn 内核"
python -c 'import fla; print("fla already", fla.__version__)' 2>/dev/null || pip install -q flash-linear-attention 2>&1 | tail -1
python -c 'import causal_conv1d' 2>/dev/null || CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -q --no-build-isolation causal-conv1d 2>&1 | tail -2
python -c 'import flash_attn' 2>/dev/null || pip install -q flash-attn --no-build-isolation 2>&1 | tail -2
python - <<'PY'
for m in ("fla", "causal_conv1d", "flash_attn"):
    try:
        mod = __import__(m); print("OK ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("FAIL", m, repr(e)[:160])
PY
echo PROBE_TRAIN_ENV_DONE

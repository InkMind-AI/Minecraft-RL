#!/usr/bin/env bash
# 组装 verl GRPO 训练用的完整 conda env：`vllmtrain`（vllm35 的克隆：torch 2.10 +
# vllm 0.17，见 colocate-p2 探针结论）+ transformers 5.15（Qwen3_5）+ fla 系内核 +
# verl-agent 训练侧依赖 + Minecraft 环境依赖（openagents + minestudio + Malmo）。
#
# 这一个 env 要同时满足两件此前分属不同 env 的事：
#   - 训练/推理模型侧（原来在 sft env / vllm35 各自的一半）——colocate-p2 已验证
#     torch2.10+transformers5.15+vllm0.17 可共存、HF→vLLM 权重同步能走通
#   - 环境侧（原来在 openha env）——因为 verl 的 Ray 训练 driver 会直接 import
#     agent_system.environments.env_package.minecraft，它顶层 import openagents
#     与 minestudio，不能只让 Ray worker 侧有这些依赖
#
# ⚠ openagents 的 pyproject.toml 声明 vllm==0.8.5 / transformers==4.54.0 /
# numpy==1.26.4——这些和本 env 的 vllm 0.17 / transformers 5.15 直接冲突。必须
# `--no-deps` 安装，只要它的代码（env.py/task_manager/action_mapping 等）本身
# 不会在 import 时因为新版本 API 变化而炸，运行时不 `import vllm` 走本地推理
# （评测走 HTTP 调远端 vllm35 服务，训练侧现在是 verl 自己管 vLLM，都不经
# openagents 的本地 vllm 路径）。
set +e
source /opt/conda/etc/profile.d/conda.sh
REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
cd "$REPO_ROOT"

if ! conda env list | grep -qE '^vllm35 '; then
  conda create -n vllm35 python=3.11 -y >/dev/null 2>&1
  conda activate vllm35 && pip install -q --no-cache-dir 'vllm==0.17.0' && conda deactivate
fi
if ! conda env list | grep -qE '^vllmtrain '; then
  echo "[setup] cloning vllm35 -> vllmtrain"
  conda create -n vllmtrain --clone vllm35 -y >/dev/null 2>&1
fi
conda activate vllmtrain

echo "[setup] ① Qwen3.5 支持：transformers 5.15"
python -c "from transformers import Qwen3_5ForConditionalGeneration" 2>/dev/null \
  || pip install -q 'transformers==5.15.0'

echo "[setup] ② 线性注意力内核（fla / causal-conv1d）"
python -c 'import fla' 2>/dev/null || pip install -q flash-linear-attention
# causal-conv1d 也是源码编译（CUDA 扩展），同样可能在新 torch/CUDA 组合下编译
# 异常缓慢——加 10 分钟超时，超时不阻塞整个 setup（fla 的 Triton 纯 Python 回退
# 路径在 fla 内部会自动接管，只是慢一些，不是功能缺失）。
python -c 'import causal_conv1d' 2>/dev/null \
  || timeout 600 env CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 pip install -q --no-build-isolation causal-conv1d \
  || echo "[setup][WARN] causal-conv1d 编译超时/失败，fla 回退纯 Triton 路径（较慢但可用）"
# ⚠ 09-28：不装 flash-attn。torch 2.10+cu128 太新，PyPI 没有匹配的预编译轮子
# （sft env 能用固定版本 2.7.4.post1 是因为那边钉的是 torch 2.6+cu124），pip 会
# 尝试本地源码编译——实测在 koala 上把 setup 任务卡死 20+ 分钟且无输出，疑似
# 编译内存不足。verl-c4 已证明本仓的 attn 组合是 fla（线性注意力层，独立于
# flash-attn）+ 全注意力层走 sdpa 也没问题（colocate-p2 的 V4 就是这样跑的），
# 且这里的序列是单图短 prompt（非训练侧 30 帧长历史），sdpa 没有 O(N^2) 顾虑。
# fsdp_workers.py 的 attn_implementation 已改为读 model.attn_implementation
# （默认 sdpa，见 ppo_trainer.yaml），因此训练侧压根不需要装它。

echo "[setup] ③ verl-agent 训练侧依赖"
pip install -e rl_train/verl_agent/ --no-deps -q
pip install -q accelerate codetiming datasets dill pandas peft pylatexenc 'ray[default]' \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' torchdata wandb qwen-vl-utils pybind11 omegaconf hydra-core gym

echo "[setup] ④ openagents（--no-deps，见文件头注释）+ minestudio"
pip install -e . --no-deps -q
python -c "import minestudio" 2>/dev/null || pip install -q minestudio
if ! command -v java >/dev/null 2>&1; then
    echo "[setup] installing openjdk=8"
    conda install --channel=conda-forge openjdk=8 -y -q
fi
if ! python -c "from cuda import cuda, cudart" >/dev/null 2>&1; then
    pip install -q "cuda-python==12.6.2.post1"
fi
ENGINE_MIRROR_S3_URI="s3://arcwm-code-us-west-2/axiom/assets/minestudio/engine.zip"
_engine_ok() {
    python -c "
import os
from minestudio.utils import get_mine_studio_dir
assert os.path.exists(os.path.join(get_mine_studio_dir(), 'engine', 'build', 'libs', 'mcprec-6.13.jar'))
" >/dev/null 2>&1
}
if ! _engine_ok; then
    MS_DIR="${MINESTUDIO_DIR:-$(python -c 'from minestudio.utils import get_mine_studio_dir; print(get_mine_studio_dir())')}"
    mkdir -p "$MS_DIR"
    if aws s3 cp "$ENGINE_MIRROR_S3_URI" "$MS_DIR/engine.zip" --only-show-errors; then
        python -c "
import os, zipfile
d = '$MS_DIR'
with zipfile.ZipFile(os.path.join(d, 'engine.zip')) as z: z.extractall(d)
os.remove(os.path.join(d, 'engine.zip'))
"
    else
        for attempt in 1 2 3 4 5; do
            python -c "from minestudio.simulator.entry import download_engine; download_engine()" && break
            sleep 20
        done
    fi
fi
unset -f _engine_ok

echo "[setup] ⑤ 逐个验证 import"
python - <<'PY'
mods = ["torch", "transformers", "vllm", "verl", "agent_system", "gigpo",
        "ray", "omegaconf", "gym", "minestudio", "openagents", "cv2", "numpy"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"OK   {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m:16s} {e!r}"[:200])
from transformers import Qwen3_5ForConditionalGeneration
print("OK   Qwen3_5ForConditionalGeneration")
PY

echo "[setup] vllmtrain env ready: java=$(command -v java || echo MISSING)"
echo SETUP_VLLMTRAIN_DONE

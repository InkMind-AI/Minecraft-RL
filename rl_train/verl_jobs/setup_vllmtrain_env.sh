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
# ⚠ 09-28 核心教训（第一版实测踩过）：openagents 声明 vllm==0.8.5/
# transformers==4.54.0/numpy==1.26.4，minestudio 也有自己一整套依赖——**任何一个
# 不带 --no-deps 的安装都可能静默把 torch/torchvision/numpy 换掉**，冲垮
# colocate-p2 刚验证过的组合。第一次跑：装完 verl 训练依赖 + `pip install
# minestudio`（未加 --no-deps）后，torch 被从 2.10.0 换成 2.8.0，还带了个 ABI
# 不匹配的 torchaudio（`undefined symbol: aoti_torch_create_device_guard`），
# import 直接崩。**因此这个脚本里所有第三方安装一律 --no-deps**，缺什么用
# `_install_missing_no_deps` 逐个补最小依赖，绝不让 pip resolver 联带升降级
# torch 系；每个大步骤后都打印 torch 版本，一旦漂移立即 FATAL 退出，不带着错误
# 环境继续跑后面几十分钟。
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

_torch_ver() { python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "?"; }
_assert_torch_unchanged() {
    local now; now="$(_torch_ver)"
    echo "[setup] torch（$1）: $now"
    if [ "$now" != "$_TORCH_BASELINE" ]; then
        echo "[setup][FATAL] torch 版本漂移: $_TORCH_BASELINE -> $now（发生于「$1」之后），" \
             "某个安装仍触发了 torch 重装，需要人工排查是哪个包" >&2
        exit 1
    fi
}
# 逐个补最小依赖：--no-deps 装的包 import 报 ModuleNotFoundError 时，只装那个
# 缺失模块本身（同样 --no-deps），绝不让 resolver 联带升降级 torch 系。最多试
# 8 轮（每轮最多补一个新缺失模块，链式缺失逐层剥开）。
_install_missing_no_deps() {
    local probe="$1"
    for _ in 1 2 3 4 5 6 7 8; do
        local err; err="$(python -c "$probe" 2>&1)"
        if [ -z "$err" ]; then return 0; fi
        local mod
        mod="$(echo "$err" | grep -oE "No module named '[^']+'" | head -1 | sed "s/No module named '//;s/'//" | cut -d. -f1)"
        if [ -z "$mod" ]; then echo "[setup][WARN] 非缺模块错误，放弃重试: $err"; return 1; fi
        echo "[setup] 补装 $mod（--no-deps）"
        pip install -q --no-deps "$mod" 2>&1 | tail -2
    done
    echo "[setup][WARN] 多次重试后仍失败: $probe"
    return 1
}

_TORCH_BASELINE="$(_torch_ver)"
echo "[setup] torch（克隆基线）: $_TORCH_BASELINE"

echo "[setup] ① Qwen3.5 支持：transformers 5.15（--no-deps：tokenizers/safetensors 等
子依赖复用 vllm35 自带的 transformers 4.57.6 已装好的版本）"
python -c "from transformers import Qwen3_5ForConditionalGeneration" 2>/dev/null \
  || pip install -q --no-deps 'transformers==5.15.0'
_install_missing_no_deps "from transformers import Qwen3_5ForConditionalGeneration"
_assert_torch_unchanged "①transformers"

echo "[setup] ② 线性注意力内核（fla / causal-conv1d）"
python -c 'import fla' 2>/dev/null || pip install -q --no-deps flash-linear-attention
_install_missing_no_deps "import fla"
# causal-conv1d 是源码编译的 CUDA 扩展，可能在新 torch/CUDA 组合下编译异常缓慢
# ——加 10 分钟超时，超时不阻塞整个 setup（fla 内部会自动回退纯 Triton 路径，
# 只是慢一些，不是功能缺失）。
python -c 'import causal_conv1d' 2>/dev/null \
  || timeout 600 env CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 pip install -q --no-build-isolation --no-deps causal-conv1d \
  || echo "[setup][WARN] causal-conv1d 编译超时/失败，fla 回退纯 Triton 路径（较慢但可用）"
# ⚠ 不装 flash-attn：torch 2.10+cu128 太新，PyPI 无匹配预编译轮子（sft env 能用
# 固定版本 2.7.4.post1 是因为那边钉的是 torch 2.6+cu124），源码编译实测把上一版
# setup 任务卡死 20+ 分钟无输出。colocate-p2 的 V4 已证明 fla（线性注意力层）+
# sdpa（全注意力层）在真实 9B 权重上没问题，这里的 rollout prompt 又是单图短
# 序列（非训练侧 30 帧长历史），sdpa 没有 O(N^2) 顾虑。fsdp_workers.py 的
# attn_implementation 已改为读 model.attn_implementation（默认 sdpa），训练侧
# 压根不需要装它。
_assert_torch_unchanged "②fla/causal-conv1d"

echo "[setup] ③ verl-agent 训练侧依赖（--no-deps）"
pip install -e rl_train/verl_agent/ --no-deps -q
pip install -q --no-deps accelerate codetiming datasets dill pandas peft pylatexenc 'ray[default]' \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' torchdata wandb qwen-vl-utils pybind11 omegaconf hydra-core gym
_install_missing_no_deps "import verl, agent_system, gigpo"
_assert_torch_unchanged "③verl训练依赖"

echo "[setup] ④ openagents（--no-deps）+ minestudio（--no-deps + 逐个补依赖）"
pip install -e . --no-deps -q
pip install -q --no-deps minestudio
_install_missing_no_deps "import minestudio"
_assert_torch_unchanged "④minestudio"

if ! command -v java >/dev/null 2>&1; then
    echo "[setup] installing openjdk=8"
    conda install --channel=conda-forge openjdk=8 -y -q
fi
if ! python -c "from cuda import cuda, cudart" >/dev/null 2>&1; then
    pip install -q --no-deps "cuda-python==12.6.2.post1"
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
ok = True
for m in mods:
    try:
        mod = __import__(m)
        print(f"OK   {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"FAIL {m:16s} {e!r}"[:200])
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    print("OK   Qwen3_5ForConditionalGeneration")
except Exception as e:
    ok = False
    print(f"FAIL Qwen3_5ForConditionalGeneration {e!r}"[:200])
import sys
sys.exit(0 if ok else 1)
PY
IMPORT_CHECK_RC=$?
_assert_torch_unchanged "⑤最终验证"

echo "[setup] vllmtrain env ready: java=$(command -v java || echo MISSING)"
if [ "$IMPORT_CHECK_RC" -ne 0 ]; then
    echo "[setup][FATAL] 上面的 import 验证有 FAIL，环境未就绪" >&2
    exit 1
fi
echo SETUP_VLLMTRAIN_DONE

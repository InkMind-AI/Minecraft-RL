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

if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[setup] installing xvfb (needed later for the real Malmo boot check)"
    apt-get install -y -qq xvfb >/dev/null 2>&1
fi

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
# 逐个补最小依赖，同样全程 --no-deps（绝不让 resolver 联带升降级 torch 系）。
# 用退出码判定成功/失败，而不是"有无 stderr 输出"——import 时的
# DeprecationWarning/FutureWarning 会打到 stderr 但退出码是 0，之前用
# `[ -z "$err" ]` 误判过一次（transformers 5.15 的一条无害警告被当成失败）。
# 覆盖两种缺陷形态：
#   ① ModuleNotFoundError: No module named 'X'      -> 装 X
#   ② ImportError: cannot import name '...' from 'X' -> **升级** X（--no-deps
#      装的新版包往往需要更新版本的间接依赖才有新符号，如 transformers 5.15
#      需要更新的 huggingface_hub 才有 is_offline_mode；普通 install 不生效，
#      必须 --upgrade）
#   ③ 版本门禁 "X>=A,<B is required ... but found X==C" -> 钉版本区间
# 最多试 20 轮，每轮最多处理一个新缺陷，链式缺失逐层剥开。minerl/minestudio 的
# 依赖树比预想深得多（09-28 实测：absl→gymnasium→gym3→imageio→coloredlogs→
# humanfriendly→daemoniker→lxml 连续 8 层，8 轮上限恰好用尽还没到底），
# 20 轮留足够余量。
#
# ⚠ 导入名 ≠ PyPI 包名的情况（`pip install $mod` 会装错包或直接 404）：
# 09-28 实测 dateutil（PyPI 是 python-dateutil）连续 8 轮无效才发现。这里只维护
# 一张已知小表，遇到新的照此模式继续加。
declare -A _PIP_NAME_MAP=(
    [dateutil]=python-dateutil [yaml]=PyYAML [PIL]=Pillow [cv2]=opencv-python
    [sklearn]=scikit-learn [absl]=absl-py [google]=protobuf [jwt]=PyJWT
    [Crypto]=pycryptodome [OpenSSL]=pyOpenSSL [dotenv]=python-dotenv
    # ⚠ 不能装最新版：omegaconf 的语法文件是用 ANTLR 4.9.x 生成的（ATN 序列化
    # 版本 3），装 antlr4-python3-runtime 最新版（4.13.x，ATN v4）会在真正解析
    # 插值语法时报 "Could not deserialize ATN with version 3 (expected 4)"——
    # 09-28 trainenv7 实测踩到。这是 OmegaConf/Hydra 生态里的经典版本坑，必须
    # 精确钉 4.9.*，与 omegaconf 官方 requirements 一致。
    [antlr4]="antlr4-python3-runtime==4.9.*"
)
_install_missing_no_deps() {
    local probe="$1"
    local runner="${2:-python}"  # 默认直接跑；传 "xvfb-run -a python" 给需要显示的探针
    for _ in $(seq 1 20); do
        local out rc
        out="$($runner -c "$probe" 2>&1)"; rc=$?
        if [ "$rc" -eq 0 ]; then return 0; fi
        local mod
        mod="$(echo "$out" | grep -oE "No module named '[^']+'" | head -1 | sed "s/No module named '//;s/'//" | cut -d. -f1)"
        if [ -n "$mod" ]; then
            local pkg="${_PIP_NAME_MAP[$mod]:-$mod}"
            echo "[setup] 补装 $mod（pip 包名 $pkg，--no-deps）"
            pip install -q --no-deps "$pkg" 2>&1 | tail -2
            continue
        fi
        mod="$(echo "$out" | grep -oE "cannot import name '[^']+' from '[^']+'" | head -1 | sed -E "s/.*from '([^']+)'/\1/" | cut -d. -f1)"
        if [ -n "$mod" ]; then
            echo "[setup] 升级 $mod（--no-deps --upgrade，缺符号）"
            pip install -q --no-deps --upgrade "$mod" 2>&1 | tail -2
            continue
        fi
        # transformers 内部版本门禁："X>=A,<B is required ... but found X==C"——
        # 上面盲目 --upgrade 到最新版会撞这个（huggingface_hub 2.0.0 比 5.15.0
        # 要求的 <2.0 还新）。直接把报错里的版本约束原样递给 pip，钉到兼容区间。
        local spec
        spec="$(echo "$out" | grep -oE "[A-Za-z0-9_.-]+[<>=,.0-9]+ is required" | head -1 | sed 's/ is required//')"
        if [ -n "$spec" ]; then
            echo "[setup] 按版本门禁钉版本: $spec（--no-deps）"
            pip install -q --no-deps "$spec" 2>&1 | tail -2
            continue
        fi
        echo "[setup][WARN] 无法识别的错误，放弃重试: $out"
        return 1
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
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' torchdata wandb qwen-vl-utils pybind11 omegaconf hydra-core gym \
  pyarrow multiprocess einops
_install_missing_no_deps "import verl, agent_system, gigpo"
# dp_actor.py 顶层无条件 `from flash_attn.bert_padding import ...`（与
# attn_implementation 无关，纯 pad/unpad 工具函数）——已在源码里加了不装
# flash-attn 时的纯 PyTorch 兜底（用 einops.rearrange），这里探一下确认能 import。
_install_missing_no_deps "import verl.workers.actor.dp_actor"
# ⚠ 09-28：`import verl` / `agent_system` 本身很浅——真正的训练入口
# `verl.trainer.main_ppo` 会往下拉一条完全不同的深链（ray_trainer.py ->
# multi_turn_rollout -> verl.utils.dataset.rl_dataset -> `import datasets` ->
# HF datasets 内部用 `multiprocess`），首次 GRPO 冒烟才在这里连环炸出
# pyarrow（pandas.to_parquet 引擎）和 multiprocess 两个缺口。不会 `main()`
# 副作用（`if __name__ == "__main__"` 守卫），可以放心 import 探测。
_install_missing_no_deps "import verl.trainer.main_ppo"
# omegaconf/hydra-core 用 ANTLR4 解析插值语法（`from antlr4 import ...`），但只在
# 真正解析带 `${...}` 插值的配置时才触发 import，`import omegaconf` 本身不会
# 暴露这个缺口——09-28 实测：step③ 探针全绿，直到 step⑤ 最终验证才 FAIL。
# 显式多探一次，提前暴露。
_install_missing_no_deps "from omegaconf import OmegaConf; OmegaConf.create({'a': '\${b}', 'b': 1})['a']"
_assert_torch_unchanged "③verl训练依赖"

echo "[setup] ④ openagents（--no-deps）+ minestudio（--no-deps + 逐个补依赖）"
pip install -e . --no-deps -q
pip install -q --no-deps minestudio
# Pyro4：minerl 与 Malmo Java 进程通信用的 RPC 库，只在真正 launch 一个 Malmo
# 实例时才被 import（MinecraftSim() 实例化路径的更深处），不在任何 import 链
# 探针能覆盖到的静态导入范围内——09-28 trainenv11 实测：D 层过、E 层（真正起
# Malmo）才炸 `No module named 'Pyro4'`。minerl 生态惯用 4.76（更新版本对 RPC
# 协议/serializer 有破坏性改动，未验证是否兼容，不冒险装最新版）。
python -c "import Pyro4" 2>/dev/null || pip install -q --no-deps "Pyro4==4.76" "serpent>=1.41"
# ⚠ 探针必须探到真正会被调用的深层路径，不能只 `import minestudio`：
# minestudio/__init__.py 本身很浅，不会触发它自己的子模块树（utils.register
# 需要 absl；utils.vpt_lib.actions 经 action_head 需要 gymnasium，这两条都是
# 09-28 分别实测踩到的独立缺口——`import minestudio` 和 `get_mine_studio_dir`
# 都不经过 vpt_lib）。与其继续追加零散子路径，直接用 smoke_step2.py 自己会跑
# 的**完整导入链**做探针：openagents.agents.utils.action_mapping（经
# minestudio.utils.vpt_lib）+ minecraft env 包本身，这样任何这条链上的新缺口
# 都会在 setup 阶段暴露，而不是留到 Malmo 冒烟才炸。
_install_missing_no_deps "from agent_system.environments.env_package.minecraft import build_minecraft_envs, minecraft_projection"
_install_missing_no_deps "from minestudio.utils import get_mine_studio_dir"
_assert_torch_unchanged "④minestudio"

# ⚠ 09-29：这个坑连续咬了三次，根源是"openjdk 的 conda 钩子普遍不兼容
# set -u"，不是某一次操作特有的：
#   smoke7 -- 装的瞬间：deactivate.d/openjdk_deactivate.sh 读
#             $JAVA_HOME_CONDA_BACKUP，装完立即 unbound variable 退出
#   smoke8 -- 装完之后、job 脚本里多余的第二次 `conda activate vllmtrain`：
#             activate.d/openjdk_activate.sh 读 $target_platform，同样炸
# 只在 install 这一行局部关 set -u（此前的修法）治不了后者，因为炸点在
# **本脚本 source 结束之后、调用方自己的代码里**——sourced 脚本改的 shell
# 选项会保留到调用方后续执行。因此在检查/安装 openjdk 之前就关掉 set -u，
# 且从这里开始**不再恢复**，兜住"java 已存在从而跳过安装分支"和"调用方
# 之后自己再 activate/deactivate 一次"两种情形。其余逻辑一律用
# `${VAR:-default}` 写法保证在 nounset 关闭下也不出错（本脚本从头至尾的风格）。
set +u
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

# ⚠ 09-28：Pyro4/xmltodict 这类依赖只在**真正启动一个 Malmo/JVM 实例**时才会
# 被 import（minerl 的 Java 通信桥接层），任何静态 import 链探针都探不到——
# trainenv10→12 连续三轮各自在这里踩出一个新缺口（先 Pyro4 后 xmltodict），
# 每轮都要等一次完整 15+ 分钟的 job 往返才能看到下一个。与其继续被动挨个撞，
# 这里直接真实起一次 Malmo（xvfb-run 提供显示；引擎/JVM 此刻均已就绪），让补丁
# 循环在 setup 阶段内部把这条运行时依赖链一次性走完。跑一次基本的 mine_block
# 任务 reset+close（不 step，避免额外拉长冒烟时间），失败信息交给
# _install_missing_no_deps 复用同一套仓库判定+修复逻辑。
echo "[setup] ⑤ 真实起一次 Malmo（捕获仅运行时触发的依赖，如 Pyro4/xmltodict）"
_install_missing_no_deps "
from openagents.envs.tasks.task_manager import choose_available_task
from agent_system.environments.env_package.minecraft.envs import _build_sim
cfg = choose_available_task('mine_block:oak_log', difficulty='easy')
sim = _build_sim(cfg, record_path=None)
sim.reset()
sim.close()
print('MALMO_BOOT_OK')
" "xvfb-run -a python"
_assert_torch_unchanged "⑤真实Malmo冒烟"

echo "[setup] ⑥ 逐个验证 import"
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
_assert_torch_unchanged "⑥最终验证"

echo "[setup] vllmtrain env ready: java=$(command -v java || echo MISSING)"
if [ "$IMPORT_CHECK_RC" -ne 0 ]; then
    echo "[setup][FATAL] 上面的 import 验证有 FAIL，环境未就绪" >&2
    exit 1
fi
echo SETUP_VLLMTRAIN_DONE

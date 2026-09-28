#!/usr/bin/env bash
# 准备 `openha` conda env（Malmo/MineStudio rollout 所需），供 verl 迁移的环境侧
# 任务复用。逻辑与 examples/eval_backbones/run_backbone_eval.sh 的 setup 段一致，
# 单独抽出来的原因：
#
#   1. koala 镜像**不预装** openha env，它是评测脚本运行时创建的。verl 的环境冒烟
#      不走 run_backbone_eval.sh，直接 `conda activate openha` 会静默失败（set -e
#      下 `|| true` 又把它吞掉），脚本于是跑在没有 torch/minestudio 的 base env 里
#      ——09-27 verl-env-d1 实测即如此（ModuleNotFoundError: No module named 'torch'）。
#   2. 每个 verl 任务里重抄一遍这段安装逻辑，迟早与评测侧漂移。
#
# 用法： source rl_train/verl_jobs/setup_openha_env.sh   # 之后即处于 openha env
#
# ⚠ 09-28：本文件是被 source 的，所以**不能**开 `set -u`。conda 的 activate/deactivate
# 钩子不兼容 nounset——openjdk 包自带的 deactivate.d/openjdk_deactivate.sh 第 3 行
# 直接读 $JAVA_HOME_CONDA_BACKUP，未定义时在 -u 下立即报错退出（verl-env-d2 实测：
# 装完 openjdk 后脚本死在 "JAVA_HOME_CONDA_BACKUP: unbound variable"）。
# run_backbone_eval.sh 从没开 -u，所以评测侧一直没踩到。
# 这里的做法：进入时记下调用方是否开了 nounset，整段关掉，结束时按原样恢复。
case "$-" in *u*) _OPENHA_RESTORE_NOUNSET=1 ;; *) _OPENHA_RESTORE_NOUNSET=0 ;; esac
set +u
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/data/work/run_codes/Minecraft-CoT}"
source /opt/conda/etc/profile.d/conda.sh
cd "$REPO_ROOT"

if ! conda env list | grep -qE "^openha "; then
    echo "[setup] creating conda env: openha"
    conda create -n openha python=3.10 -y
fi
conda activate openha

# openjdk 必须每次独立检查：koala 节点跨 job 复用持久化 env，若该 env 由某次
# openjdk 安装失败的历史 job 创建，下面的 python 包检查会判定"已装"而跳过，
# openjdk 就永远补不上，rollout 会炸 `java: not found`（见评测脚本同处注释）。
if ! command -v java >/dev/null 2>&1; then
    echo "[setup] installing openjdk=8 (MineStudio/Malmo launcher)"
    conda install --channel=conda-forge openjdk=8 -y -q
fi

# 用真实第三方依赖判断，不能用 `import openagents`——cwd 下就有同名源码目录，
# 未安装也能 import 成功，会误判。
if ! python -c "import torch, vllm, minestudio, ray" >/dev/null 2>&1; then
    echo "[setup] installing openagents + deps into openha env"
    for attempt in 1 2 3 4 5; do
        pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124 \
            && pip install -q -e . \
            && break
        echo "[retry] deps install failed (attempt ${attempt}/5), retry in 20s" >&2
        sleep 20
    done
    if ! python -c "import torch, vllm, minestudio, ray" >/dev/null 2>&1; then
        echo "[setup][FATAL] torch/vllm/minestudio/ray still missing after retries" >&2
        exit 1
    fi
fi

# openagents/agents/base.py 顶层无条件 import sam2，即使 text_action 用不到。
if ! python -c "from sam2.build_sam import build_sam2_camera_predictor" >/dev/null 2>&1; then
    echo "[setup] installing sam2 (zhwang4ai/SAM2 fork, for base.py import compat)"
    for attempt in 1 2 3 4 5; do
        pip install -q "git+https://github.com/zhwang4ai/SAM2.git" && break
        echo "[retry] sam2 install failed (attempt ${attempt}/5), retry in 15s" >&2
        sleep 15
    done
fi

echo "[setup] openha env ready: python=$(python -V 2>&1), java=$(command -v java || echo MISSING)"

if [ "$_OPENHA_RESTORE_NOUNSET" = "1" ]; then set -u; fi
unset _OPENHA_RESTORE_NOUNSET

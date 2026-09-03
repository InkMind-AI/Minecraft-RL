#!/bin/bash
# ============================================================================
# 无vLLM版"骨干VLM"评测 —— 与 run_backbone_eval.sh 完全相同的任务集/设置/汇总，
# 但不起 vLLM 服务：rollout 进程用 transformers 直接加载模型推理
# (rollout_openha.py --vlm_client_mode hf -> VLMClient._generate_local_hf)。
#
# 用途：hf 直载 vs vLLM 服务的"效果 + 效率"对照实验。
#
# 与 run_backbone_eval.sh 的差异（且仅此而已）：
#   1. 跳过整个 vLLM serve 段（无端口/健康检查/VLLM_CONDA_ENV）
#   2. openha 环境 pip install -e . 之后强制 transformers==5.15.0（与训练侧
#      完全一致，Qwen3.5 架构所需；vlm_client.py 已改为惰性 vllm 导入，升级
#      transformers 不会破坏 openha 环境里 vllm 的存在性）
#   3. GPU_PER_ROLLOUT 默认 0.3（每个 rollout 进程独占加载一份完整 fp16 模型
#      ~19GB + 激活，3 个并发 ≈ 70GB，适配单张 H200 141GB；vLLM 模式是 0.1，
#      因为 10 个模拟器共享同一个服务进程）
#   4. rollout 传 --vlm_client_mode hf
#
# 用法（由 launch_<model>_hf*.sh 设置好环境变量后 source 本脚本）：
#   MODEL_LOCAL_NAME / MODEL_S3_URI / SERVED_MODEL_NAME（仅用于目录命名）
# ============================================================================
set -o pipefail

: "${MODEL_LOCAL_NAME:?must set MODEL_LOCAL_NAME}"
: "${MODEL_S3_URI:?must set MODEL_S3_URI}"
: "${SERVED_MODEL_NAME:?must set SERVED_MODEL_NAME}"

# ---- 固定评测设置（与 run_backbone_eval.sh 逐字节一致的部分） -----------------
export EVAL_BENCHMARK=${EVAL_BENCHMARK:-mini}
export TASK_LIST=${TASK_LIST:-"mine_block:oak_log kill_entity:sheep craft_item:crafting_table"}
export TASK_DIFFICULTY_LIST=${TASK_DIFFICULTY_LIST:-}
case "${EVAL_BENCHMARK}" in
    mini) export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-10} ;;
    full) export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-3} ;;
    *)    export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-5} ;;
esac
export MAX_STEPS_NUM=${MAX_STEPS_NUM:-200}
export MAXIMUM_HISTORY_LENGTH=${MAXIMUM_HISTORY_LENGTH:-3}
export DIFFICULTY=${DIFFICULTY:-zero}
export OUTPUT_MODE=${OUTPUT_MODE:-text_action}
export SYSTEM_MESSAGE_TAG=${SYSTEM_MESSAGE_TAG:-text_action}
export TEMPERATURE=${TEMPERATURE:-0.8}
export TOP_P=${TOP_P:-0.99}
export TOP_K=${TOP_K:--1}
export FPS=${FPS:-20}
export LIMIT_MM_IMAGE=${LIMIT_MM_IMAGE:-5}   # hf 模式不用，保持变量一致性
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}   # hf 模式不用
export TP_SIZE=${TP_SIZE:-1}                # hf 模式不用
# 每个rollout进程独占加载一份完整模型（区别于vLLM模式共享服务的0.1）
export GPU_PER_ROLLOUT=${GPU_PER_ROLLOUT:-0.3}
export VLLM_PORT=${VLLM_PORT:-11000}        # hf 模式不用，占位保持一致
# 与 vLLM 模式相同的 chat_template_kwargs 透传（Qwen3.x enable_thinking=false，
# vlent _generate_local_hf 已支持从 extra_body 读取）
export EXTRA_BODY_JSON=${EXTRA_BODY_JSON:-'{"chat_template_kwargs": {"enable_thinking": false}}'}

REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
LOCAL_MODEL_DIR="/local-ssd/models/${MODEL_LOCAL_NAME}"
RECORD_ROOT="/local-ssd/eval_output/${MODEL_LOCAL_NAME}"
RESULT_S3_URI="s3://arcwm-code-us-west-2/axiom/eval_results/${MODEL_LOCAL_NAME}"
LOG_DIR="/local-ssd/logs/${MODEL_LOCAL_NAME}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-/local-ssd/minestudio}"
mkdir -p "${LOCAL_MODEL_DIR}" "${RECORD_ROOT}" "${LOG_DIR}" "${MINESTUDIO_DIR}"

if [ "${EVAL_BENCHMARK}" != "smoke" ] && [ -z "${TASK_DIFFICULTY_LIST}" ]; then
    echo "[task-list] generating EVAL_BENCHMARK=${EVAL_BENCHMARK} task list via build_task_list.py"
    export TASK_LIST_MANIFEST="${RECORD_ROOT}/task_list_manifest.json"
    TASK_DIFFICULTY_LIST="$(python3 "$(dirname "${BASH_SOURCE[0]}")/build_task_list.py" --scope "${EVAL_BENCHMARK}")"
    export TASK_DIFFICULTY_LIST
    echo "[task-list] $(echo "${TASK_DIFFICULTY_LIST}" | wc -w) tasks, manifest -> ${TASK_LIST_MANIFEST}"
fi

echo "=============================================================="
echo "[eval-hf] model=${MODEL_LOCAL_NAME} (transformers in-process, NO vLLM)"
echo "[eval-hf] benchmark=${EVAL_BENCHMARK} rollouts_per_task=${ROLLOUTS_PER_TASK} max_steps=${MAX_STEPS_NUM} gpu_per_rollout=${GPU_PER_ROLLOUT}"
if [ -n "${TASK_DIFFICULTY_LIST}" ]; then
    echo "[eval-hf] tasks(task@difficulty, $(echo "${TASK_DIFFICULTY_LIST}" | wc -w) total)"
else
    echo "[eval-hf] tasks=${TASK_LIST} difficulty=${DIFFICULTY}"
fi
echo "=============================================================="

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[setup] installing xvfb (system package, needed by MineStudio headless rendering)"
    apt-get update -qq && apt-get install -y -qq xvfb
fi

# ---------------------------------------------------------------------------
# 1. openha 评测环境（与 run_backbone_eval.sh 相同）
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
if ! conda env list | grep -qE "^openha "; then
    echo "[setup] creating conda env: openha"
    conda create -n openha python=3.10 -y
fi
conda activate openha
if ! command -v java >/dev/null 2>&1; then
    echo "[setup] installing openjdk=8 (required by MineStudio/Malmo simulator launcher)"
    conda install --channel=conda-forge openjdk=8 -y -q
fi
if ! python -c "import torch, minestudio, ray" >/dev/null 2>&1; then
    echo "[setup] installing openagents + deps into openha env"
    for attempt in 1 2 3 4 5; do
        pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
            && pip install -q -e . \
            && break
        echo "[retry] openagents/torch install failed (attempt ${attempt}/5), retrying in 20s..." >&2
        sleep 20
    done
    if ! python -c "import torch, minestudio, ray" >/dev/null 2>&1; then
        echo "[setup][FATAL] torch/minestudio/ray install/import still failing after retries, aborting" >&2
        exit 1
    fi
fi
if ! python -c "from sam2.build_sam import build_sam2_camera_predictor" >/dev/null 2>&1; then
    echo "[setup] installing sam2 (zhwang4ai/SAM2 fork, for base.py import compatibility)"
    for attempt in 1 2 3 4 5; do
        pip install -q "git+https://github.com/zhwang4ai/SAM2.git" && break
        echo "[retry] sam2 pip install failed (attempt ${attempt}/5), retrying in 15s..." >&2
        sleep 15
    done
    if ! python -c "from sam2.build_sam import build_sam2_camera_predictor" >/dev/null 2>&1; then
        echo "[setup][FATAL] sam2 install/import still failing after retries, aborting" >&2
        exit 1
    fi
fi
if ! python -c "from cuda import cuda, cudart" >/dev/null 2>&1; then
    echo "[setup] pinning cuda-python==12.6.2.post1 for minestudio/minerl gpu_utils compatibility"
    pip install -q "cuda-python==12.6.2.post1"
fi

# ---------------------------------------------------------------------------
# 1.5 【hf模式特有】强制 transformers==5.15.0（与训练侧一致；Qwen3.5 架构需要）
#     vlm_client.py 已惰性化 vllm 导入，因此这里升级 transformers 不会因为
#     openha 环境里旧 vllm 的存在而崩溃（本任务也不使用 vllm）。
# ---------------------------------------------------------------------------
if ! python -c "import transformers; assert transformers.__version__ == '5.15.0'" >/dev/null 2>&1; then
    echo "[setup][hf] installing transformers==5.15.0 (training-side version, Qwen3.5 support)"
    for attempt in 1 2 3; do
        pip install -q "transformers==5.15.0" && break
        echo "[retry] transformers install failed (attempt ${attempt}/3), retrying in 15s..." >&2
        sleep 15
    done
    if ! python -c "import transformers; assert transformers.__version__ == '5.15.0'" >/dev/null 2>&1; then
        echo "[setup][FATAL] transformers==5.15.0 install failed, aborting" >&2
        exit 1
    fi
fi
echo "[setup][hf] transformers=$(python -c 'import transformers; print(transformers.__version__)')"

# MineStudio 模拟器引擎（与 run_backbone_eval.sh 相同的 S3 镜像逻辑）
ENGINE_MIRROR_S3_URI="s3://arcwm-code-us-west-2/axiom/assets/minestudio/engine.zip"
if ! python -c "
import os
from minestudio.utils import get_mine_studio_dir
jar = os.path.join(get_mine_studio_dir(), 'engine', 'build', 'libs', 'mcprec-6.13.jar')
assert os.path.exists(jar)
" >/dev/null 2>&1; then
    MINESTUDIO_DIR_RESOLVED="${MINESTUDIO_DIR:-$(python -c 'from minestudio.utils import get_mine_studio_dir; print(get_mine_studio_dir())')}"
    mkdir -p "${MINESTUDIO_DIR_RESOLVED}"
    if aws s3 cp "${ENGINE_MIRROR_S3_URI}" "${MINESTUDIO_DIR_RESOLVED}/engine.zip" --only-show-errors; then
        echo "[setup] downloaded MineStudio simulator engine from S3 mirror, extracting..."
        python -c "
import os, zipfile
d = '${MINESTUDIO_DIR_RESOLVED}'
with zipfile.ZipFile(os.path.join(d, 'engine.zip'), 'r') as z:
    z.extractall(d)
os.remove(os.path.join(d, 'engine.zip'))
"
    else
        echo "[setup] S3 mirror unavailable, falling back to HuggingFace (with retry)"
        for attempt in 1 2 3 4 5; do
            python -c "from minestudio.simulator.entry import download_engine; download_engine()" && break
            echo "[retry] download_engine() failed (attempt ${attempt}/5)" >&2
            sleep 20
        done
    fi
    if ! python -c "
import os
from minestudio.utils import get_mine_studio_dir
jar = os.path.join(get_mine_studio_dir(), 'engine', 'build', 'libs', 'mcprec-6.13.jar')
assert os.path.exists(jar)
" >/dev/null 2>&1; then
        echo "[setup][FATAL] simulator engine missing, aborting" >&2
        exit 1
    fi
fi
echo "[setup] openha env ready (hf mode): $(python -c 'import transformers; print(f"transformers={transformers.__version__}")')"

# ---------------------------------------------------------------------------
# 2. 下载模型权重到本地盘（transformers 5.15 原生读取新 schema，无需 vLLM 的
#    config/preprocessor 兼容性改造；仅排除优化器状态目录）
# ---------------------------------------------------------------------------
if [ ! -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    echo "[download] syncing ${MODEL_S3_URI} -> ${LOCAL_MODEL_DIR}"
    aws s3 sync "${MODEL_S3_URI%/}/" "${LOCAL_MODEL_DIR}/" --no-progress \
        --exclude 'checkpoint-*/*' --exclude 'global_step*/*'
else
    echo "[download] model already present at ${LOCAL_MODEL_DIR}, skip"
fi

# ---------------------------------------------------------------------------
# 3. 逐任务跑 rollout —— 无 vLLM 服务，rollout 进程内 transformers 直载推理
# ---------------------------------------------------------------------------
EVAL_START_TS=$(date +%s)
conda activate openha
cd "${REPO_ROOT}"
if [ -n "${TASK_DIFFICULTY_LIST}" ]; then
    for TASK_DIFF in ${TASK_DIFFICULTY_LIST}; do
        TASK="${TASK_DIFF%@*}"
        TASK_DIFFICULTY="${TASK_DIFF##*@}"
        echo "------------------------------------------------------------"
        echo "[rollout-hf] task=${TASK} difficulty=${TASK_DIFFICULTY} num_rollouts=${ROLLOUTS_PER_TASK}"
        echo "------------------------------------------------------------"
        python examples/rollout_openha.py \
            --output_mode "${OUTPUT_MODE}" \
            --vlm_client_mode hf \
            --system_message_tag "${SYSTEM_MESSAGE_TAG}" \
            --model_id "${SERVED_MODEL_NAME}" \
            --model_path "${LOCAL_MODEL_DIR}" \
            --record_path "${RECORD_ROOT}" \
            --max_steps_num "${MAX_STEPS_NUM}" \
            --maximum_history_length "${MAXIMUM_HISTORY_LENGTH}" \
            --task "${TASK}" \
            --difficulty "${TASK_DIFFICULTY}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --top_k "${TOP_K}" \
            --fps "${FPS}" \
            --gpu_per_rollout "${GPU_PER_ROLLOUT}" \
            --num_rollouts "${ROLLOUTS_PER_TASK}" \
            --extra_body "${EXTRA_BODY_JSON}" \
            2>&1 | tee -a "${LOG_DIR}/rollout_${TASK//[:,]/_}.log"
    done
else
    for TASK in ${TASK_LIST}; do
        echo "------------------------------------------------------------"
        echo "[rollout-hf] task=${TASK} num_rollouts=${ROLLOUTS_PER_TASK}"
        echo "------------------------------------------------------------"
        python examples/rollout_openha.py \
            --output_mode "${OUTPUT_MODE}" \
            --vlm_client_mode hf \
            --system_message_tag "${SYSTEM_MESSAGE_TAG}" \
            --model_id "${SERVED_MODEL_NAME}" \
            --model_path "${LOCAL_MODEL_DIR}" \
            --record_path "${RECORD_ROOT}" \
            --max_steps_num "${MAX_STEPS_NUM}" \
            --maximum_history_length "${MAXIMUM_HISTORY_LENGTH}" \
            --task "${TASK}" \
            --difficulty "${DIFFICULTY}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --top_k "${TOP_K}" \
            --fps "${FPS}" \
            --gpu_per_rollout "${GPU_PER_ROLLOUT}" \
            --num_rollouts "${ROLLOUTS_PER_TASK}" \
            --extra_body "${EXTRA_BODY_JSON}" \
            2>&1 | tee -a "${LOG_DIR}/rollout_${TASK//[:,]/_}.log"
    done
fi
EVAL_ELAPSED=$(( $(date +%s) - EVAL_START_TS ))
echo "[eval-hf] rollout wall-clock: ${EVAL_ELAPSED}s ($(( EVAL_ELAPSED / 60 )) min)"

# ---------------------------------------------------------------------------
# 4. 汇总成功率（同 run_backbone_eval.sh）
# ---------------------------------------------------------------------------
echo "=============================================================="
echo "[summary] aggregating results for ${MODEL_LOCAL_NAME}"
python "${REPO_ROOT}/examples/eval_backbones/aggregate_results.py" \
    --record_path "${RECORD_ROOT}" \
    --model_name "${MODEL_LOCAL_NAME}" \
    --output_json "${RECORD_ROOT}/summary.json"
cat "${RECORD_ROOT}/summary.json"

OVERALL_TOTAL=$(python3 -c "import json; print(json.load(open('${RECORD_ROOT}/summary.json'))['overall']['total'])")
if [ "${OVERALL_TOTAL}" = "0" ]; then
    echo "[summary][FATAL] 0 rollout results produced -- likely a silent setup/import failure upstream" >&2
    aws s3 sync "${RECORD_ROOT}/" "${RESULT_S3_URI}/" --no-progress || true
    exit 1
fi

echo "[upload] syncing results -> ${RESULT_S3_URI}"
aws s3 sync "${RECORD_ROOT}/" "${RESULT_S3_URI}/" --no-progress

echo "[done] ${MODEL_LOCAL_NAME} (hf mode) evaluation finished. rollout wall-clock: ${EVAL_ELAPSED}s"

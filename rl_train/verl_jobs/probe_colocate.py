"""探针：同一进程里 HF Qwen3.5（训练侧）+ vLLM 0.17（推理侧）能否共存并同步权重。

这是 verl colocated rollout 的最小可行性问题。逐项独立检查、全部打印，不因一项
失败而中断：

  V1 版本：torch / transformers / vllm；transformers 有无 Qwen3_5ForConditionalGeneration
  V2 vLLM 进程内引擎（VLLM_ENABLE_V1_MULTIPROCESSING=0）加载真实 9B 权重 +
     带图生成一次（与评测同 prompt 形态）
  V3 找到进程内 model 对象：探测 llm_engine 下通往 model_runner.model 的属性链
     （verl 权重同步要从这里 load_weights）
  V4 HF 侧加载同一权重、前向算 log_probs（fla 内核）——训练侧可用
  V5 权重同步回路：把 HF 模型的 state_dict 经 model.load_weights 灌进 vLLM，
     同 prompt 贪心生成与同步前逐字一致（验证 HF→vLLM 参数名映射正确）

用法：python probe_colocate.py --model /local-ssd/model_cache
"""
import argparse
import os
import time
import traceback

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

RESULTS = {}


def step(name):
    def deco(fn):
        def run(*a, **kw):
            print(f"\n########## {name}", flush=True)
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                RESULTS[name] = "OK"
                print(f"[{name}] OK ({time.time() - t0:.1f}s)", flush=True)
                return out
            except Exception as e:  # noqa: BLE001
                RESULTS[name] = f"FAIL: {e!r}"[:200]
                traceback.print_exc()
                print(f"[{name}] FAIL ({time.time() - t0:.1f}s): {e!r}", flush=True)
                return None
        return run
    return deco


@step("V1 版本")
def v1():
    import torch, transformers, vllm
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| vllm", vllm.__version__)
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
    print("Qwen3_5ForConditionalGeneration: present")


@step("V2 vLLM 进程内加载+带图生成")
def v2(model_path):
    from vllm import LLM, SamplingParams
    from PIL import Image
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192,
              gpu_memory_utilization=0.45, limit_mm_per_prompt={"image": 1},
              enforce_eager=True)
    img = Image.new("RGB", (640, 360), (90, 140, 70))
    prompt = ("<|im_start|>user\nTask: mine the oak log.<|vision_start|><|image_pad|><|vision_end|>"
              "<|im_end|>\n<|im_start|>assistant\n")
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], sp)
    text = out[0].outputs[0].text
    print(f"greedy 输出: {text!r} | finish={out[0].outputs[0].finish_reason}")
    return llm, prompt, img, text


@step("V3 进程内 model 对象属性链")
def v3(llm):
    candidates = [
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.engine_core.model_executor.driver_worker.worker.model_runner.model",
    ]
    for path in candidates:
        obj = llm
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            print(f"命中: llm.{path} -> {type(obj).__name__}")
            return obj
        except AttributeError as e:
            print(f"未命中: llm.{path} ({e})")
    # 兜底：collective_rpc 是 vLLM 官方 RLHF 接口，属性链全挂时它仍可用
    if hasattr(llm, "collective_rpc"):
        names = llm.collective_rpc(lambda self: type(self.model_runner.model).__name__)
        print(f"collective_rpc 可用，worker 内 model 类型: {names}")
        return "collective_rpc"
    raise RuntimeError("找不到进程内 model 对象，也没有 collective_rpc")


@step("V4 HF 侧前向（fla 内核）")
def v4(model_path):
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").cuda().eval()
    ids = torch.tensor([[151644, 872, 198, 9707, 151645]], device="cuda")
    with torch.no_grad():
        logits = m(input_ids=ids, use_cache=False).logits
    assert torch.isfinite(logits).all()
    print("HF logits", tuple(logits.shape))
    try:
        import fla, causal_conv1d  # noqa: F401
        print("fla + causal_conv1d present")
    except Exception as e:  # noqa: BLE001
        print("⚠ 线性注意力内核缺失（HF 会走纯 PyTorch fallback，慢且吃显存）:", repr(e))
    return m


@step("V5 HF→vLLM 权重同步回路")
def v5(llm, model_obj, hf_model, prompt, img, ref_text):
    from vllm import SamplingParams
    sd = {k: v for k, v in hf_model.state_dict().items()}
    print(f"HF state_dict 张量数: {len(sd)}")
    if model_obj == "collective_rpc":
        def _load(self, items=list(sd.items())):
            return len(self.model_runner.model.load_weights(iter(items)) or [])
        n = llm.collective_rpc(_load)
    else:
        loaded = model_obj.load_weights(iter(sd.items()))
        n = len(loaded) if loaded is not None else -1
    print(f"load_weights 返回已加载参数数: {n}")
    out = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}],
                       SamplingParams(temperature=0.0, max_tokens=32))
    text = out[0].outputs[0].text
    print(f"同步后 greedy: {text!r}")
    assert text == ref_text, "同步前后贪心输出不一致 → 参数名映射或 dtype 有问题"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/local-ssd/model_cache")
    args = ap.parse_args()

    v1()
    r2 = v2(args.model)
    model_obj = v3(r2[0]) if r2 else None
    hf = v4(args.model)
    if r2 and model_obj is not None and hf is not None:
        v5(r2[0], model_obj, hf, r2[1], r2[2], r2[3])

    print("\n========== COLOCATE_PROBE 汇总 ==========")
    for k, v in RESULTS.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

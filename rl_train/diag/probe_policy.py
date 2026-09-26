#!/usr/bin/env python3
"""策略生成行为探针：量测每步输出长度/停止原因，定位"评测慢 70 倍"的根因。

背景（09-26）：自举 iteration-1 权重（minecraft-rl-policy/iter1）的 h29 评测实测
70-80 s/step，而同 harness 下 SFT 权重是 1.04 s/step（快 70 倍）。rollout 输入
策略（v2-e4 + FORCE_THOUGHT）的输出中位数只有 158 字符，说明退化发生在自举训练
之后。本探针不跑 Malmo，直接用 vLLM 离线接口喂固定图文 prompt，对比两组权重的
输出长度与 finish_reason —— 若 iter1 组大量 finish_reason=="length"，即证实
"模型不再生成 EOS、每步顶到 max_tokens"这一退化模式。
"""
import argparse
import json
import statistics

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, trust_remote_code=True, limit_mm_per_prompt={"image": 2},
              max_model_len=8192, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()

    img = Image.new("RGB", (640, 360), color=(90, 140, 70))
    prompt = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              "Task: kill the sheep. Decide the next action.<|im_end|>\n"
              "<|im_start|>assistant\n")
    reqs = [{"prompt": prompt, "multi_modal_data": {"image": img}} for _ in range(args.n)]
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_tokens)
    outs = llm.generate(reqs, sp)

    lens, chars, reasons, samples = [], [], {}, []
    for o in outs:
        c = o.outputs[0]
        lens.append(len(c.token_ids))
        chars.append(len(c.text))
        reasons[c.finish_reason] = reasons.get(c.finish_reason, 0) + 1
        if len(samples) < 3:
            samples.append(c.text[:300])

    print(json.dumps({
        "model": args.model,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "gen_tokens_median": statistics.median(lens),
        "gen_tokens_max": max(lens),
        "gen_chars_median": statistics.median(chars),
        "finish_reasons": reasons,
        "hit_length_cap_pct": round(100 * reasons.get("length", 0) / len(outs), 1),
        "has_action_pct": round(100 * sum(1 for o in outs if "Action:" in o.outputs[0].text) / len(outs), 1),
    }, ensure_ascii=False, indent=2), flush=True)
    for i, s in enumerate(samples):
        print(f"--- 样例{i}: {s!r}", flush=True)


if __name__ == "__main__":
    main()

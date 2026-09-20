"""GRPO trainer（自研 RL v1，09-20 设计文档见 experiment_summary §7）。

复用现有组件（最大化复用 train_sft.py 的骨架）：
- 模型加载（load_model_and_processor，含 S3 下载与 vision tower 冻结）
- 数据管线（build_minecraft_dataset + MultiStepVLMCollator——RL 的训练样本与 SFT
  的 parquet 完全同构：conversations + image_bytes；collator 产出的 labels 天然
  把 loss 位置标在 assistant token 上，正是 RL 需要"生成 token 位置"）
- DeepSpeed ZeRO-2（train_sft 同款配置）

新增（RL 核心）：
- 优势从旁车文件读（--advantages_file，numpy .npy，与 parquet 行序对齐；
  由 grpo_core.compute_group_advantages 离线算好）
- 损失 = -A * logp(token)（组内中心化的 REINFORCE；on-policy 单次更新）
- 可选 KL 惩罚（--kl_beta > 0 时加载冻结的参考模型）

用法（阶段2编排脚本 rl_loop.sh 驱动，亦可手动）:
    torchrun --nproc_per_node=8 train_grpo.py \
        --model_path s3://.../rl_policy_ckpt \
        --data_path s3://.../rl_batch.parquet \
        --advantages_file /local-ssd/rl_batch_adv.npy \
        --output_dir ./rl_out --deepspeed ds_zero2_no_offload.json
"""
import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_sft import _load_model_and_processor as load_model_and_processor  # noqa: E402
from dataset import build_minecraft_dataset  # noqa: E402
from collators import MultiStepVLMCollator  # noqa: E402

import numpy as np  # noqa: E402
# deepspeed 只在 main() 里用——顶层不 import，使 token_logprobs_from_logits 等纯
# tensor 工具函数能在无 deepspeed 的环境（如本地 CPU 冒烟测试）独立导入验证。


def token_logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor,
                               chunk: int = 2048) -> tuple:
    """逐 chunk 计算 label 位置的 logprob（fp32 logsumexp 保数值稳定）。

    logits: [B, T, V]（bf16），labels: [B, T]（-100 = 非生成位置，已与 logits 对齐
    ——调用方需保证传入的 labels 已做过 shift）。
    返回 (logp_sum_per_sample [B], token_mask_sum_per_sample [B])
    """
    B = logits.shape[0]
    logp_sum = torch.zeros(B, device=logits.device, dtype=torch.float32)
    n_tok = torch.zeros(B, device=logits.device, dtype=torch.float32)
    T = logits.shape[1]
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lg = logits[:, s:e].float()
        lb = labels[:, s:e]
        mask = lb != -100
        if not mask.any():
            continue
        logp = torch.log_softmax(lg, dim=-1)
        gathered = logp.gather(-1, lb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        gathered = gathered * mask
        logp_sum += gathered.sum(dim=-1)
        n_tok += mask.sum(dim=-1).float()
    return logp_sum, n_tok


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--advantages_file", required=True, help=".npy，与 parquet 行序对齐")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--s3_output_dir", default=None)
    ap.add_argument("--deepspeed", default=os.path.join(SCRIPT_DIR, "ds_zero2_no_offload.json"))
    ap.add_argument("--attn_implementation", default="flash_attention_2")
    ap.add_argument("--freeze_vision_tower", action="store_true", default=True)
    ap.add_argument("--download_model", default=None,
                    help="S3 模型本地缓存目录，透传给 train_sft._load_model_and_processor")
    ap.add_argument("--max_seq_length", type=int, default=19456)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--kl_beta", type=float, default=0.0,
                    help=">0 时加载参考模型计算 KL 惩罚（显存代价 +18GB/卡）")
    ap.add_argument("--ref_model_path", default=None, help="KL 参考模型，默认=model_path")
    ap.add_argument("--keep_no_op_p", type=float, default=1.0)
    ap.add_argument("--focal_decay", type=float, default=1.0,
                    help="RL 阶段默认关闭 focal（RL 损失自带 advantage 权重）")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = 单遍数据")
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_rank", type=int, default=-1)
    return ap.parse_args()


def main():
    import deepspeed  # 延迟导入：训练入口才需要，见顶部说明

    args = build_args()
    torch.manual_seed(args.seed)

    model, processor = load_model_and_processor(args)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ref_model = None
    if args.kl_beta > 0:
        ref_path = args.ref_model_path or args.model_path
        from transformers import AutoModelForImageTextToText
        ref_model = AutoModelForImageTextToText.from_pretrained(
            ref_path if not ref_path.startswith("s3://") else ref_path,
            dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {
            "lr": args.learning_rate, "weight_decay": 0.0, "betas": [0.9, 0.95]}},
        "scheduler": {"type": "WarmupDecayLR", "params": {
            "warmup_min_lr": 0, "warmup_max_lr": args.learning_rate,
            "warmup_num_steps": 10, "total_num_steps": 10_000_000}},
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 2},
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model, config=ds_config, model_parameters=model.parameters())

    dataset = build_minecraft_dataset(
        data_path=args.data_path, streaming=False, data_format="parquet",
        processor=processor, max_seq_length=args.max_seq_length,
        full_trajectory=True, keep_no_op_p=args.keep_no_op_p,
        no_op_seed=args.seed, num_proc=8,
    )
    advantages = np.load(args.advantages_file)
    assert len(advantages) == len(dataset), \
        f"advantages({len(advantages)}) 与数据集({len(dataset)})行数不一致"
    print(f"[rl] 数据集 {len(dataset)} 行, 优势非零比例 "
          f"{100*(advantages != 0).mean():.1f}%", flush=True)

    collator = MultiStepVLMCollator(
        processor=processor, max_length=args.max_seq_length,
        focal_decay=args.focal_decay, focal_seed=args.seed)

    # on-policy 单次更新：按序迭代（与 SFT 的 SequentialSampler 语义一致），
    # 微批手动组包——advantage 通过行号从旁车数组取，不进数据管线
    indices = list(range(len(dataset)))
    B = args.per_device_batch_size
    n_batches = (len(indices) + B - 1) // B
    total_updates = args.max_steps or (n_batches // args.gradient_accumulation_steps)
    print(f"[rl] {n_batches} 微批 / {total_updates} 更新步", flush=True)

    update = 0
    accum = 0
    for bi in range(n_batches):
        if update >= total_updates:
            break
        rows = indices[bi * B:(bi + 1) * B]
        batch = collator([dataset[i] for i in rows])
        adv = torch.tensor(advantages[rows], dtype=torch.float32,
                           device=engine.device)
        batch_dev = {k: (v.to(engine.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        labels = batch_dev["labels"]
        shift_labels = labels[:, 1:].clone()
        out = engine(**{k: v for k, v in batch_dev.items() if k != "labels"},
                     use_cache=False)
        logp_sum, n_tok = token_logprobs_from_logits(out.logits[:, :-1], shift_labels)

        # 每样本按其生成 token 数归一（Dr.GRPO 风格，去长度偏差），再加权 advantage
        per_sample_pg = -(logp_sum / n_tok.clamp(min=1))
        loss = (per_sample_pg * adv.to(per_sample_pg.device)).mean()

        if ref_model is not None:
            with torch.no_grad():
                ref_out = ref_model(
                    **{k: v for k, v in batch_dev.items() if k != "labels"},
                    use_cache=False)
            ref_logp, _ = token_logprobs_from_logits(ref_out.logits[:, :-1], shift_labels)
            kl = (logp_sum - ref_logp) / n_tok.clamp(min=1)
            loss = loss + args.kl_beta * kl.mean()

        engine.backward(loss)
        engine.step()
        accum += 1
        if accum % args.gradient_accumulation_steps == 0:
            update += 1
            accum = 0
            if update % args.logging_steps == 0:
                with torch.no_grad():
                    pg_val = per_sample_pg.detach().mean().item()
                print(f"[rl] step {update}/{total_updates} | loss={loss.item():.4f} "
                      f"pg={pg_val:.4f} |adv|={adv.abs().mean().item():.3f}", flush=True)
            if update % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{update}")
                engine.save_16bit_model(save_dir, "model.safetensors")
                if processor:
                    processor.save_pretrained(save_dir)
                if args.s3_output_dir and engine.global_rank == 0:
                    os.system(f"aws s3 sync {save_dir}/ "
                              f"{args.s3_output_dir}/checkpoint-{update}/ --only-show-errors")
                print(f"[rl] 已存 checkpoint-{update}", flush=True)

    final_dir = os.path.join(args.output_dir, "final")
    engine.save_16bit_model(final_dir, "model.safetensors")
    processor.save_pretrained(final_dir)
    if args.s3_output_dir and engine.global_rank == 0:
        os.system(f"aws s3 sync {final_dir}/ {args.s3_output_dir}/final/ --only-show-errors")
    print("[rl] 训练完成", flush=True)


if __name__ == "__main__":
    main()

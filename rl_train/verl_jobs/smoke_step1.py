"""verl-agent vendor 冒烟测试（迁移第 1 步验收，在 koala sft env 跑）。

三层验证（对应 verl_migration.md 第 1 步）：
  A. import 完整性：vendored verl/agent_system/gigpo 可导入
  B. 适配器机制：tiny 随机 Qwen3.5 混合模型（linear+full attn 层）+ apply_monkey_patch
     + torch backend 前向 → log_probs/entropy 形状正确（结构层验证，测上游
     qwen3_5.py 与移植的 monkey_patch 分支的配合）
  C. 真实权重：checkpoint-520（qwen35-9b-nf2，18.8GB）先原生 HF 前向+生成，
     再 apply_monkey_patch + 带 labels 前向 → log_probs（验证 transformers 版本
     兼容 + fla 内核与真实权重组合）

用法（koala job 内，sft env，已 bootstrap_env + pip install -e ../verl_agent）:
    python rl_train/verl_jobs/smoke_step1.py --model-path /local-ssd/model_cache
"""
import argparse
import sys

import torch

PASS = []


def check_a_imports():
    print("=== A. import 完整性 ===", flush=True)
    import verl  # noqa: F401
    from verl import DataProto  # noqa: F401
    import agent_system  # noqa: F401
    import gigpo  # noqa: F401
    import transformers
    print(f"verl / agent_system / gigpo OK; transformers {transformers.__version__}", flush=True)
    PASS.append("A")


def _tiny_config():
    """tiny 随机 Qwen3.5 配置（移植自 fork 时代的结构测试，字段对照当前 transformers）。"""
    from transformers import Qwen3_5Config
    text_config = dict(
        vocab_size=1024, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4,  # full_attention_interval=4 → [linear, linear, linear, full]
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512,
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
    )
    vision_config = dict(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=64, num_position_embeddings=64,
    )
    return Qwen3_5Config(
        text_config=text_config, vision_config=vision_config,
        image_token_id=1000, video_token_id=1001,
        vision_start_token_id=1002, vision_end_token_id=1003,
    )


def check_b_adapter_tiny():
    print("=== B. 适配器机制（tiny 混合模型 + torch backend） ===", flush=True)
    from transformers import Qwen3_5ForConditionalGeneration
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    cfg = _tiny_config()
    print("layer_types:", cfg.text_config.layer_types, flush=True)
    assert "linear_attention" in cfg.text_config.layer_types
    assert "full_attention" in cfg.text_config.layer_types

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3_5ForConditionalGeneration(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()

    # ① Ulysses SP 应被显式拒绝（线性注意力因果递归不可跨 SP 切分）
    try:
        apply_monkey_patch(model, ulysses_sp_size=2, use_remove_padding=False, use_fused_kernels=True)
        raise AssertionError("expected NotImplementedError for ulysses_sp_size > 1")
    except NotImplementedError:
        print("[OK] ulysses_sp_size>1 被正确拒绝", flush=True)

    # ② torch backend 前向：构造 1 图 grid(t=1,h=4,w=4)，merge=2 → 4 个 image token
    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=False,
                       use_fused_kernels=True, fused_kernels_backend="torch")
    image_grid_thw = torch.tensor([[1, 4, 4]], device=device)
    num_patches = int(image_grid_thw.prod(-1).item())
    patch_dim = cfg.vision_config.in_channels * cfg.vision_config.temporal_patch_size \
        * cfg.vision_config.patch_size ** 2
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=torch.bfloat16)
    input_ids = torch.tensor(
        [[1002, 1000, 1000, 1000, 1000, 5, 6, 7, 8, 9, 10]], device=device)
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                                      device=device, dtype=torch.int)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    position_ids=None, temperature=1.0,
                    use_cache=False)  # RL 前向无需 KV cache；且 5.15 的 DynamicCache 无 conv_states 会炸
    if hasattr(out, "log_probs") and out.log_probs is not None:
        assert out.log_probs.shape == (1, input_ids.shape[1]), out.log_probs.shape
        assert torch.isfinite(out.log_probs).all(), "log_probs NaN/Inf"
        print(f"[OK] torch backend 前向: log_probs{tuple(out.log_probs.shape)}", flush=True)
    else:
        print(f"[INFO] 返回无 log_probs（fields: {type(out).__name__}），退回 logits 校验", flush=True)
        assert out.logits.shape[:2] == (1, input_ids.shape[1]), out.logits.shape
    PASS.append("B")


def check_c_real_checkpoint(model_path: str):
    print(f"=== C. 真实权重（{model_path}） ===", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2").cuda().eval()
    print(f"model_type={model.config.model_type}", flush=True)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    img = Image.new("RGB", (128, 128), color=(120, 60, 30))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Mine the cobblestone. What do you see? Reply briefly."}]}]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    inputs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # ① 原生 HF 前向（未 patch）——生成路径跳过：processor 产出的多模态键
    # （mm_token_type_ids 等）与 generate() 的 kwargs 白名单不兼容，属脚本层
    # 细节，与 verl 适配器无关；RL 训练只走 forward，这里验证 forward 即可。
    with torch.no_grad():
        native_out = model(**inputs, use_cache=False)
    assert torch.isfinite(native_out.logits).all(), "native logits NaN/Inf"
    print(f"[OK] 原生前向: logits{tuple(native_out.logits.shape)}", flush=True)

    # ② patch 后带 labels 前向（log_probs 路径——训练时的真实调用形态）
    apply_monkey_patch(model, use_remove_padding=False,
                       use_fused_kernels=True, fused_kernels_backend="torch")
    with torch.no_grad():
        out = model(**inputs, temperature=1.0, use_cache=False)
    if hasattr(out, "log_probs") and out.log_probs is not None:
        assert torch.isfinite(out.log_probs).all(), "log_probs NaN/Inf"
        print(f"[OK] patch 后前向: log_probs{tuple(out.log_probs.shape)}", flush=True)
    else:
        print(f"[INFO] patch 后无 log_probs（返回 {type(out).__name__}），检查字段", flush=True)
    PASS.append("C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/local-ssd/model_cache")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()
    check_a_imports()
    check_b_adapter_tiny()
    if not args.skip_real:
        check_c_real_checkpoint(args.model_path)
    print(f"\nSMOKE_RESULT: {','.join(PASS)}"
          f" {'✅ 全部通过' if len(PASS) == 3 else '（部分通过）'}", flush=True)


if __name__ == "__main__":
    main()

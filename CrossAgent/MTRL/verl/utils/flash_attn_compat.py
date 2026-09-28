"""Small PyTorch fallbacks for the FlashAttention padding helpers.

These helpers are only used by the remove-padding path. They keep the
training code importable on machines where FlashAttention is unavailable;
the real FlashAttention implementation is still preferred when installed.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from einops import rearrange
except ImportError:  # pragma: no cover - einops is a normal verl dependency
    def rearrange(tensor: torch.Tensor, pattern: str, **_: Any) -> torch.Tensor:
        if pattern == "b s ... -> (b s) ...":
            return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
        if pattern == "c b s ... -> (b s) c ...":
            return tensor.permute(1, 2, 0, *range(3, tensor.ndim)).reshape(
                tensor.shape[1] * tensor.shape[2], *tensor.shape[0:1], *tensor.shape[3:]
            )
        raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def index_first_axis(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select rows along the first axis, matching flash_attn.bert_padding."""
    return hidden_states.index_select(0, indices)


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Remove masked tokens and return the FlashAttention-compatible metadata."""
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have shape [batch, sequence, ...]")
    batch_size, sequence_length = hidden_states.shape[:2]
    mask = attention_mask.to(dtype=torch.bool).reshape(batch_size, sequence_length)
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    flat_states = hidden_states.reshape(batch_size * sequence_length, *hidden_states.shape[2:])
    unpadded = flat_states.index_select(0, indices)

    sequence_lengths = mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=mask.device)
    cu_seqlens[1:] = torch.cumsum(sequence_lengths, dim=0)
    max_seqlen = int(sequence_lengths.max().item()) if batch_size else 0
    return unpadded, indices, cu_seqlens, max_seqlen, sequence_lengths


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Restore a flattened tensor to ``[batch, sequence, ...]`` layout."""
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(0, indices, hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])

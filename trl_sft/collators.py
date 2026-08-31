"""Model helpers and VLM collator adapters for SFT training.

- `freeze_vision_tower`: freeze ViT + adapter (JARVIS-VLA Stage I recipe).
- `_ImmutableVisionCollatorAdapter`: give TRL's mutating VLM collator disposable
  sample containers so dataset rows stay pristine.
- `_MultiStepVLMCollator`: TRL's VLM collator subclass that masks non-assistant
  tokens to -100, training on every assistant turn (Stage III full-trajectory).
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Set

import torch
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

logger = logging.getLogger(__name__)

# ─── model helpers ─────────────────────────────────────────────────────────────

# Substrings matched (case-insensitively) against each submodule's own leaf name to
# find the vision tower. For Qwen2-VL/Qwen2.5-VL/Qwen3-VL/Qwen3.5-VL, ViT + adapter
# both live under one submodule named "visual", so freezing it matches JARVIS-VLA's
# Stage I recipe ("ViT + adapter frozen, only LLM trained"). Other hints are fallbacks
# for architectures that split encoder/adapter differently.
_VISION_SUBMODULE_HINTS = ("visual", "vision_tower", "vision_model", "image_encoder")


def freeze_vision_tower(model: torch.nn.Module) -> None:
    """
    Freeze the vision encoder + adapter, leaving only the LLM backbone trainable --
    JARVIS-VLA's Stage I recipe (Stage II unfreezes everything again once real image
    data is introduced, so only call this for `--text_only` runs).

    Walks `model.named_modules()` for a submodule whose own name matches
    `_VISION_SUBMODULE_HINTS` and freezes every parameter under it (skipping submodules
    already nested inside a frozen one). Raises `RuntimeError` if nothing matches --
    silently no-op'ing would look like Stage I ran correctly while actually training
    the full model.
    """
    frozen_modules: List[str] = []
    frozen_params = 0

    for name, module in model.named_modules():
        if not name:
            continue
        leaf_name = name.rsplit(".", 1)[-1].lower()
        if not any(hint in leaf_name for hint in _VISION_SUBMODULE_HINTS):
            continue
        if any(name == m or name.startswith(f"{m}.") for m in frozen_modules):
            continue  # nested inside an already-frozen submodule

        n = 0
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n += p.numel()
        if n > 0:
            frozen_modules.append(name)
            frozen_params += n

    if not frozen_modules:
        raise RuntimeError(
            "freeze_vision_tower: could not find any submodule matching "
            f"{_VISION_SUBMODULE_HINTS} (case-insensitive, matched against each "
            "submodule's own leaf name) on this model. Inspect `model.named_modules()` "
            "for this architecture and extend `_VISION_SUBMODULE_HINTS`."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"freeze_vision_tower: froze {frozen_modules} ({frozen_params:,} params). "
        f"Trainable now: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)."
    )


# ─── immutable VLM collator adapter ───────────────────────────────────────────


def _clone_conversation(messages):
    """Clone mutable chat containers while retaining immutable/PIL payload references."""
    if not isinstance(messages, list):
        return messages
    cloned_messages = []
    for message in messages:
        if not isinstance(message, dict):
            cloned_messages.append(message)
            continue
        cloned_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            cloned_message["content"] = [dict(item) if isinstance(item, dict) else item for item in content]
        cloned_messages.append(cloned_message)
    return cloned_messages


class ImmutableVisionCollatorAdapter:
    """Give TRL's mutating VLM collator disposable sample containers.

    `DataCollatorForVisionLanguageModeling` injects decoded images into prompt content
    and writes the resulting messages back to the supplied example dict. Dataset rows
    must remain pristine because an iterable/dataloader may hand the same Python object
    to the collator again. This adapter copies only the mutable dict/list structure;
    decoded PIL images remain shared references, so it does not duplicate pixel memory.
    It intentionally does not catch or alter collator exceptions.
    """

    def __init__(self, inner_collator):
        self.inner_collator = inner_collator

    def __call__(self, examples):
        working_examples = []
        for example in examples:
            working = dict(example)
            for field in ("messages", "prompt", "completion"):
                if field in working:
                    working[field] = _clone_conversation(working[field])
            if isinstance(working.get("images"), list):
                working["images"] = list(working["images"])
            working_examples.append(working)
        return self.inner_collator(working_examples)


def _assistant_turn_text(message: Dict) -> str:
    """Flatten one chat message's content list into the plain text used for focal
    repeat-detection. Mirrors what VeOmni compares: the assistant turn's rendered
    textual content (for us, "Action: move(...) and press(...)"), ignoring any
    non-text content items.

    Deliberately duck-typed over the content sequence instead of requiring `list`:
    depending on where a row came from, `content` can be a real list (built by
    `dataset.py::_build_full_trajectory`), or a `numpy.ndarray`/tuple (a raw
    parquet/Arrow round-trip hands back ndarrays -- verified against the real
    minecraft-text-action-dataset). An `isinstance(content, list)` check silently
    returned "" for those, which made EVERY assistant turn compare equal and had focal
    drop 86% of all steps including 82% of genuine decision points -- far worse than no
    focal at all. Caught by the unit test in this module's test suite.
    """
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    try:
        items = list(content)
    except TypeError:
        return ""
    parts = [item.get("text") or "" for item in items if isinstance(item, dict) and item.get("type") == "text"]
    return " ".join(parts).strip()


class MultiStepVLMCollator(DataCollatorForVisionLanguageModeling):
    """TRL's VLM collator, but only assistant turns contribute to the loss, with
    VeOmni's "focal" down-weighting of consecutively-repeated actions.

    The parent already injects images, renders the chat template, expands image tokens
    and pads; its `_collate_language_modeling` sets `labels = input_ids` (loss on every
    token except padding). We only override that labels step: assistant turns are
    delimited by `<|im_start|>assistant` ... `<|im_end|>`, so every token outside those
    spans (system/user/image/padding) is masked to -100. This reproduces JARVIS-VLA
    Stage III, where every "Action: ..." turn is a training target.

    On top of that, `focal_decay` (<1.0) reproduces VeOmni's
    `Qwen2_5VLFocalChatTemplate` repeat suppression -- the mechanism its real, published
    TextVLA training runs used (`--data.chat_template qwen2_5vlwithfocal*`). Within a
    trajectory it walks the assistant turns in order and, whenever a turn's text is
    IDENTICAL to the previous assistant turn's, decays `alpha *= focal_decay` and drops
    that whole turn from the loss with probability `1 - alpha`; any turn whose text
    differs resets `alpha = 1.0` and is always kept. Consecutive repeats therefore get
    progressively more likely to be skipped, while every "the action just changed"
    decision point keeps full supervision.

    Why this is necessary (measured on the real `minecraft-text-action-dataset`): 56.1%
    of assistant steps are byte-identical to the immediately preceding step and 37.3%
    are a pure no-op (`move(0, 0) and press()`). Because the loss is a plain mean over
    unmasked tokens, that majority class numerically dominates training, and our first
    Stage III runs without this mechanism collapsed exactly as predicted -- evaluating
    checkpoint-1400 gave 200/200 identical `move(0, 0) and press()` steps on
    kill_entity/mine_block rollouts (0% success), while the much earlier checkpoint-200
    still produced varied actions. Note this mechanism acts at the LOSS level only:
    OpenHA's published text-action pipeline does no no-op filtering either (its
    `keep_no_op_p` defaults to 1.0 and is never overridden for text actions), so the
    original distribution is kept and only its gradient contribution is rebalanced.

    The complementary DATA-level knob (OpenHA's `keep_no_op_p`, which its MotionTokenizer
    route enables by default and which VPT's own data loader applies as a hard skip) lives
    in `dataset.py::_no_op_dropped_turns` behind `--keep_no_op_p` (default 1.0 = off).
    The two are independent and compose: focal only ever masks loss, so a frame dropped
    upstream is simply never seen here.

    `focal_decay=1.0` disables the mechanism entirely (alpha stays 1.0, so
    `random() > alpha` is never true) and restores the previous behaviour.
    """

    def __init__(self, *args, focal_decay: float = 0.75, focal_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= focal_decay <= 1.0:
            raise ValueError(f"focal_decay must be in [0.0, 1.0], got {focal_decay}")
        self.focal_decay = focal_decay
        self._warned_empty_text = False
        # Dedicated RNG so enabling/disabling focal never perturbs the global `random`
        # stream (which other data-pipeline code may rely on), and so the drop pattern
        # is reproducible for a given seed. Per-sample masks need not agree across
        # ranks: every rank collates its own disjoint samples.
        self._focal_rng = random.Random(focal_seed)

    def _focal_dropped_turns(self, example: Dict) -> Set[int]:
        """Indices (counted over assistant turns only, in order) whose loss to drop.

        Empty when focal is disabled or the example has no `messages` (e.g. a
        prompt/completion-shaped row), so those paths behave exactly as before.
        """
        if self.focal_decay >= 1.0:
            return set()
        messages = example.get("messages")
        if not isinstance(messages, list):
            return set()

        dropped: Set[int] = set()
        assistant_idx = -1
        total_assistant = 0
        n_empty_text = 0
        last_text = None
        alpha = 1.0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            assistant_idx += 1
            total_assistant += 1
            text = _assistant_turn_text(message)
            if not text:
                n_empty_text += 1
            if last_text is not None and text == last_text:
                alpha *= self.focal_decay
                if self._focal_rng.random() > alpha:
                    dropped.add(assistant_idx)
            else:
                # A changed action (and the trajectory's first assistant turn, where
                # `last_text is None`) resets the decay and is never dropped.
                alpha = 1.0
            last_text = text

        # Fail SAFE, not silent: if we could not read text out of any assistant turn,
        # every turn compares equal to the previous one and focal would shred the
        # trajectory (measured 86% of steps, including 82% of real decision points, when
        # a content-type mismatch made extraction return ""). That is strictly worse than
        # no focal at all, so treat it as "focal not applicable" and say so loudly.
        if total_assistant and n_empty_text == total_assistant:
            if not self._warned_empty_text:
                self._warned_empty_text = True
                logger.warning(
                    "MultiStepVLMCollator: could not extract any assistant text for focal "
                    "repeat-detection (all %d assistant turns produced empty text). Focal "
                    "suppression is DISABLED for such samples to avoid mass-dropping the "
                    "loss. Check the shape of messages[i]['content'] -- see "
                    "`_assistant_turn_text`.",
                    total_assistant,
                )
            return set()

        # The first assistant turn always survives (alpha is 1.0 there), so a fully
        # dropped trajectory should be unreachable -- but an all -100 row makes the loss
        # NaN and would silently poison training, so keep one turn defensively.
        if total_assistant and len(dropped) >= total_assistant:
            dropped.discard(assistant_idx)
        return dropped

    def _collate_language_modeling(self, examples):
        # Computed BEFORE `super()`, which injects decoded images into the examples'
        # content lists as it renders the chat template.
        dropped_per_example = [self._focal_dropped_turns(example) for example in examples]

        output = super()._collate_language_modeling(examples)
        tok = self.processor.tokenizer
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        assistant_id = tok.convert_tokens_to_ids("assistant")
        labels = output["input_ids"].clone()
        for b, row in enumerate(output["input_ids"].tolist()):
            dropped = dropped_per_example[b] if b < len(dropped_per_example) else set()
            in_assistant = False
            assistant_idx = -1
            for t, tok_id in enumerate(row):
                if tok_id == im_start and t + 1 < len(row) and row[t + 1] == assistant_id:
                    # Mask the two format tokens that open an assistant turn
                    # (<|im_start|> and the "assistant" role token); only the
                    # content tokens that follow contribute to the loss.
                    labels[b, t] = -100
                    labels[b, t + 1] = -100
                    assistant_idx += 1
                    # A focal-dropped turn simply never enters the "inside assistant"
                    # state, so the masking below strips its content AND its closing
                    # <|im_end|> -- i.e. the whole turn contributes no loss, matching
                    # VeOmni's per-message `loss_mask = 0`.
                    in_assistant = assistant_idx not in dropped
                    continue
                if tok_id == im_end:
                    # Do NOT mask the closing <|im_end|> of a KEPT assistant turn: it
                    # doubles as the EOS signal for that turn, and it is the ONLY
                    # supervision the model gets for "this action is finished, stop
                    # generating". Masking it (as an earlier version of this collator
                    # did) produced a checkpoint that never emitted EOS at inference
                    # time -- every rollout step ran to the max_tokens cap and
                    # degenerated into repeated filler ("... 0 0 0 0" / "and
                    # click(left) and click(right) ..."), which corrupted the parsed
                    # action and drove the real-env success rate to 0%. Verified by
                    # contrast: a checkpoint trained with <|im_end|> in the loss emits
                    # ~31-char actions and stops cleanly; the masked one emitted
                    # 1630-5588 chars at every single step.
                    if not in_assistant:
                        labels[b, t] = -100
                    in_assistant = False
                    continue
                if not in_assistant:
                    labels[b, t] = -100
        labels[output["attention_mask"] == 0] = -100
        output["labels"] = labels
        return output

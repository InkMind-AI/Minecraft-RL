"""Dataset construction for Minecraft SFT: message building, tokenization, and
dataset loading for parquet trajectory and jsonl flat-QA formats.

Supports two on-disk layouts:
  - parquet trajectories (e.g. minecraft-text-action-dataset): each row is one
    trajectory with embedded `image_bytes` per turn.
  - jsonl flat QA (e.g. minecraft-vlp): each row is a short independent Q&A
    session with image paths relative to an `image_root`.

Also supports `text_only` mode (Stage I, no images) and `full_trajectory` mode
(Stage III, multi-step loss on every assistant turn), plus `keep_no_op_p`
(data-level no-op frame dropping, see `_is_no_op_action_text`).

Non-streaming (`streaming=False`, now the default) materializes the dataset into
an Arrow cache (known length -> exact-epoch max_steps, index-level resume skip,
optional `num_proc` parallel preprocessing). Streaming (`--streaming`) keeps the
legacy IterableDataset path (unknown length, sequential order, replays the whole
map pipeline on every resume).

Image contract (both modes): sample dicts carry `images` as the RAW encoded bytes
(JPEG/PNG) exactly as stored on disk -- decoding to PIL happens in the collators
(`collators.py::_decode_raw_images`). This keeps the non-streaming Arrow cache
byte-identical to the source files (no PNG re-encode blowup, no Image-feature
magic) and makes dataloader-worker pickling cheap. Images are still decoded once
inside the map function purely to validate them (corrupt image -> "[image]" text
fallback, same as before).
"""

from __future__ import annotations

import io
import logging
import random
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from datasets import concatenate_datasets, load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


# ─── no-op action detection (data-level `keep_no_op_p` dropping) ───────────────

# Text-action serialization produced by `openagents/agents/utils/action_mapping.py::
# TextActionTokenizer._json_to_text`: "Action: " + " and ".join(parts), where parts are
# `move(dx, dy)` / `press(k1, k2, ...)` / `click(left|right|middle)`, or the literal
# "no_op" when every part is empty. `minecraft-text-action-dataset` was generated with
# `reserved_camera=reserved_keyboard=True`, so a frame in which the human did nothing at
# all still renders as the FULL "Action: move(0, 0) and press()" rather than
# "Action: no_op" -- both spellings are therefore accepted below.
#
# Verified against the real dataset (train-00000-of-00363.parquet, row group 0, 2831
# assistant steps): "Action: move(0, 0) and press()" is exactly 24.8% of all steps (the
# single most common action by a factor of ~3), and 53.1% of steps are byte-identical to
# the immediately preceding step.
_ACTION_SPLIT_RE = re.compile(r"Action\s*:")
_CAMERA_RE = re.compile(r"""move\(\s*['"]?(-?\d+(?:\.\d+)?)['"]?\s*,\s*['"]?(-?\d+(?:\.\d+)?)['"]?\s*\)""")
_KEYBOARD_RE = re.compile(r"press\(([^)]*)\)")
_CLICK_RE = re.compile(r"click\(")


def _is_no_op_action_text(text: str) -> bool:
    """Whether one assistant turn's text is a pure no-op (camera still, no keys, no clicks).

    Parses the action instead of substring-matching one exact spelling, so it stays
    correct across `TextActionTokenizer`'s variants (`Action: no_op` vs the reserved-field
    `Action: move(0, 0) and press()`), whitespace/quoting differences, float-vs-int
    coordinates, and `action_chunk_len > 1` (several "Action: ..." segments in one turn --
    ALL of them must be no-op for the turn to count as one).

    Deliberately FAIL-SAFE: anything unrecognized (empty text, an action naming neither
    `move` nor `press` nor `no_op`) returns False, i.e. "not a no-op, never drop it".
    Getting this wrong in the other direction would silently delete real decision points
    from training -- the exact failure mode that a content-type mismatch once caused in
    `collators.py::_assistant_turn_text` (86% of steps dropped, including 82% of genuine
    decision points).
    """
    text = (text or "").strip()
    if not text:
        return False
    segments = [seg.strip() for seg in _ACTION_SPLIT_RE.split(text) if seg.strip()]
    if not segments:
        return False
    for seg in segments:
        if seg == "no_op":
            continue
        if _CLICK_RE.search(seg):
            return False
        camera = _CAMERA_RE.search(seg)
        if camera and (float(camera.group(1)) != 0.0 or float(camera.group(2)) != 0.0):
            return False
        keyboard = _KEYBOARD_RE.search(seg)
        if keyboard and keyboard.group(1).strip():
            return False
        if camera is None and keyboard is None:
            return False  # unrecognized action text -> fail safe
    return True


def _turn_text(message: Dict) -> str:
    """Flatten one chat message's text content items into plain text.

    Duck-typed over the content sequence (not `isinstance(..., list)`) because a raw
    parquet/Arrow round-trip can hand back a `numpy.ndarray` or tuple instead of a real
    list; requiring `list` there is what once made every assistant turn compare equal in
    `collators.py`.
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


def _no_op_dropped_turns(
    conversations: List[Dict],
    keep_no_op_p: float,
    rng: random.Random,
) -> Set[int]:
    """Turn indices to delete outright, implementing OpenHA's `keep_no_op_p` at the DATA
    level (`action_mapping.py::TextActionTokenizer.encode` drops the frame's `frame_id`,
    so both the observation and the action vanish from the encoded trajectory).

    Each no-op assistant turn is kept with probability `keep_no_op_p` and otherwise
    dropped TOGETHER WITH the user turn immediately before it -- that user turn holds the
    frame's observation image, and keeping an observation whose action was deleted would
    both break the strict user/assistant alternation and silently re-pair every image with
    the *next* frame's action.

    A user turn that carries text content is never dropped (and so neither is its
    assistant turn): in `minecraft-text-action-dataset` that is exactly the first turn,
    which holds the system prompt + task instruction. This also guarantees at least one
    assistant turn always survives, so the trajectory can never be emptied.

    Returns an empty set when `keep_no_op_p >= 1.0` (mechanism disabled), making that path
    byte-identical to the previous behaviour.
    """
    if keep_no_op_p >= 1.0:
        return set()

    dropped: Set[int] = set()
    for idx, conv in enumerate(conversations):
        if conv.get("role") != "assistant":
            continue
        if not _is_no_op_action_text(_turn_text(conv)):
            continue
        # Mirrors OpenHA's `random.random() > keep_no_op_p` predicate exactly, so
        # keep_no_op_p=0.0 drops every (droppable) no-op and 1.0 drops none.
        if not rng.random() > keep_no_op_p:
            continue
        prev = idx - 1
        if prev < 0 or conversations[prev].get("role") != "user":
            continue
        prev_content = conversations[prev].get("content")
        try:
            prev_items = list(prev_content) if prev_content is not None else []
        except TypeError:
            continue
        if any(isinstance(item, dict) and item.get("type") == "text" for item in prev_items):
            continue  # instruction-bearing turn: keep it (and its action)
        dropped.add(prev)
        dropped.add(idx)
    return dropped


# ─── dataset helpers ──────────────────────────────────────────────────────────


def _decode_images(images: List[bytes]) -> List[Image.Image]:
    """Decode the sample's raw `images` bytes into PIL images (transient use).

    Only needed where PIL metadata (height/width) is required on the spot -- i.e.
    the `_exceeds_max_length` pre-check inside `_row_to_trl_sample`. The training
    samples themselves keep the raw bytes all the way to the collator (see the
    module docstring's image contract).
    """
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in images]


def _split_prompt_completion_with_images(
    conversations: List[Dict],
    image_bytes_list: List[bytes],
) -> Tuple[List[Dict], List[Dict], List[bytes]]:
    """
    Shared tail used by `build_messages_qa` (flat QA/jsonl rows): walks
    `conversations`, keeps `{"type": "image"}` placeholders in-place while
    validating the matching entry of the flat `image_bytes_list` (one entry per
    placeholder, in encounter order across the whole conversation) and collecting
    the RAW bytes into a separate `images` list (see the module docstring's image
    contract; PIL decoding happens in the collators). Finally splits off the final
    assistant turn as `completion`, with everything before it as `prompt`.
    """
    messages = []
    images: List[bytes] = []
    image_idx = 0

    for conv in conversations:
        role = conv["role"]
        content_list = []

        for item in conv["content"]:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                if image_idx < len(image_bytes_list):
                    try:
                        Image.open(io.BytesIO(image_bytes_list[image_idx])).convert("RGB")  # validate only
                        content_list.append({"type": "image"})
                        images.append(image_bytes_list[image_idx])
                    except Exception as e:
                        logger.warning(f"Failed to decode image at idx {image_idx}: {e}")
                        content_list.append({"type": "text", "text": "[image]"})
                image_idx += 1
            elif item.get("type") == "point":
                # Grounding rows (e.g. minecraft-vlp/mc-grounding-point-*.jsonl, JARVIS-VLA
                # Stage II spatial-grounding data): the assistant turn's answer is
                # structured `{"point": [[x, y], ...], "label": "..."}` -- x/y are
                # percentages (0-100) of image width/height, not pixels, and a single
                # turn can list multiple points (e.g. "Spot the 2 slot" -> up to ~10
                # points for repeated/ambiguous targets). There is no free-form "text"
                # answer to fall back on, so serialize the coordinates into plain text
                # the LM can actually be trained to generate: "(x, y)" per point,
                # multiple points joined with "; ". This keeps the format simple/
                # deterministic and resolution-independent (matches the 0-100 percentage
                # scale already used by the raw labels).
                points = item.get("point") or []
                coord_text = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in points)
                content_list.append({"type": "text", "text": coord_text})

        # TRL expects roles: "user", "assistant", "system"
        messages.append({"role": role, "content": content_list})

    # Split off the final assistant turn as the `completion`; everything before it
    # becomes the `prompt` (context that `completion_only_loss` will mask out).
    prompt, completion = messages[:-1], [messages[-1]]

    # TRL's collator requires #`{"type":"image"}` placeholders WITHIN `prompt` to equal
    # `len(images)`, or it raises mid-training. A small minority of source rows put an
    # image placeholder in the FINAL (assistant) turn -- which becomes `completion`, not
    # `prompt` -- so that placeholder goes uncounted while `images` still includes it.
    # Caught two real 16-GPU jobs crashing ~13h in on the one bad row out of hundreds of
    # thousands; checking here converts that into a silent per-row drop instead.
    num_prompt_placeholders = sum(
        1 for m in prompt for item in m["content"] if item.get("type") == "image"
    )
    if num_prompt_placeholders != len(images):
        logger.warning(
            f"Dropping a sample where prompt-side image placeholders ({num_prompt_placeholders}) "
            f"!= total images ({len(images)}) -- likely an image placeholder landed in the final "
            f"(completion) turn. See this function's docstring/comment for why this is unsafe."
        )
        return None, None, []
    return prompt, completion, images


_logged_no_op_example = False


def _build_full_trajectory(
    conversations: List[Dict],
    image_bytes_list: List[bytes],
    keep_no_op_p: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[List[Dict]], Optional[List[bytes]]]:
    """One parquet trajectory row -> full (messages, images) for multi-step loss.

    Mirrors `_split_prompt_completion_with_images` (same content normalization +
    image validation) but keeps the WHOLE conversation: no prompt/completion split,
    so every assistant "Action: ..." turn stays a training target for
    `_MultiStepVLMCollator`. `images` carries the RAW encoded bytes (see the module
    docstring's image contract).

    `keep_no_op_p < 1.0` additionally deletes no-op (observation, action) turn pairs
    outright -- OpenHA's `keep_no_op_p` applied at the data level; see
    `_no_op_dropped_turns`. Dropped images are never validated (only their index is
    consumed), so this also makes the trajectory cheaper to preprocess AND shorter in
    vision tokens.
    """
    global _logged_no_op_example

    if not conversations or len(conversations) < 2:
        return None, None
    if conversations[0]["role"] != "user":
        conversations = conversations[1:]
    if not conversations or conversations[-1]["role"] != "assistant":
        return None, None

    dropped = _no_op_dropped_turns(conversations, keep_no_op_p, rng or random)

    messages, images, image_idx = [], [], 0
    for turn_idx, conv in enumerate(conversations):
        keep_turn = turn_idx not in dropped
        content_list = []
        for item in conv["content"]:
            if item.get("type") == "text":
                if keep_turn:
                    content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                # `image_idx` walks `image_bytes_list` in encounter order and must advance
                # for dropped turns too, or every later turn would be paired with the
                # wrong frame's image.
                if keep_turn and image_idx < len(image_bytes_list):
                    try:
                        Image.open(io.BytesIO(image_bytes_list[image_idx])).convert("RGB")  # validate only
                        content_list.append({"type": "image"})
                        images.append(image_bytes_list[image_idx])
                    except Exception as e:
                        logger.warning(f"Failed to decode image at idx {image_idx}: {e}")
                        content_list.append({"type": "text", "text": "[image]"})
                image_idx += 1
        if keep_turn:
            messages.append({"role": conv["role"], "content": content_list})

    # Unreachable in practice (the instruction-bearing first turn and its action are never
    # droppable, see `_no_op_dropped_turns`), but an empty/assistant-less trajectory would
    # produce an all -100 label row and a NaN loss, so fail loudly-but-safely instead.
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        logger.warning(
            "keep_no_op_p dropping left a trajectory with no usable assistant turn "
            f"({len(conversations)} turns in, {len(messages)} out); dropping the sample."
        )
        return None, None

    if dropped and not _logged_no_op_example:
        _logged_no_op_example = True
        logger.info(
            f"keep_no_op_p={keep_no_op_p}: first affected trajectory went from "
            f"{len(conversations)} to {len(messages)} turns "
            f"({len(dropped) // 2} no-op (observation, action) pairs dropped, "
            f"{len(images)} images kept)."
        )
    return messages, images


def _read_bytes(uri: str) -> bytes:
    """Read raw bytes from a local path or any `fsspec`-supported URI (e.g. `s3://...`,
    via the `s3fs` dependency)."""
    import fsspec

    with fsspec.open(uri, "rb") as f:
        return f.read()


def build_messages_qa(
    conversations: list,
    image_paths: list,
    image_root: str,
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[List[Image.Image]]]:
    """
    Convert one JSONL "flat QA" row (e.g. `minecraft-vlp/*.jsonl`) into TRL's
    prompt/completion/images format: these rows are short, independent multi-turn
    Q&A sessions about a small, fixed set of images (declared once in the row's
    "image" field, referenced by whichever `{"type": "image"}` placeholder(s) appear
    anywhere in `conversations`, in encounter order), so the whole conversation is
    kept as-is (no history-window truncation needed). Keeping the whole conversation
    also guarantees any image placeholder (almost always in the first user turn)
    always stays inside the resulting `prompt` instead of possibly being sliced away.

    Args:
        conversations: list of {role, content[{type, text/image}]}.
        image_paths: list of paths *relative to `image_root`* (as stored in the row's
            "image" field), one entry per `{"type": "image"}` placeholder in encounter
            order across the whole conversation.
        image_root: directory (local path or any `fsspec` URI, e.g.
            "s3://bucket/prefix") that `image_paths` entries are relative to -- normally
            the directory containing the jsonl file itself; see
            `build_minecraft_dataset`/`_default_image_root`.
    """
    if not conversations or len(conversations) < 2:
        return None, None, None

    if conversations[0]["role"] != "user":
        conversations = conversations[1:]
    if not conversations or conversations[-1]["role"] != "assistant":
        return None, None, None

    image_bytes_list: List[bytes] = []
    for rel_path in image_paths or []:
        uri = f"{image_root.rstrip('/')}/{str(rel_path).lstrip('/')}"
        try:
            image_bytes_list.append(_read_bytes(uri))
        except Exception as e:
            logger.warning(f"Failed to read image {uri}: {e}")
            # Keep a (deliberately undecodable) placeholder so the flat index stays in
            # sync with the placeholders in `conversations`; it will simply fail to
            # decode below and fall back to a "[image]" text placeholder.
            image_bytes_list.append(b"")

    return _split_prompt_completion_with_images(conversations, image_bytes_list)


def build_messages_text_only(conversations: list) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """
    Convert one pure-text row (e.g. `minecraft-vlp/mc-qa-*.jsonl`'s
    `label=["qa","wiki","self-instruct"]` rows: a leading `system` turn + a `user`
    question + an `assistant` answer, `image=[]`) into TRL prompt/completion format,
    for JARVIS-VLA's Stage I ("Minecraft world knowledge" text-only post-training --
    see `--text_only`/`--freeze_vision_tower`).

    Unlike `build_messages_qa`, this does NOT strip a leading non-"user" turn: that
    stripping exists there to drop a stray artifact turn seen in trajectory/VQA
    preprocessing, but here a leading turn is normally a legitimate system prompt
    ("You are a helpful assistant...") that must be preserved inside `prompt`, not
    discarded.

    Returns `(prompt, completion)` -- no `images` list, since this data format never
    has any (see `_row_to_trl_sample`/`build_minecraft_dataset` for why the caller must
    omit the "images" key entirely from the resulting sample dict rather than passing
    an empty list, so `SFTTrainer` picks its plain-text collator).

    Any `{"type": "image"}` placeholder encountered is unexpected for this data format
    and gets replaced with a "[image]" text stand-in (with a warning) rather than
    silently producing an inconsistent sample -- if you see that warning, the file
    you're pointing --text_only at probably isn't actually text-only.
    """
    if not conversations or len(conversations) < 2:
        return None, None
    if conversations[-1]["role"] != "assistant":
        return None, None

    messages = []
    for conv in conversations:
        content_list = []
        for item in conv["content"]:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                logger.warning(
                    "build_messages_text_only: found an image placeholder in a "
                    "--text_only row; replacing with a '[image]' text stand-in. This "
                    "data file may not actually be text-only -- consider --text_only=False."
                )
                content_list.append({"type": "text", "text": "[image]"})
        messages.append({"role": conv["role"], "content": content_list})

    prompt, completion = messages[:-1], [messages[-1]]
    return prompt, completion


# TRL's `SFTTrainer` checks that tokenized `prompt` is a token-for-token PREFIX of
# tokenized `prompt+completion` to compute `completion_mask`. For Qwen3.x "thinking"
# chat templates, leaving `enable_thinking` unset breaks that: `add_generation_prompt=
# True` alone emits an unclosed `<think>\n`, while the full render's assistant turn
# emits a closed `<think>\n\n</think>\n\n` -- character-prefix but NOT token-prefix
# (verified: the tokenizer merges that boundary differently), so 2 boilerplate tokens
# leak into the loss on every sample. `enable_thinking=False` makes both renderings
# emit the same closed form, eliminating the mismatch.
#
# Only applied (see `_resolve_chat_template_kwargs`) when the processor's template
# actually references `enable_thinking` (Qwen3.x): passing it to a template that
# doesn't (Qwen2-VL/Qwen2.5-VL) renders identically but trips a per-sample
# `transformers` "kwargs not in processor_kwargs" warning -- avoided by gating on it.
#
# NOTE: the resolved kwargs are applied at COLLATE time (injected into each example by
# `collators.py`), NOT stored as a per-row dataset column. A dict-typed column makes
# Arrow's type inference unstable across map shards (empty dict -> Json, non-empty ->
# struct; datasets 5.x additionally has a built-in expected type for a
# "chat_template_kwargs" column), which hard-crashes the non-streaming map with
# "features can't be aligned" -- and the value is a dataset-level constant anyway
# (depends only on the processor), so per-row storage buys nothing.
_CHAT_TEMPLATE_KWARGS: Dict = {"enable_thinking": False}


def _resolve_chat_template_kwargs(processor) -> Dict:
    """Return `_CHAT_TEMPLATE_KWARGS` only if `processor`'s own chat template actually
    references `enable_thinking` (Qwen3.x "thinking mode" templates); otherwise `{}`.
    See `_CHAT_TEMPLATE_KWARGS`'s comment above for why this avoids a per-sample
    `transformers` log-spam warning on models (e.g. Qwen2-VL/Qwen2.5-VL) whose template
    doesn't use it. Falls back to the unconditional dict if `processor` is `None`
    (callers that don't have one yet can't detect support either way).
    """
    if processor is None:
        return _CHAT_TEMPLATE_KWARGS
    template = getattr(processor, "chat_template", None)
    if isinstance(template, dict):
        template = "\n".join(v for v in template.values() if isinstance(v, str))
    if isinstance(template, str) and "enable_thinking" in template:
        return _CHAT_TEMPLATE_KWARGS
    return {}


def _row_to_trl_sample(
    sample: Dict,
    idx: int,
    data_format: str,
    image_root: Optional[str],
    text_only: bool = False,
    processor=None,
    max_seq_length: Optional[int] = None,
    full_trajectory: bool = False,
    keep_no_op_p: float = 1.0,
    no_op_seed: int = 0,
) -> Dict:
    """
    Map ONE raw dataset row (parquet trajectory step OR jsonl QA session, per
    `data_format`) to a TRL prompt/completion/images sample.

    Used as the `function` arg of `datasets.Dataset.map`/`IterableDataset.map` (with
    `with_indices=True`), not a hand-rolled `torch.utils.data.IterableDataset`:
    `SFTTrainer.__init__` requires `train_dataset` to `isinstance`-check against
    `datasets.Dataset`/`IterableDataset`, and chaining `.map()` on `load_dataset(...)`'s
    return value preserves that (plus SFTTrainer/Accelerate's sharding).

    `text_only=True` (Stage I) bypasses `build_messages_qa` for
    `build_messages_text_only`, and the returned dict has NO "images" key at all --
    `SFTTrainer` picks its vision collator purely based on whether that key is
    *present*, so omitting it (not just leaving it empty) is what keeps Stage I on the
    plain-text path.

    Invalid rows get `"_keep": False` (a `.map()` fn must return one row per input row);
    the caller chains `.filter(lambda x: x["_keep"])` to actually drop them.

    If `processor`/`max_seq_length` are given, image-bearing samples are also
    length-checked (`_exceeds_max_length`) -- otherwise `SFTConfig`'s raw-token-level
    truncation can land inside an image's placeholder-token block, and the VLM forward
    pass crashes with a tokens/features mismatch (observed for real on Qwen2-VL-7B).

    `keep_no_op_p` (`full_trajectory` only) drops no-op frames at the data level; its RNG
    is seeded per row from `(no_op_seed, idx)` so the decision is reproducible for a given
    seed and independent of the global `random` stream (which `datasets`' multiprocess
    `.map` workers would otherwise make nondeterministic).
    """
    chat_template_kwargs = _resolve_chat_template_kwargs(processor)
    if text_only:
        prompt, completion = build_messages_text_only(conversations=sample["conversations"])
        if prompt is None:
            return {"prompt": [], "completion": [], "_keep": False}
        return {
            "prompt": prompt,
            "completion": completion,
            "_keep": True,
        }

    if full_trajectory:
        # Stage III multi-step loss: keep the whole parquet trajectory as a flat
        # `messages` list (not prompt/completion) so `_MultiStepVLMCollator` trains on
        # every assistant turn. Oversized trajectories are dropped (same rationale as
        # the prompt/completion path below: truncation through a vision-token block
        # crashes the forward pass).
        messages, images = _build_full_trajectory(
            sample["conversations"],
            sample.get("image_bytes", []),
            keep_no_op_p=keep_no_op_p,
            rng=random.Random(f"{no_op_seed}:{idx}"),
        )
        if messages is None:
            return {"messages": [], "images": [], "_keep": False}
        if images and processor is not None and max_seq_length is not None:
            # `_exceeds_max_length` needs PIL sizes; decode transiently, emit raw bytes.
            pil_images = _decode_images(images)
            if _exceeds_max_length(processor, messages[:-1], [messages[-1]], pil_images, max_seq_length, chat_template_kwargs):
                return {"messages": [], "images": [], "_keep": False}
        return {"messages": messages, "images": images, "_keep": True}

    if data_format == "jsonl":
        prompt, completion, images = build_messages_qa(
            conversations=sample["conversations"],
            image_paths=sample.get("image", []),
            image_root=image_root,
        )
        if prompt is None:
            return {"prompt": [], "completion": [], "images": [], "_keep": False}
        if images and processor is not None and max_seq_length is not None:
            # `_exceeds_max_length` needs PIL sizes; decode transiently, emit raw bytes.
            pil_images = _decode_images(images)
            if _exceeds_max_length(processor, prompt, completion, pil_images, max_seq_length, chat_template_kwargs):
                return {"prompt": [], "completion": [], "images": [], "_keep": False}
    else:
        # data_format == "parquet" with full_trajectory=False. `build_minecraft_dataset`
        # rejects this combination before ever calling `.map()` (see its own validation),
        # because the single-step random-history-window sampler that used to run here
        # (`build_messages`) caused real Stage III checkpoints to collapse to always
        # predicting `move(0,0)` (~99.5% of camera outputs): long runs of identical
        # "walk straight" actions numerically dominated the loss when each training
        # sample only targets ONE action. It was replaced by `--full_trajectory` (see
        # `_build_full_trajectory` above), which trains on every assistant turn in a
        # trajectory per sample instead of a single randomly-sampled one. This branch is
        # therefore unreachable in practice; the assertion exists only as a defensive
        # backstop in case a future caller invokes `_row_to_trl_sample` directly.
        raise AssertionError(
            "data_format='parquet' requires full_trajectory=True; "
            "build_minecraft_dataset should have validated this before calling .map()."
        )
    return {
        "prompt": prompt,
        "completion": completion,
        "images": images,
        "_keep": True,
    }


def _fast_encoded_length(
    processor,
    rendered: str,
    images: List["Image.Image"],
) -> int:
    """Compute the exact token count `processor(text=[rendered], images=images)` would
    produce, without paying for the actual (expensive) pixel resize/rescale/normalize/
    patchify work -- only each image's (height, width) is needed.

    Uses `Qwen2VLProcessor`/`Qwen3VLProcessor`'s own `_get_num_multimodal_tokens(...)`
    (pure arithmetic on image size + patch_size/merge_size/min_pixels/max_pixels) for
    each image's vision-token count, combined with a plain-text tokenization of
    `rendered` (pre-expansion, one placeholder token per image) to reconstruct the
    post-expansion length: `base_len - num_images + sum(num_image_tokens_per_image)`.
    Verified byte-exact against real `processor(...)` output for 1-5 images across a
    range of sizes. Raises if `processor` lacks `_get_num_multimodal_tokens` (caller
    falls back to the exact-but-slow path).
    """
    raw_ids = processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    base_len = len(raw_ids)
    if not images:
        return base_len
    image_sizes = [(im.height, im.width) for im in images]
    num_tokens_per_image = processor._get_num_multimodal_tokens(image_sizes=image_sizes)["num_image_tokens"]
    return base_len - len(images) + sum(num_tokens_per_image)


def _exceeds_max_length(
    processor,
    prompt: List[Dict],
    completion: List[Dict],
    images: List["Image.Image"],
    max_seq_length: int,
    chat_template_kwargs: Optional[Dict] = None,
) -> bool:
    """Compute `prompt+completion+images`'s token count under the REAL processor/
    chat-template about to be used, and check whether it overflows `max_seq_length`.

    Pre-filters samples that would otherwise crash training mid-run: an overflowing
    VLM sample risks the truncation point landing inside an image's placeholder-token
    block, which crashes the forward pass with a tokens/features mismatch (unlike
    plain-text overflow, which the collator truncates away harmlessly). Any overflow on
    an image-bearing sample is therefore treated as unsafe and dropped.

    Tries the cheap `_fast_encoded_length` path first (avoids redoing the same
    expensive image preprocessing the collator is about to do for real -- this was
    silently capping GPU utilization around 30%), falling back to the exact
    `processor(...)` call if that raises. Any error from both paths drops the sample
    defensively rather than propagating out of a `.map()` call.
    """
    try:
        rendered = processor.apply_chat_template(
            list(prompt) + list(completion),
            tokenize=False,
            add_generation_prompt=False,
            **(chat_template_kwargs or {}),
        )
    except Exception as e:
        logger.warning(f"Length pre-check's apply_chat_template failed for a sample ({e!r}); dropping it defensively.")
        return True

    try:
        total_len = _fast_encoded_length(processor, rendered, images)
    except Exception:
        try:
            encoded = processor(text=[rendered], images=images, return_tensors=None)
            total_len = len(encoded["input_ids"][0])
        except Exception as e:
            logger.warning(f"Length pre-check failed for a sample ({e!r}); dropping it defensively.")
            return True
    return total_len > max_seq_length


def _detect_data_format(data_path: Union[str, List[str]]) -> str:
    """Infer "parquet" vs "jsonl" from the file extension in `data_path` (which may be a
    glob, e.g. "s3://.../train-*.parquet" or "s3://.../*.jsonl", or (see
    `build_minecraft_dataset`) a list of several such globs/paths -- in that case every
    entry is checked and detection fails loudly on a mix of extensions rather than
    silently guessing)."""
    paths = data_path if isinstance(data_path, list) else [data_path]
    formats = {"jsonl" if (".jsonl" in p.lower() or ".json" in p.lower()) else "parquet" for p in paths}
    if len(formats) > 1:
        raise ValueError(f"--data_path mixes parquet and jsonl extensions ({paths!r}); pass --data_format explicitly.")
    return formats.pop()


def _default_image_root(data_path: Union[str, List[str]]) -> str:
    """Directory containing `data_path`'s file(s) -- e.g. for
    "s3://bucket/minecraft-vlp/mc-vqa-241102.jsonl" (or the glob
    "s3://bucket/minecraft-vlp/*.jsonl") this is "s3://bucket/minecraft-vlp". That is
    also where `minecraft-vlp`-style datasets keep their `images/` subdirectory, which
    is what each row's "image" (relative-path) field is rooted at.

    When `data_path` is a list of several files (e.g. combining VQA + Caption +
    Grounding jsonls for JARVIS-VLA Stage II), this uses the FIRST entry's directory --
    only correct if every file lives directly alongside the others under the same
    `<root>/images/...` layout (true for all of `minecraft-vlp`'s files); pass
    `--image_root` explicitly if that doesn't hold."""
    first = data_path[0] if isinstance(data_path, list) else data_path
    return first.rsplit("/", 1)[0]


# Columns actually read anywhere downstream (`_row_to_trl_sample` / `build_messages_qa` /
# `build_messages_text_only` / `_build_full_trajectory`): the conversation itself and
# the image path list. Everything else (`id`, `label`, `model`, `datetime`, `source`,
# ...) is pure metadata never touched by training code.
_USED_COLUMNS = {"conversations", "image"}


def _load_dataset_multi(
    builder_name: str,
    data_path: Union[str, List[str]],
    streaming: bool,
    cache_dir: Optional[str] = None,
):
    """`load_dataset(builder_name, data_files=data_path, split="train", streaming=...)`,
    except when `data_path` is a `list`: those files are loaded and trimmed to
    `_USED_COLUMNS` ONE AT A TIME then stitched with `concatenate_datasets`, instead of
    a single `load_dataset(..., data_files=[...])` call.

    Needed because `minecraft-vlp`'s jsonl files (combined for JARVIS-VLA Stage II)
    have wildly different schemas for columns training never uses (e.g. `source`'s
    struct shape differs per file, one file even has a typo'd `datatime` key). A single
    combined `load_dataset` call tries to unify ALL files into one Arrow schema up
    front and hard-crashes (`TypeError: Couldn't cast array...`) the moment it hits a
    disagreeing file; loading each file separately never exposes pyarrow to more than
    one schema at a time, sidestepping this entirely (verified against the real 5-file
    Stage II combination).

    `cache_dir` (non-streaming only) is where the materialized Arrow cache is written
    -- point it at a big node-local SSD for the ~170GB stage3 parquet dataset (the
    datasets default `~/.cache/huggingface` usually lives on a much smaller disk).
    """
    if not isinstance(data_path, list):
        return load_dataset(builder_name, data_files=data_path, split="train", streaming=streaming, cache_dir=cache_dir)

    per_file = []
    for path in data_path:
        ds = load_dataset(builder_name, data_files=path, split="train", streaming=streaming, cache_dir=cache_dir)
        drop = [c for c in ds.column_names if c not in _USED_COLUMNS]
        if drop:
            ds = ds.remove_columns(drop)
        per_file.append(ds)
    return concatenate_datasets(per_file)


def build_minecraft_dataset(
    data_path: Union[str, List[str]],
    streaming: bool = False,
    data_format: str = "auto",
    image_root: Optional[str] = None,
    text_only: bool = False,
    processor=None,
    max_seq_length: Optional[int] = None,
    full_trajectory: bool = False,
    keep_no_op_p: float = 1.0,
    no_op_seed: int = 0,
    num_proc: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """
    Build the Minecraft SFT dataset as a genuine `datasets.Dataset` (`streaming=False`)
    or `datasets.IterableDataset` (`streaming=True`). Samples are either
    `{"prompt": [...], "completion": [...], "images": [...]}` (default -- lets
    `SFTConfig(completion_only_loss=True)` mask context out of the loss) or, with
    `full_trajectory=True`, `{"messages": [...], "images": [...]}` for
    `_MultiStepVLMCollator`'s per-step masking instead.

    Supports two on-disk layouts, auto-detected from `data_path`'s extension (override
    with `data_format=`):
      - "parquet" (e.g. `minecraft-text-action-dataset`): each row is one trajectory
        (`conversations` + one `image_bytes` entry per turn pair). REQUIRES
        `full_trajectory=True` (see below) -- `_build_full_trajectory` keeps the whole
        trajectory as the training sample.
      - "jsonl" (e.g. `minecraft-vlp`): each row is a short, independent Q&A session
        with an `image` field listing path(s) relative to `image_root` (default: the
        directory containing the jsonl file). `build_messages_qa` loads those on-the-fly
        via `fsspec` and keeps the whole (already-short) conversation.

    `full_trajectory=False` with `data_format="parquet"` raises `ValueError`: an earlier
    single-step random-history-window sampler (`build_messages`, since removed) trained
    each sample on only ONE randomly-chosen action per trajectory, and real Stage III
    checkpoints trained that way collapsed to always predicting `move(0,0)` (~99.5% of
    camera outputs) because long runs of identical "walk straight" actions numerically
    dominated the loss. `--full_trajectory` (training on every assistant turn in a
    trajectory, not just one) replaced it and is now the only supported parquet path.

    `text_only=True` (Stage I, no images): routes every row through
    `build_messages_text_only` instead, and the resulting samples carry NO "images" key
    at all -- `SFTTrainer` picks its vision collator purely based on whether that key
    is *present*. Combine with `freeze_vision_tower(model)` to match JARVIS-VLA's Stage
    I recipe (ViT+adapter frozen, only LLM trained).

    IMPORTANT: returns the result of chaining `.map()`/`.filter()`/`.remove_columns()`
    directly on `datasets.load_dataset(...)`'s return value -- NOT a hand-rolled
    `torch.utils.data.IterableDataset` (which `SFTTrainer` rejects with `TypeError` at
    construction time). The dataset is deliberately returned FULL and UNSHARDED:
    SFTTrainer/Accelerate performs the one and only process-level shard while preparing
    the DataLoader; calling `split_dataset_by_node` here would divide each stream by
    world_size a second time (deterministic early exhaustion + mismatched collectives).

    `processor`/`max_seq_length`: when both given, image-bearing samples whose REAL
    tokenized length would overflow `max_seq_length` are dropped instead of silently
    truncated (see `_exceeds_max_length` for why blind truncation crashes VLM training).
    Pass the same `AutoProcessor` used for the model. Omit both to skip this check
    (e.g. for `--text_only` data, which has no images and thus no risk of this crash).

    `keep_no_op_p` (default 1.0 = disabled, requires `full_trajectory=True`): probability
    of KEEPING each pure-no-op frame. This is OpenHA's `keep_no_op_p` knob applied at the
    DATA level -- a dropped frame's observation image and action both disappear from the
    trajectory, so they contribute neither loss nor context (contrast
    `MultiStepVLMCollator(focal_decay=...)`, which keeps the original distribution and only
    rebalances its gradient contribution). The two mechanisms are independent and compose:
    OpenHA itself ships both, using data-level dropping for its `MotionTokenizer` route
    (`keep_no_op_p=0` by default) and loss-level focal masking for the text-action route
    (`keep_no_op_p=1.0` by default). Measured on `minecraft-text-action-dataset`, 24.8% of
    assistant steps are pure no-ops, so e.g. `keep_no_op_p=0.2` removes ~20% of all steps.

    `num_proc` (non-streaming only): worker processes for the one-time `.map()` cache
    build. Falls back to single-process automatically if the map fn / `fn_kwargs` fail to
    pickle. `cache_dir` (non-streaming only): where the materialized Arrow cache is
    written -- for the ~170GB stage3 parquet dataset this MUST point at a big node-local
    SSD (e.g. /local-ssd/hf_datasets_cache), not datasets' default ~/.cache/huggingface.
    """
    if data_format == "auto":
        data_format = _detect_data_format(data_path)
    if data_format not in ("parquet", "jsonl"):
        raise ValueError(f"Unknown data_format: {data_format!r} (expected 'parquet', 'jsonl', or 'auto')")
    if text_only and data_format == "parquet":
        logger.warning(
            "--text_only was set together with a parquet (trajectory) data_format; "
            "this is unusual -- --text_only is designed for Stage I text-QA jsonl rows "
            "(e.g. minecraft-vlp/mc-qa-*.jsonl). Any image_bytes on these rows will be "
            "ignored."
        )
    if data_format == "parquet" and not text_only and not full_trajectory:
        raise ValueError(
            "data_format='parquet' requires --full_trajectory. The old single-step "
            "random-history-window sampler was removed after it caused real Stage III "
            "checkpoints to collapse to always predicting move(0,0) (long runs of "
            "identical actions dominated the loss when each sample only targets one "
            "action). Pass --full_trajectory to train on every assistant turn instead."
        )
    if not 0.0 <= keep_no_op_p <= 1.0:
        raise ValueError(f"keep_no_op_p must be in [0.0, 1.0], got {keep_no_op_p}")
    if keep_no_op_p < 1.0 and not full_trajectory:
        # Only `_build_full_trajectory` can drop a frame: the prompt/completion paths
        # produce ONE target action per sample, so "dropping the no-op frame" there means
        # dropping the whole sample, which is a different (and untested) operation.
        # Raise rather than silently ignore the flag -- a run that looks like it filtered
        # no-ops but didn't is the worst possible outcome for an A/B comparison.
        raise ValueError(
            f"keep_no_op_p={keep_no_op_p} requires full_trajectory=True (it drops no-op "
            "(observation, action) turn pairs inside a trajectory); it has no meaning for "
            "the single-target prompt/completion data paths."
        )
    if data_format == "jsonl" and image_root is None:
        image_root = _default_image_root(data_path)

    builder_name = "parquet" if data_format == "parquet" else "json"
    if streaming and num_proc:
        logger.info("num_proc is not supported by IterableDataset.map(); ignoring it in streaming mode.")
    if streaming:
        dataset = _load_dataset_multi(builder_name, data_path, streaming=True, cache_dir=cache_dir)
        # Return the complete HF IterableDataset. SFTTrainer/Accelerate shards it once
        # while preparing the DataLoader. Pre-sharding here caused double sharding:
        # N/world_size^2 samples per process and deterministic early stream exhaustion.
        logger.info(
            f"Dataset loaded in streaming mode (format={data_format}, length unknown ahead of time); "
            "process sharding is delegated to SFTTrainer/Accelerate"
        )
    else:
        dataset = _load_dataset_multi(builder_name, data_path, streaming=False, cache_dir=cache_dir)
        logger.info(f"Dataset loaded (format={data_format}): {len(dataset)} samples")

    raw_columns = dataset.column_names
    map_kwargs = dict(
        function=_row_to_trl_sample,
        with_indices=True,
        fn_kwargs={
            "data_format": data_format,
            "image_root": image_root,
            "text_only": text_only,
            "processor": processor,
            "max_seq_length": max_seq_length,
            "full_trajectory": full_trajectory,
            "keep_no_op_p": keep_no_op_p,
            "no_op_seed": no_op_seed,
        },
        remove_columns=raw_columns,
    )
    if not streaming:
        # `desc` (progress bar label) and `num_proc` exist only on `Dataset.map`,
        # NOT on `IterableDataset.map` -- passing them in streaming mode raises.
        map_kwargs["desc"] = "trl_sft map (decode/normalize/length-check)"
        if num_proc and num_proc > 1:
            map_kwargs["num_proc"] = num_proc
    try:
        dataset = dataset.map(**map_kwargs)
    except Exception as e:
        # Most likely cause: `fn_kwargs`' `processor` (or the map fn closure) failing
        # to pickle across `num_proc` worker processes on some transformers/processor
        # combinations. Fall back to single-process map rather than dying -- slower
        # (the one-time cache build pays the whole cost serially) but correct.
        if map_kwargs.pop("num_proc", None) is not None:
            logger.warning(f"parallel .map(num_proc={num_proc}) failed ({e!r}); retrying single-process")
            dataset = dataset.map(**map_kwargs)
        else:
            raise
    dataset = dataset.filter(_keep_row)
    dataset = dataset.remove_columns(["_keep"])
    return dataset


def _keep_row(example: Dict) -> bool:
    """Filter predicate for `build_minecraft_dataset`'s `.filter()` -- a named function
    (not a lambda) so it stays picklable for datasets' internal hashing."""
    return bool(example["_keep"])

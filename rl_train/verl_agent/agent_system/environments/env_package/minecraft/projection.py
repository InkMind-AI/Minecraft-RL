"""模型文本输出 → MineStudio env 动作（verl-agent projection 接口）。

09-28 重写。原版两个问题：
  1. `TextActionTokenizer.decode()` 在文本里找不到 `Action:` 时**不抛异常**，而是
     静默返回一个空动作（见 action_mapping.py decode：`if not env_actions:
     env_actions = [null_action]`）。原 projection 靠 try/except 判定合法性，于是
     任何垃圾输出都被记为 valid=1 —— RL 永远不会因格式错误受罚。iter1 自举的
     格式崩溃（100% 触顶 max_tokens、丢失动作语法）正是这类信号缺失的后果。
  2. 模块顶层 import torch / MinecraftSim，以及一段 qwen2_vl 保留 token 的
     死代码（get_sim/raw_str2env，全仓无调用方）——拖慢导入且与本模型无关。

合法性判定与 OpenHA 解码路径一致（同一个 TextActionTokenizer），且比"有 Action:
前缀"更严：匹配到的动作片段里必须含至少一个可识别原语——move(dx,dy) / press(...)
/ click(left|right) / no_op。否则像 iter1 崩溃样例 `Action: Walk forward across
the terrain` 这种**有前缀、无语法**的自然语言也会被判 valid，然后静默解码成空动作。
"""
from typing import Any, Dict, List, Tuple

from openagents.agents.utils.action_mapping import TextActionTokenizer

_tokenizer = None


def _get_tokenizer() -> TextActionTokenizer:
    global _tokenizer
    if _tokenizer is None:
        # 与 OpenHA text_action 分支相同的默认：act_beg_token="Action:"、chunk_len=1
        _tokenizer = TextActionTokenizer()
    return _tokenizer


def _has_action_primitive(tok: TextActionTokenizer, text: str) -> bool:
    """文本中至少一个 `Action:` 片段含可识别的动作原语。"""
    for seg in tok.action_re.findall(text):
        seg = seg.strip()
        if "no_op" in seg:
            return True
        if tok.camera_re.search(seg) or tok.keyboard_re.search(seg):
            return True
        for part in seg.split(" and "):
            if tok.mouse_click_re.match(part.strip()):
                return True
    return False


def minecraft_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """把一批模型输出解码为 env 动作。

    返回 (projected, valids)：
      projected[i] = {"raw_action": env_action_dict 或 None, "thought": 原始文本}
      valids[i]    = 1 当且仅当文本里有可解析的 `Action:` 片段
    raw_action=None 时 worker 会按 noop 执行（见 envs.py MinecraftWorker.step）。
    输入列表不做原地修改。
    """
    tok = _get_tokenizer()
    projected: List[Dict[str, Any]] = []
    valids: List[int] = []
    for text in actions:
        text = text if isinstance(text, str) else ""
        try:
            if not _has_action_primitive(tok, text):
                projected.append({"raw_action": None, "thought": text})
                valids.append(0)
                continue
            env_action = tok.decode(text)[0]
            projected.append({"raw_action": env_action, "thought": text})
            valids.append(1)
        except Exception as e:  # noqa: BLE001
            print(f"[minecraft_projection] 解码失败: {e!r} | text={text[:120]!r}", flush=True)
            projected.append({"raw_action": None, "thought": text})
            valids.append(0)
    return projected, valids

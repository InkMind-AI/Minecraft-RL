"""可插拔标注器：Stub（链路验证）/ OpenAI（GPT）/ Gemini。

API 到位后通过环境变量注入密钥并选择后端，管线代码零改动：
    COT_ANNOTATOR = stub | openai | gemini
    COT_API_KEY   = <key>            # 对应平台的 API key
    COT_MODEL     = gpt-4o | gemini-2.0-flash | ...（可选，有默认值）
"""

import json
import os
import re
import time
from typing import Dict, List, Any

from .decision_points import parse_action


class StubAnnotator:
    """离线 Stub：用动作语义生成格式正确的占位 Thought（路线 A0 风格）。

    只用于验证链路（检测 -> payload -> 渲染 -> 插入），不代表内容质量。
    """

    name = "stub"

    def annotate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        thoughts = []
        for dp in payload["decision_points"]:
            prev, curr = dp["prev_action"], dp["action"]
            thoughts.append({
                "step": dp["step"],
                "thought": self._render(prev, curr, payload["instruction"]),
            })
        return {"thoughts": thoughts}

    @staticmethod
    def _sem_desc(action: str) -> str:
        a = parse_action(action)
        if a.is_noop:
            return "wait and observe"
        parts = []
        if "w" in a.keys: parts.append("move forward")
        if "a" in a.keys: parts.append("strafe left")
        if "s" in a.keys: parts.append("back up")
        if "d" in a.keys: parts.append("strafe right")
        if "space" in a.keys: parts.append("jump")
        if "e" in a.keys: parts.append("open inventory")
        if a.click == "L": parts.append("attack")
        if a.click == "R": parts.append("use/place")
        if not parts: parts.append("reposition")
        return " and ".join(parts)

    def _render(self, prev: str, curr: str, instruction: str) -> str:
        return (
            f"[stub] continuing task '{instruction[:40]}' | switching from "
            f"'{self._sem_desc(prev)}' to '{self._sem_desc(curr)}'"
        )


class OpenAIAnnotator:
    """OpenAI Chat Completions（GPT 系列，多图输入）。

    支持自定义网关：设置 COT_BASE_URL（OpenAI 兼容代理），默认官方端点。
    """

    name = "openai"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, model: str = None, max_retries: int = 3):
        self.api_key = api_key
        self.model = model or os.environ.get("COT_MODEL", self.DEFAULT_MODEL)
        self.max_retries = max_retries
        self.base_url = (os.environ.get("COT_BASE_URL", "https://api.openai.com/v1")
                         ).rstrip("/")

    def annotate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import urllib.request

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": payload["user_text"]}
        ]
        for img_bytes in payload["images"]:
            b64 = base64.b64encode(img_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        base_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": payload["system_text"]},
                {"role": "user", "content": content},
            ],
        }
        # 渐进降级重试矩阵：先满配置，按 400 错误逐项摘除不支持的参数
        # use_rf: response_format | use_temp: temperature
        for use_rf, use_temp in ((True, True), (True, False), (False, False)):
            body = dict(base_body)
            if use_rf:
                body["response_format"] = {"type": "json_object"}
            if use_temp:
                body["temperature"] = 0.7
            attempt = 0
            while attempt < 8:
                try:
                    req = urllib.request.Request(
                        f"{self.base_url}/chat/completions",
                        data=json.dumps(body).encode(),
                        headers={"Authorization": f"Bearer {self.api_key}",
                                 "Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        out = json.loads(resp.read())
                    msg = out["choices"][0]["message"]
                    # reasoning 模型：正文在 content（reasoning_content 是思考过程）
                    text = msg.get("content") or ""
                    if not text and msg.get("reasoning_content"):
                        text = msg["reasoning_content"]
                    if not text:
                        raise ValueError("空回复")
                    return self._parse_json(text)
                except urllib.error.HTTPError as e:
                    err_body = b""
                    try:
                        err_body = e.read()
                    except Exception:
                        pass
                    if e.code == 400:
                        # 按错误信息定位不支持的字段，降级后重试
                        if use_rf and b"response_format" in err_body:
                            break
                        if use_temp and b"temperature" in err_body:
                            break
                        if attempt >= self.max_retries - 1:
                            raise
                    if e.code == 429:
                        # 限流：长退避（10/20/40/80/160/320s），最多8次
                        wait = min(10 * (2 ** attempt), 320)
                        time.sleep(wait)
                        attempt += 1
                        continue
                    if attempt >= self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
                    attempt += 1
                except Exception:
                    if attempt >= self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
                    attempt += 1
        raise RuntimeError("标注请求失败（含参数降级重试）")

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """容错 JSON 解析（模型可能裹 markdown 代码块）。"""
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
        return json.loads(text)


class GeminiAnnotator:
    """Google Gemini generateContent（多图输入）。

    支持自定义网关：设置 COT_BASE_URL（Google 原生格式的代理地址），默认官方端点。
    """

    name = "gemini"
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str, model: str = None, max_retries: int = 3):
        self.api_key = api_key
        self.model = model or os.environ.get("COT_MODEL", self.DEFAULT_MODEL)
        self.max_retries = max_retries
        self.base_url = (os.environ.get("COT_BASE_URL",
                                        "https://generativelanguage.googleapis.com/v1beta")
                         ).rstrip("/")

    def annotate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import base64
        import urllib.request

        parts: List[Dict[str, Any]] = [{"text": payload["user_text"]}]
        for img_bytes in payload["images"]:
            b64 = base64.b64encode(img_bytes).decode()
            parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": b64},
            })
        body = json.dumps({
            "system_instruction": {"parts": [{"text": payload["system_text"]}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.7,
                "responseMimeType": "application/json",
            },
        }).encode()

        for attempt in range(self.max_retries):
            try:
                url = (f"{self.base_url}/models/"
                       f"{self.model}:generateContent?key={self.api_key}")
                req = urllib.request.Request(
                    url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    out = json.loads(resp.read())
                text = out["candidates"][0]["content"]["parts"][0]["text"]
                return OpenAIAnnotator._parse_json(text)
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def get_annotator() -> Any:
    """按环境变量构造标注器。默认 stub（链路验证模式）。"""
    backend = os.environ.get("COT_ANNOTATOR", "stub").lower()
    if backend == "stub":
        return StubAnnotator()
    key = os.environ.get("COT_API_KEY")
    if not key:
        raise RuntimeError(f"COT_ANNOTATOR={backend} 需要设置 COT_API_KEY")
    if backend == "openai":
        return OpenAIAnnotator(key)
    if backend == "gemini":
        return GeminiAnnotator(key)
    raise ValueError(f"未知标注器后端: {backend}")

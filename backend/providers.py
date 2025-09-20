# backend/providers.py
import asyncio
import json
import logging
from typing import List, Dict, AsyncGenerator, Any
import httpx

from backend.config import (
    get_next_key,
    has_keys,
    OPENAI_FINE_TUNE_MODEL,
    OPENAI_KEYS,
    GEMINI_KEYS,
    OPENROUTER_KEYS,
)

logger = logging.getLogger("bot4code.providers")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

REQUEST_TIMEOUT = 60.0
RETRY_SLEEP = 0.25

def _safe_extract_openai(resp_json: Any) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp_json)

def _safe_extract_gemini(resp_json: Any) -> str:
    # best-effort extraction (Gemini shape varies)
    try:
        if isinstance(resp_json, dict):
            if "candidates" in resp_json and resp_json["candidates"]:
                cand = resp_json["candidates"][0]
                # content.parts common shape
                cont = cand.get("content")
                if isinstance(cont, dict) and "parts" in cont:
                    return "".join([p.get("text", "") for p in cont["parts"]])
                return str(cont)
            if "output" in resp_json:
                return str(resp_json["output"])
    except Exception:
        pass
    return json.dumps(resp_json)

def _safe_extract_openrouter(resp_json: Any) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp_json)

# -------------------------
# OpenAI - non-stream
# -------------------------
async def call_openai_full_messages(messages: List[Dict[str, str]]) -> str:
    if not has_keys("openai"):
        raise RuntimeError("OpenAI keys not configured")
    last_exc = None
    attempts = max(1, len(OPENAI_KEYS))
    for _ in range(attempts):
        key = get_next_key("openai")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_FINE_TUNE_MODEL,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                return _safe_extract_openai(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning("OpenAI status %s, try next key", status)
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception("OpenAI call failed, trying next key")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    raise last_exc or RuntimeError("OpenAI calls failed")

# -------------------------
# OpenAI - streaming (robust)
# -------------------------
async def stream_openai_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """
    Stream tokens from OpenAI. Yields strings (pieces of text).
    Parsing is robust to arbitrary chunk boundaries.
    """
    if not has_keys("openai"):
        raise RuntimeError("OpenAI keys not configured")
    attempts = max(1, len(OPENAI_KEYS))
    last_exc = None
    for _ in range(attempts):
        key = get_next_key("openai")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_FINE_TUNE_MODEL,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    # buffer bytes and split by newline to get "data:" lines safely
                    buffer = ""
                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        buffer += chunk.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            # OpenAI sends lines starting with "data: "
                            if line.startswith("data:"):
                                payload_text = line[len("data:"):].strip()
                            else:
                                payload_text = line
                            if payload_text in ("[DONE]", ""):
                                return
                            try:
                                obj = json.loads(payload_text)
                                choices = obj.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                # ignore partial JSON until complete
                                continue
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning("OpenAI stream HTTP error %s", status)
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception("OpenAI stream error (will try next key if any)")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    raise last_exc or RuntimeError("OpenAI streaming failed")

# -------------------------
# Gemini non-streaming (best effort)
# -------------------------
async def call_gemini_full_messages(messages: List[Dict[str, str]]) -> str:
    if not has_keys("gemini"):
        raise RuntimeError("Gemini keys not configured")
    attempts = max(1, len(GEMINI_KEYS))
    last_exc = None
    for _ in range(attempts):
        key = get_next_key("gemini")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText?key={key}"
        text_prompt = "".join([m.get("content", "") for m in messages])
        payload = {"prompt": {"text": text_prompt}, "temperature": 0.2, "maxOutputTokens": 1024}
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                return _safe_extract_gemini(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning("Gemini HTTP error %s", status)
            last_exc = e
            if status == 429:
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception("Gemini call failed")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    raise last_exc or RuntimeError("Gemini calls failed")

# -------------------------
# OpenRouter non-streaming
# -------------------------
async def call_openrouter_full_messages(messages: List[Dict[str, str]]) -> str:
    if not has_keys("openrouter"):
        raise RuntimeError("OpenRouter keys not configured")
    attempts = max(1, len(OPENROUTER_KEYS))
    last_exc = None
    for _ in range(attempts):
        key = get_next_key("openrouter")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": "openai/gpt-4-turbo-preview", "messages": messages}
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                return _safe_extract_openrouter(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning("OpenRouter HTTP error %s", status)
            last_exc = e
            if status == 429:
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception("OpenRouter call failed")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    raise last_exc or RuntimeError("OpenRouter calls failed")

# -------------------------
# Auto (non-stream)
# -------------------------
async def call_auto_messages(messages: List[Dict[str, str]]) -> str:
    try:
        return await call_openai_full_messages(messages)
    except Exception as e:
        logger.warning("OpenAI non-stream failed: %s", str(e))
    try:
        return await call_gemini_full_messages(messages)
    except Exception as e:
        logger.warning("Gemini non-stream failed: %s", str(e))
    return await call_openrouter_full_messages(messages)

# -------------------------
# Auto stream wrapper - prefers OpenAI streaming
# -------------------------
async def stream_auto_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    try:
        async for token in stream_openai_messages(messages):
            yield token
        return
    except Exception as e:
        logger.warning("OpenAI streaming failed: %s", str(e))

    # fallback to non-stream providers — yield full reply as one chunk
    try:
        reply = await call_gemini_full_messages(messages)
        yield reply
        return
    except Exception as e:
        logger.warning("Gemini fallback failed: %s", str(e))

    reply = await call_openrouter_full_messages(messages)
    yield reply
    return

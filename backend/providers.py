import asyncio
import random
import httpx
import json
import logging
from typing import List, Dict
from backend.config import (
    OPENAI_KEYS,
    GEMINI_KEYS,
    OPENROUTER_KEYS,
    REQUEST_TIMEOUT,
    OPENAI_FINE_TUNE_MODEL,
)

logger = logging.getLogger("bot4code.providers")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

RETRY_SLEEP = 0.5

def has_keys(provider: str) -> bool:
    """Kiểm tra xem có khóa API cho nhà cung cấp không."""
    if provider == "openai":
        return bool(OPENAI_KEYS)
    elif provider == "gemini":
        return bool(GEMINI_KEYS)
    elif provider == "openrouter":
        return bool(OPENROUTER_KEYS)
    return False

def get_next_key(provider: str) -> str:
    """Lấy một khóa API ngẫu nhiên từ danh sách khóa."""
    if provider == "openai":
        return random.choice(OPENAI_KEYS)
    elif provider == "gemini":
        return random.choice(GEMINI_KEYS)
    elif provider == "openrouter":
        return random.choice(OPENROUTER_KEYS)
    return ""

async def call_openai_full_messages(messages: List[Dict[str, str]]) -> str:
    """Gọi API OpenAI để nhận phản hồi đầy đủ."""
    if not has_keys("openai"):
        logger.error("Không có khóa OpenAI được cấu hình")
        raise RuntimeError("Không có khóa OpenAI được cấu hình")
    last_exc = None
    attempts = max(1, len(OPENAI_KEYS))
    for i in range(attempts):
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
            logger.info(f"Thử gọi OpenAI, lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                logger.info("Gọi OpenAI thành công")
                return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning(f"OpenAI trả về trạng thái {status}, thử khóa tiếp theo")
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception(f"Gọi OpenAI thất bại, lần {i+1}/{attempts}")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    logger.error("Tất cả các lần thử OpenAI đều thất bại")
    raise last_exc or RuntimeError("Gọi OpenAI thất bại")

async def stream_openai_messages(messages: List[Dict[str, str]]):
    """Gọi API OpenAI ở chế độ stream."""
    if not has_keys("openai"):
        logger.error("Không có khóa OpenAI được cấu hình")
        raise RuntimeError("Không có khóa OpenAI được cấu hình")
    last_exc = None
    attempts = max(1, len(OPENAI_KEYS))
    for i in range(attempts):
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
            logger.info(f"Thử stream OpenAI, lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_text():
                        if chunk and chunk.strip() != "data: [DONE]":
                            try:
                                data = json.loads(chunk.replace("data: ", ""))
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                    logger.info("Stream OpenAI thành công")
                    return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning(f"OpenAI stream trả về trạng thái {status}, thử khóa tiếp theo")
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception(f"Stream OpenAI thất bại, lần {i+1}/{attempts}")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    logger.error("Tất cả các lần thử stream OpenAI đều thất bại")
    raise last_exc or RuntimeError("Stream OpenAI thất bại")

async def call_gemini_full_messages(messages: List[Dict[str, str]]) -> str:
    """Gọi API Gemini để nhận phản hồi đầy đủ."""
    if not has_keys("gemini"):
        logger.error("Không có khóa Gemini được cấu hình")
        raise RuntimeError("Không có khóa Gemini được cấu hình")
    last_exc = None
    attempts = max(1, len(GEMINI_KEYS))
    for i in range(attempts):
        key = get_next_key("gemini")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": m["content"]} for m in messages]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1500},
        }
        try:
            logger.info(f"Thử gọi Gemini, lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(f"{url}?key={key}", json=payload, headers=headers)
                r.raise_for_status()
                logger.info("Gọi Gemini thành công")
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning(f"Gemini trả về trạng thái {status}, thử khóa tiếp theo")
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception(f"Gọi Gemini thất bại, lần {i+1}/{attempts}")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    logger.error("Tất cả các lần thử Gemini đều thất bại")
    raise last_exc or RuntimeError("Gọi Gemini thất bại")

async def stream_gemini_messages(messages: List[Dict[str, str]]):
    """Gọi API Gemini ở chế độ stream (giả lập vì Gemini không hỗ trợ stream thực sự)."""
    if not has_keys("gemini"):
        logger.error("Không có khóa Gemini được cấu hình")
        raise RuntimeError("Không có khóa Gemini được cấu hình")
    content = await call_gemini_full_messages(messages)
    for i in range(0, len(content), 50):
        yield content[i:i+50]
        await asyncio.sleep(0.05)
    logger.info("Stream Gemini (giả lập) thành công")

async def call_openrouter_full_messages(messages: List[Dict[str, str]]) -> str:
    """Gọi API OpenRouter để nhận phản hồi đầy đủ."""
    if not has_keys("openrouter"):
        logger.error("Không có khóa OpenRouter được cấu hình")
        raise RuntimeError("Không có khóa OpenRouter được cấu hình")
    last_exc = None
    attempts = max(1, len(OPENROUTER_KEYS))
    for i in range(attempts):
        key = get_next_key("openrouter")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1,
        }
        try:
            logger.info(f"Thử gọi OpenRouter, lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                logger.info("Gọi OpenRouter thành công")
                return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning(f"OpenRouter trả về trạng thái {status}, thử khóa tiếp theo")
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception(f"Gọi OpenRouter thất bại, lần {i+1}/{attempts}")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    logger.error("Tất cả các lần thử OpenRouter đều thất bại")
    raise last_exc or RuntimeError("Gọi OpenRouter thất bại")

async def stream_openrouter_messages(messages: List[Dict[str, str]]):
    """Gọi API OpenRouter ở chế độ stream."""
    if not has_keys("openrouter"):
        logger.error("Không có khóa OpenRouter được cấu hình")
        raise RuntimeError("Không có khóa OpenRouter được cấu hình")
    last_exc = None
    attempts = max(1, len(OPENROUTER_KEYS))
    for i in range(attempts):
        key = get_next_key("openrouter")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1,
            "stream": True,
        }
        try:
            logger.info(f"Thử stream OpenRouter, lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_text():
                        if chunk and chunk.strip() != "data: [DONE]":
                            try:
                                data = json.loads(chunk.replace("data: ", ""))
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                    logger.info("Stream OpenRouter thành công")
                    return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning(f"OpenRouter stream trả về trạng thái {status}, thử khóa tiếp theo")
            last_exc = e
            if status in (429, 401, 403):
                await asyncio.sleep(RETRY_SLEEP)
                continue
            raise
        except Exception as e:
            logger.exception(f"Stream OpenRouter thất bại, lần {i+1}/{attempts}")
            last_exc = e
            await asyncio.sleep(RETRY_SLEEP)
            continue
    logger.error("Tất cả các lần thử stream OpenRouter đều thất bại")
    raise last_exc or RuntimeError("Stream OpenRouter thất bại")

async def call_auto_messages(messages: List[Dict[str, str]]) -> str:
    """Tự động chọn nhà cung cấp để gọi API."""
    providers = []
    if has_keys("openai"):
        providers.append(("openai", call_openai_full_messages))
    if has_keys("gemini"):
        providers.append(("gemini", call_gemini_full_messages))
    if has_keys("openrouter"):
        providers.append(("openrouter", call_openrouter_full_messages))
    
    if not providers:
        logger.error("Không có nhà cung cấp nào được cấu hình")
        raise RuntimeError("Không có nhà cung cấp nào được cấu hình")

    random.shuffle(providers)
    last_exc = None
    for provider_name, call_func in providers:
        try:
            logger.info(f"Thử gọi {provider_name}")
            return await call_func(messages)
        except Exception as e:
            logger.warning(f"Nhà cung cấp {provider_name} thất bại: {str(e)}")
            last_exc = e
            continue
    logger.error("Tất cả các nhà cung cấp đều thất bại")
    raise last_exc or RuntimeError("Tất cả các nhà cung cấp đều thất bại")

async def stream_auto_messages(messages: List[Dict[str, str]]):
    """Tự động chọn nhà cung cấp để stream."""
    providers = []
    if has_keys("openai"):
        providers.append(("openai", stream_openai_messages))
    if has_keys("gemini"):
        providers.append(("gemini", stream_gemini_messages))
    if has_keys("openrouter"):
        providers.append(("openrouter", stream_openrouter_messages))
    
    if not providers:
        logger.error("Không có nhà cung cấp nào được cấu hình cho stream")
        raise RuntimeError("Không có nhà cung cấp nào được cấu hình cho stream")

    random.shuffle(providers)
    last_exc = None
    for provider_name, stream_func in providers:
        try:
            logger.info(f"Thử stream {provider_name}")
            async for chunk in stream_func(messages):
                yield chunk
            logger.info(f"Stream {provider_name} thành công")
            return
        except Exception as e:
            logger.warning(f"Stream {provider_name} thất bại: {str(e)}")
            last_exc = e
            continue
    logger.error("Tất cả các nhà cung cấp stream đều thất bại")
    raise last_exc or RuntimeError("Tất cả các nhà cung cấp stream đều thất bại")

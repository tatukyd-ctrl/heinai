# backend/providers.py
import asyncio
import json
import logging
from typing import List, Dict, AsyncGenerator
import httpx
import random

# Import từ config - nếu REQUEST_TIMEOUT chưa định nghĩa, đặt mặc định
try:
    from backend.config import (
        OPENAI_KEYS, GEMINI_KEYS, OPENROUTER_KEYS, OPENAI_FINE_TUNE_MODEL,
        REQUEST_TIMEOUT  # Nếu không có, sẽ dùng mặc định dưới
    )
except ImportError as e:
    logging.error(f"Lỗi import config: {e}")
    OPENAI_KEYS = []
    GEMINI_KEYS = []
    OPENROUTER_KEYS = []
    OPENAI_FINE_TUNE_MODEL = "gpt-4o-mini"
    REQUEST_TIMEOUT = 60.0

# Đặt mặc định nếu không có
REQUEST_TIMEOUT = getattr(globals(), 'REQUEST_TIMEOUT', 60.0)
RETRY_SLEEP = 0.5

logger = logging.getLogger("bot4code.providers")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

def _safe_extract_openai(resp_json: dict) -> str:
    """Trích xuất nội dung từ phản hồi OpenAI."""
    try:
        return resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("Phản hồi OpenAI không đúng định dạng, trả về JSON thô")
        return json.dumps(resp_json)

def _safe_extract_gemini(resp_json: dict) -> str:
    """Trích xuất nội dung từ phản hồi Gemini."""
    try:
        if "candidates" in resp_json and resp_json["candidates"]:
            candidate = resp_json["candidates"][0]
            content = candidate.get("content", {})
            if "parts" in content:
                return "".join(part.get("text", "") for part in content["parts"])
            return content.get("text", str(content))
    except Exception:
        pass
    logger.warning("Phản hồi Gemini không đúng định dạng, trả về JSON thô")
    return json.dumps(resp_json)

def _safe_extract_openrouter(resp_json: dict) -> str:
    """Trích xuất nội dung từ phản hồi OpenRouter."""
    try:
        return resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("Phản hồi OpenRouter không đúng định dạng, trả về JSON thô")
        return json.dumps(resp_json)

def has_keys(provider: str) -> bool:
    """Kiểm tra xem có khóa API cho nhà cung cấp không."""
    p = provider.lower()
    if p == "openai":
        return len(OPENAI_KEYS) > 0
    if p == "gemini":
        return len(GEMINI_KEYS) > 0
    if p == "openrouter":
        return len(OPENROUTER_KEYS) > 0
    return False

def get_next_key(provider: str) -> str:
    """Lấy khóa API tiếp theo (ngẫu nhiên từ danh sách)."""
    p = provider.lower()
    if p == "openai" and OPENAI_KEYS:
        return random.choice(OPENAI_KEYS)
    if p == "gemini" and GEMINI_KEYS:
        return random.choice(GEMINI_KEYS)
    if p == "openrouter" and OPENROUTER_KEYS:
        return random.choice(OPENROUTER_KEYS)
    raise ValueError(f"Không có khóa cho nhà cung cấp {provider}")

# ------------------------- OpenAI Non-Stream -------------------------
async def call_openai_full_messages(messages: List[Dict[str, str]]) -> str:
    if not has_keys("openai"):
        logger.error("Không có khóa OpenAI được cấu hình")
        raise RuntimeError("Không có khóa OpenAI được cấu hình")
    last_exc = None
    attempts = len(OPENAI_KEYS)
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
            logger.info(f"Thử gọi OpenAI (không stream), lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                logger.info("Gọi OpenAI thành công")
                return _safe_extract_openai(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
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

# ------------------------- OpenAI Stream -------------------------
async def stream_openai_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    if not has_keys("openai"):
        logger.error("Không có khóa OpenAI được cấu hình")
        raise RuntimeError("Không có khóa OpenAI được cấu hình")
    last_exc = None
    attempts = len(OPENAI_KEYS)
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
            async with httpx.AsyncClient(timeout=None) as client:  # Không timeout cho stream
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for chunk_bytes in resp.aiter_bytes():
                        if not chunk_bytes:
                            continue
                        chunk = chunk_bytes.decode("utf-8", errors="ignore")
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or line.startswith(":"):  # Bỏ qua comment SSE
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                            else:
                                data_str = line
                            if data_str in ("[DONE]", ""):
                                return
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue  # Bỏ qua JSON không hợp lệ
            logger.info("Stream OpenAI thành công")
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
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

# ------------------------- Gemini Non-Stream (Cập nhật endpoint 2025) -------------------------
async def call_gemini_full_messages(messages: List[Dict[str, str]]) -> str:
    if not has_keys("gemini"):
        logger.error("Không có khóa Gemini được cấu hình")
        raise RuntimeError("Không có khóa Gemini được cấu hình")
    last_exc = None
    attempts = len(GEMINI_KEYS)
    for i in range(attempts):
        key = get_next_key("gemini")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
        headers = {"Content-Type": "application/json"}
        # Chuyển messages thành contents cho Gemini
        contents = [{"parts": [{"text": m["content"]} for m in messages]}]
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1500,
            },
        }
        try:
            logger.info(f"Thử gọi Gemini, lần {i+1}/{attempts}")
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(f"{url}?key={key}", json=payload, headers=headers)
                r.raise_for_status()
                logger.info("Gọi Gemini thành công")
                return _safe_extract_gemini(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
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

# ------------------------- Gemini Stream (Giả lập vì Gemini không hỗ trợ stream thực) -------------------------
async def stream_gemini_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    content = await call_gemini_full_messages(messages)
    chunk_size = 50
    for i in range(0, len(content), chunk_size):
        yield content[i:i + chunk_size]
        await asyncio.sleep(0.05)  # Giả lập stream
    logger.info("Stream Gemini (giả lập) thành công")

# ------------------------- OpenRouter Non-Stream -------------------------
async def call_openrouter_full_messages(messages: List[Dict[str, str]]) -> str:
    if not has_keys("openrouter"):
        logger.error("Không có khóa OpenRouter được cấu hình")
        raise RuntimeError("Không có khóa OpenRouter được cấu hình")
    last_exc = None
    attempts = len(OPENROUTER_KEYS)
    for i in range(attempts):
        key = get_next_key("openrouter")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": "anthropic/claude-3.5-sonnet",  # Model mặc định, có thể thay đổi
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
                return _safe_extract_openrouter(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
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

# ------------------------- OpenRouter Stream -------------------------
async def stream_openrouter_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    if not has_keys("openrouter"):
        logger.error("Không có khóa OpenRouter được cấu hình")
        raise RuntimeError("Không có khóa OpenRouter được cấu hình")
    last_exc = None
    attempts = len(OPENROUTER_KEYS)
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
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for chunk_bytes in resp.aiter_bytes():
                        if not chunk_bytes:
                            continue
                        chunk = chunk_bytes.decode("utf-8", errors="ignore")
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or line.startswith(":"):  # Bỏ qua comment SSE
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                            else:
                                data_str = line
                            if data_str in ("[DONE]", ""):
                                return
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue  # Bỏ qua JSON không hợp lệ
            logger.info("Stream OpenRouter thành công")
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
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

# ------------------------- Auto Non-Stream -------------------------
async def call_auto_messages(messages: List[Dict[str, str]]) -> str:
    """Tự động chọn nhà cung cấp (ưu tiên OpenAI)."""
    try:
        logger.info("Thử gọi OpenAI (ưu tiên)")
        return await call_openai_full_messages(messages)
    except Exception as e:
        logger.warning(f"OpenAI thất bại: {str(e)}")

    try:
        logger.info("Thử gọi Gemini (dự phòng)")
        return await call_gemini_full_messages(messages)
    except Exception as e:
        logger.warning(f"Gemini thất bại: {str(e)}")

    logger.info("Thử gọi OpenRouter (dự phòng cuối)")
    return await call_openrouter_full_messages(messages)

# ------------------------- Auto Stream -------------------------
async def stream_auto_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Tự động chọn nhà cung cấp cho stream (ưu tiên OpenAI)."""
    try:
        logger.info("Thử stream OpenAI (ưu tiên)")
        async for token in stream_openai_messages(messages):
            yield token
        return
    except Exception as e:
        logger.warning(f"OpenAI stream thất bại: {str(e)}")

    try:
        logger.info("Thử stream Gemini (dự phòng)")
        async for token in stream_gemini_messages(messages):
            yield token
        return
    except Exception as e:
        logger.warning(f"Gemini stream thất bại: {str(e)}")

    logger.info("Thử stream OpenRouter (dự phòng cuối)")
    async for token in stream_openrouter_messages(messages):
        yield token
    return

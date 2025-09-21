# backend/providers.py
import asyncio
import json
import logging
import random
import time
from typing import List, Dict, AsyncGenerator, Optional, Any
import httpx
from datetime import datetime, timedelta

from backend.config import config

logger = logging.getLogger("heinbot.providers")

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self):
        self.requests = []
    
    async def wait_if_needed(self, per_minute: int = 60):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove old requests (older than 1 minute)
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= per_minute:
            sleep_time = 60 - (now - self.requests[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limiting: waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

rate_limiter = RateLimiter()

class ProviderError(Exception):
    """Base exception for provider errors"""
    pass

class NoProvidersAvailableError(ProviderError):
    """Raised when no providers are available"""
    pass

class APIKeyError(ProviderError):
    """Raised when API key is invalid or exhausted"""
    pass

def get_random_key(provider: str) -> str:
    """Get a random API key for the specified provider"""
    provider = provider.lower()
    
    if provider == "openai" and config.OPENAI_KEYS:
        return random.choice(config.OPENAI_KEYS)
    elif provider == "gemini" and config.GEMINI_KEYS:
        return random.choice(config.GEMINI_KEYS)
    elif provider == "openrouter" and config.OPENROUTER_KEYS:
        return random.choice(config.OPENROUTER_KEYS)
    elif provider == "anthropic" and config.ANTHROPIC_KEYS:
        return random.choice(config.ANTHROPIC_KEYS)
    
    raise APIKeyError(f"No API keys available for provider: {provider}")

def format_messages_for_provider(messages: List[Dict[str, str]], provider: str) -> Any:
    """Format messages according to provider requirements"""
    provider = provider.lower()
    
    if provider == "gemini":
        # Gemini uses a different message format
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                # Include system message as first user message for Gemini
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {msg['content']}"}]
                })
            elif msg["role"] in ["user", "assistant"]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        return contents
    
    # Standard OpenAI format for other providers
    return messages

async def call_openai(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI API"""
    if not config.has_provider("openai"):
        raise APIKeyError("OpenAI keys not configured")
    
    await rate_limiter.wait_if_needed(config.RATE_LIMIT_PER_MINUTE)
    
    for attempt in range(config.MAX_RETRIES):
        try:
            key = get_random_key("openai")
            
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.OPENAI_MODEL,
                        "messages": messages,
                        "max_tokens": config.DEFAULT_MAX_TOKENS,
                        "temperature": config.DEFAULT_TEMPERATURE,
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                return data["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 403, 429]:
                logger.warning(f"OpenAI API error {e.response.status_code}, attempt {attempt + 1}")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (2 ** attempt))
                    continue
            raise ProviderError(f"OpenAI API error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY)
                continue
            raise ProviderError(f"OpenAI request failed: {e}")
    
    raise ProviderError("OpenAI: All retry attempts failed")

async def stream_openai(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream from OpenAI API"""
    if not config.has_provider("openai"):
        raise APIKeyError("OpenAI keys not configured")
    
    await rate_limiter.wait_if_needed(config.RATE_LIMIT_PER_MINUTE)
    
    for attempt in range(config.MAX_RETRIES):
        try:
            key = get_random_key("openai")
            
            async with httpx.AsyncClient(timeout=config.STREAM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.OPENAI_MODEL,
                        "messages": messages,
                        "max_tokens": config.DEFAULT_MAX_TOKENS,
                        "temperature": config.DEFAULT_TEMPERATURE,
                        "stream": True,
                    }
                ) as response:
                    
                    response.raise_for_status()
                    
                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                            
                        buffer += chunk.decode("utf-8", errors="ignore")
                        
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            
                            if not line or line.startswith(":"):
                                continue
                                
                            if line.startswith("data: "):
                                data_str = line[6:]
                            else:
                                data_str = line
                            
                            if data_str in ["[DONE]", ""]:
                                return
                            
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                
                                if content:
                                    yield content
                                    
                            except json.JSONDecodeError:
                                continue
                    
                    return  # Success
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 403, 429]:
                logger.warning(f"OpenAI stream error {e.response.status_code}, attempt {attempt + 1}")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (2 ** attempt))
                    continue
            raise ProviderError(f"OpenAI stream error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"OpenAI stream failed: {e}")
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY)
                continue
            raise ProviderError(f"OpenAI stream failed: {e}")
    
    raise ProviderError("OpenAI stream: All retry attempts failed")

async def call_anthropic(messages: List[Dict[str, str]]) -> str:
    """Call Anthropic Claude API"""
    if not config.has_provider("anthropic"):
        raise APIKeyError("Anthropic keys not configured")
    
    await rate_limiter.wait_if_needed(config.RATE_LIMIT_PER_MINUTE)
    
    # Convert messages to Anthropic format
    system_message = ""
    anthropic_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    for attempt in range(config.MAX_RETRIES):
        try:
            key = get_random_key("anthropic")
            
            payload = {
                "model": config.ANTHROPIC_MODEL,
                "max_tokens": config.DEFAULT_MAX_TOKENS,
                "temperature": config.DEFAULT_TEMPERATURE,
                "messages": anthropic_messages
            }
            
            if system_message:
                payload["system"] = system_message
            
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01"
                    },
                    json=payload
                )
                
                response.raise_for_status()
                data = response.json()
                
                return data["content"][0]["text"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 403, 429]:
                logger.warning(f"Anthropic API error {e.response.status_code}, attempt {attempt + 1}")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (2 ** attempt))
                    continue
            raise ProviderError(f"Anthropic API error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY)
                continue
            raise ProviderError(f"Anthropic request failed: {e}")
    
    raise ProviderError("Anthropic: All retry attempts failed")

async def call_gemini(messages: List[Dict[str, str]]) -> str:
    """Call Google Gemini API"""
    if not config.has_provider("gemini"):
        raise APIKeyError("Gemini keys not configured")
    
    await rate_limiter.wait_if_needed(config.RATE_LIMIT_PER_MINUTE)
    
    for attempt in range(config.MAX_RETRIES):
        try:
            key = get_random_key("gemini")
            formatted_messages = format_messages_for_provider(messages, "gemini")
            
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": formatted_messages,
                        "generationConfig": {
                            "temperature": config.DEFAULT_TEMPERATURE,
                            "maxOutputTokens": config.DEFAULT_MAX_TOKENS,
                        }
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                return data["candidates"][0]["content"]["parts"][0]["text"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 403, 429]:
                logger.warning(f"Gemini API error {e.response.status_code}, attempt {attempt + 1}")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (2 ** attempt))
                    continue
            raise ProviderError(f"Gemini API error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY)
                continue
            raise ProviderError(f"Gemini request failed: {e}")
    
    raise ProviderError("Gemini: All retry attempts failed")

async def stream_gemini(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream from Gemini API (simulated since Gemini doesn't support native streaming)"""
    content = await call_gemini(messages)
    
    # Simulate streaming by yielding chunks
    chunk_size = 50
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield chunk
        await asyncio.sleep(0.05)  # Small delay to simulate real streaming

async def call_openrouter(messages: List[Dict[str, str]]) -> str:
    """Call OpenRouter API"""
    if not config.has_provider("openrouter"):
        raise APIKeyError("OpenRouter keys not configured")
    
    await rate_limiter.wait_if_needed(config.RATE_LIMIT_PER_MINUTE)
    
    for attempt in range(config.MAX_RETRIES):
        try:
            key = get_random_key("openrouter")
            
            async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://heinbot.local",
                        "X-Title": "HeinBot"
                    },
                    json={
                        "model": "x-ai/grok-4-fast:free",
                        "messages": messages,
                        "max_tokens": config.DEFAULT_MAX_TOKENS,
                        "temperature": config.DEFAULT_TEMPERATURE,
                    }
                )
                
                response.raise_for_status()
                data = response.json()
                
                return data["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [401, 403, 429]:
                logger.warning(f"OpenRouter API error {e.response.status_code}, attempt {attempt + 1}")
                if attempt < config.MAX_RETRIES - 1:
                    await asyncio.sleep(config.RETRY_DELAY * (2 ** attempt))
                    continue
            raise ProviderError(f"OpenRouter API error: {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY)
                continue
            raise ProviderError(f"OpenRouter request failed: {e}")
    
    raise ProviderError("OpenRouter: All retry attempts failed")

# Provider registry for easier management
PROVIDERS = {
    "openai": {"call": call_openai, "stream": stream_openai},
    "anthropic": {"call": call_anthropic, "stream": None},
    "gemini": {"call": call_gemini, "stream": stream_gemini},
    "openrouter": {"call": call_openrouter, "stream": None},
}

async def call_auto_messages(messages: List[Dict[str, str]]) -> str:
    """Auto-select provider and make a call"""
    available_providers = config.get_available_providers()
    
    if not available_providers:
        raise NoProvidersAvailableError("No API providers are configured")
    
    # Try providers in priority order
    priority_order = ["anthropic", "openai", "gemini", "openrouter"]
    providers_to_try = [p for p in priority_order if p in available_providers]
    
    last_error = None
    
    for provider in providers_to_try:
        try:
            logger.info(f"Trying provider: {provider}")
            call_func = PROVIDERS[provider]["call"]
            result = await call_func(messages)
            logger.info(f"Success with provider: {provider}")
            return result
            
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            last_error = e
            continue
    
    # If all providers failed
    if last_error:
        raise last_error
    else:
        raise NoProvidersAvailableError("All providers failed")

async def stream_auto_messages(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Auto-select provider and stream response"""
    available_providers = config.get_available_providers()
    
    if not available_providers:
        raise NoProvidersAvailableError("No API providers are configured")
    
    # Try providers that support streaming first
    streaming_providers = ["openai", "gemini"]
    providers_to_try = [p for p in streaming_providers if p in available_providers]
    
    # Add non-streaming providers as fallback
    non_streaming_providers = ["anthropic", "openrouter"]
    providers_to_try.extend([p for p in non_streaming_providers if p in available_providers])
    
    last_error = None
    
    for provider in providers_to_try:
        try:
            logger.info(f"Trying to stream from provider: {provider}")
            stream_func = PROVIDERS[provider]["stream"]
            
            if stream_func:
                # True streaming
                async for chunk in stream_func(messages):
                    yield chunk
                logger.info(f"Stream success with provider: {provider}")
                return
            else:
                # Fallback to regular call and simulate streaming
                call_func = PROVIDERS[provider]["call"]
                content = await call_func(messages)
                
                # Simulate streaming
                chunk_size = 30
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    yield chunk
                    await asyncio.sleep(0.03)
                
                logger.info(f"Simulated stream success with provider: {provider}")
                return
                
        except Exception as e:
            logger.warning(f"Stream provider {provider} failed: {e}")
            last_error = e
            continue
    
    # If all providers failed
    if last_error:
        raise last_error
    else:
        raise NoProvidersAvailableError("All streaming providers failed")

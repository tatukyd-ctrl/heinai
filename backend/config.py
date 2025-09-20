# backend/config.py
import os, threading
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    load_dotenv()  # fallback to environment

def _split_env(name: str):
    raw = os.getenv(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]

OPENAI_KEYS = _split_env("OPENAI_KEYS")
GEMINI_KEYS = _split_env("GEMINI_KEYS")
OPENROUTER_KEYS = _split_env("OPENROUTER_KEYS")
OPENAI_FINE_TUNE_MODEL = os.getenv("OPENAI_FINE_TUNE_MODEL", "gpt-4o-mini")

# Dropbox config
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", None)
DROPBOX_BASE_FOLDER = os.getenv("DROPBOX_BASE_FOLDER", "/")  # default root of app folder

_lock = threading.Lock()
_openai_idx = 0
_gemini_idx = 0
_openrouter_idx = 0

def has_keys(provider: str) -> bool:
    p = provider.lower()
    if p == "openai":
        return len(OPENAI_KEYS) > 0
    if p == "gemini":
        return len(GEMINI_KEYS) > 0
    if p == "openrouter":
        return len(OPENROUTER_KEYS) > 0
    return False

def get_next_key(provider: str) -> str:
    """
    Round-robin, thread-safe. Raise ValueError if no keys configured.
    """
    global _openai_idx, _gemini_idx, _openrouter_idx
    p = provider.lower()
    with _lock:
        if p == "openai":
            if not OPENAI_KEYS:
                raise ValueError("No OpenAI keys configured (OPENAI_KEYS).")
            key = OPENAI_KEYS[_openai_idx % len(OPENAI_KEYS)]
            _openai_idx += 1
            return key
        if p == "gemini":
            if not GEMINI_KEYS:
                raise ValueError("No Gemini keys configured (GEMINI_KEYS).")
            key = GEMINI_KEYS[_gemini_idx % len(GEMINI_KEYS)]
            _gemini_idx += 1
            return key
        if p == "openrouter":
            if not OPENROUTER_KEYS:
                raise ValueError("No OpenRouter keys configured (OPENROUTER_KEYS).")
            key = OPENROUTER_KEYS[_openrouter_idx % len(OPENROUTER_KEYS)]
            _openrouter_idx += 1
            return key
    raise ValueError(f"Unknown provider: {provider}")

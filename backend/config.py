# backend/config.py
import os
import logging
from typing import List, Optional
from pathlib import Path

# Thiết lập logging với format đẹp hơn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("heinbot.config")

class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Load all configuration from environment variables"""
        
        # Dropbox Configuration
        self.DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "")
        self.DROPBOX_BASE_FOLDER = os.getenv("DROPBOX_BASE_FOLDER", "/heinbot_chats")
        
        # API Keys Configuration
        self.OPENAI_KEYS = self._parse_api_keys("OPENAI_KEYS")
        self.GEMINI_KEYS = self._parse_api_keys("GEMINI_KEYS") 
        self.OPENROUTER_KEYS = self._parse_api_keys("OPENROUTER_KEYS")
        self.ANTHROPIC_KEYS = self._parse_api_keys("ANTHROPIC_KEYS")
        
        # Model Configuration
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
        self.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        self.ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        
        # Request Configuration
        self.REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "90.0"))
        self.STREAM_TIMEOUT = float(os.getenv("STREAM_TIMEOUT", "300.0"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))
        
        # Rate Limiting
        self.RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        self.RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
        
        # Generation Parameters
        self.DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))
        self.DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
        
        # Server Configuration  
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        
        # File Storage
        self.CHAT_STORAGE_DIR = Path(os.getenv("CHAT_STORAGE_DIR", "chats"))
        self.CHAT_STORAGE_DIR.mkdir(exist_ok=True)
        
    def _parse_api_keys(self, env_var: str) -> List[str]:
        """Parse comma-separated API keys from environment variable"""
        keys_str = os.getenv(env_var, "")
        if not keys_str:
            return []
        
        keys = [key.strip() for key in keys_str.split(",") if key.strip()]
        return keys
    
    def _validate_config(self):
        """Validate configuration and log warnings"""
        
        # Check if at least one API provider is configured
        total_keys = len(self.OPENAI_KEYS) + len(self.GEMINI_KEYS) + len(self.OPENROUTER_KEYS) + len(self.ANTHROPIC_KEYS)
        
        if total_keys == 0:
            logger.error("❌ Không có API key nào được cấu hình!")
            raise ValueError("Cần ít nhất một API key để hoạt động")
        
        # Log available providers
        providers = []
        if self.OPENAI_KEYS:
            providers.append(f"OpenAI ({len(self.OPENAI_KEYS)} keys)")
        if self.GEMINI_KEYS:
            providers.append(f"Gemini ({len(self.GEMINI_KEYS)} keys)")
        if self.OPENROUTER_KEYS:
            providers.append(f"OpenRouter ({len(self.OPENROUTER_KEYS)} keys)")
        if self.ANTHROPIC_KEYS:
            providers.append(f"Anthropic ({len(self.ANTHROPIC_KEYS)} keys)")
            
        logger.info(f"🤖 Providers available: {', '.join(providers)}")
        
        # Check Dropbox configuration
        if not self.DROPBOX_ACCESS_TOKEN:
            logger.warning("⚠️  Dropbox access token not configured - file upload disabled")
        else:
            logger.info(f"📁 Dropbox folder: {self.DROPBOX_BASE_FOLDER}")
        
        # Log other settings
        logger.info(f"⚙️  Request timeout: {self.REQUEST_TIMEOUT}s")
        logger.info(f"🔄 Max retries: {self.MAX_RETRIES}")
        logger.info(f"📊 Rate limits: {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour")
        
    def has_provider(self, provider: str) -> bool:
        """Check if a provider has available keys"""
        provider = provider.lower()
        if provider == "openai":
            return len(self.OPENAI_KEYS) > 0
        elif provider == "gemini":
            return len(self.GEMINI_KEYS) > 0
        elif provider == "openrouter":
            return len(self.OPENROUTER_KEYS) > 0
        elif provider == "anthropic":
            return len(self.ANTHROPIC_KEYS) > 0
        return False
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        providers = []
        if self.has_provider("openai"):
            providers.append("openai")
        if self.has_provider("anthropic"):
            providers.append("anthropic") 
        if self.has_provider("gemini"):
            providers.append("gemini")
        if self.has_provider("openrouter"):
            providers.append("openrouter")
        return providers

# Global config instance
config = Config()

# Export commonly used values for backward compatibility
DROPBOX_ACCESS_TOKEN = config.DROPBOX_ACCESS_TOKEN
DROPBOX_BASE_FOLDER = config.DROPBOX_BASE_FOLDER
OPENAI_KEYS = config.OPENAI_KEYS
GEMINI_KEYS = config.GEMINI_KEYS
OPENROUTER_KEYS = config.OPENROUTER_KEYS
OPENAI_FINE_TUNE_MODEL = config.OPENAI_MODEL
REQUEST_TIMEOUT = config.REQUEST_TIMEOUT

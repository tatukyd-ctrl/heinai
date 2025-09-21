# backend/main.py
import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import config
from backend.providers import call_auto_messages, stream_auto_messages, NoProvidersAvailableError, ProviderError
from backend.dropbox_utils import upload_to_dropbox

logger = logging.getLogger("heinbot.main")

# Load prompts/templates
PROMPTS_FILE = Path(__file__).parent / "prompts.json"
try:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
except FileNotFoundError:
    logger.warning("prompts.json not found, using defaults")
    PROMPTS = {
        "default": "Bạn là HeinBot, một AI assistant chuyên về lập trình và công nghệ. Hãy trả lời một cách chính xác, hữu ích và thân thiện.",
        "python_expert": "Bạn là một chuyên gia Python với kinh nghiệm sâu rộng. Hãy cung cấp code examples chi tiết và giải thích rõ ràng.",
        "creative": "Bạn là một AI sáng tạo, giúp brainstorm ý tưởng và giải pháp innovative cho các vấn đề lập trình.",
        "analyst": "Bạn là một data analyst chuyên nghiệp, giúp phân tích và visualization dữ liệu một cách hiệu quả."
    }

# FastAPI app setup
app = FastAPI(
    title="HeinBot API",
    description="Advanced AI Assistant for Programming and Technology",
    version="2.0.0",
    docs_url="/api/docs" if config.DEBUG else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security and performance middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "interest-cohort=()"
    
    return response

# Static files serving
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
else:
    logger.error(f"Frontend directory not found: {frontend_dir}")

# Request/Response models
class ChatRequest(BaseModel):
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    template: str = Field(default="default", description="Template to use for system prompt")
    provider: str = Field(default="auto", description="AI provider to use")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    reply: str
    provider_used: str
    file_path: Optional[str] = None
    dropbox_link: Optional[str] = None
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    error_type: str
    timestamp: str

# Utility functions
def ensure_system_message(messages: List[Dict[str, str]], template: str) -> List[Dict[str, str]]:
    """Ensure messages have a system message based on template"""
    if not messages:
        system_prompt = PROMPTS.get(template, PROMPTS["default"])
        return [{"role": "system", "content": system_prompt}]
    
    # Check if system message exists
    has_system = any(msg.get("role") == "system" for msg in messages)
    
    if not has_system:
        system_prompt = PROMPTS.get(template, PROMPTS["default"])
        return [{"role": "system", "content": system_prompt}] + messages
    
    return messages

async def save_conversation_file(
    messages: List[Dict[str, str]], 
    response: str, 
    metadata: Dict[str, str]
) -> Optional[str]:
    """Save conversation to file and optionally upload to Dropbox"""
    try:
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_id = str(uuid.uuid4())[:8]
        filename = f"chat_{timestamp}_{chat_id}.md"
        file_path = config.CHAT_STORAGE_DIR / filename
        
        # Create markdown content
        content = f"# HeinBot Conversation\n\n"
        content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"**Template:** {metadata.get('template', 'default')}\n"
        content += f"**Provider:** {metadata.get('provider', 'unknown')}\n\n"
        content += "---\n\n"
        
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "").title()
            content += f"## {role} Message {i}\n\n"
            content += f"{msg.get('content', '')}\n\n"
        
        content += "## Assistant Response\n\n"
        content += response
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Conversation saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")
        return None

async def upload_to_dropbox_background(file_path: str) -> Optional[str]:
    """Background task to upload file to Dropbox"""
    try:
        if config.DROPBOX_ACCESS_TOKEN and file_path:
            return upload_to_dropbox(file_path, config.DROPBOX_BASE_FOLDER)
    except Exception as e:
        logger.error(f"Dropbox upload failed: {e}")
    return None

# API Endpoints

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    available_providers = config.get_available_providers()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "providers": available_providers,
        "dropbox_enabled": bool(config.DROPBOX_ACCESS_TOKEN),
        "version": "2.0.0"
    }

@app.get("/api/providers")
async def get_providers():
    """Get available AI providers"""
    providers = []
    
    if config.has_provider("openai"):
        providers.append({"name": "openai", "display": "OpenAI GPT", "streaming": True})
    if config.has_provider("anthropic"):
        providers.append({"name": "anthropic", "display": "Anthropic Claude", "streaming": False})
    if config.has_provider("gemini"):
        providers.append({"name": "gemini", "display": "Google Gemini", "streaming": True})
    if config.has_provider("openrouter"):
        providers.append({"name": "openrouter", "display": "OpenRouter", "streaming": False})
    
    return {"providers": providers}

@app.get("/api/templates")
async def get_templates():
    """Get available chat templates"""
    templates = []
    for key, prompt in PROMPTS.items():
        templates.append({
            "name": key,
            "display": key.replace("_", " ").title(),
            "description": prompt[:100] + "..." if len(prompt) > 100 else prompt
        })
    
    return {"templates": templates}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Non-streaming chat endpoint"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Prepare messages
        messages = request.messages
        if not messages and request.prompt:
            messages = [{"role": "user", "content": request.prompt}]
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages or prompt provided")
        
        messages = ensure_system_message(messages, request.template)
        
        # Call AI provider
        response = await call_auto_messages(messages)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Save conversation in background
        metadata = {
            "template": request.template,
            "provider": request.provider,
            "processing_time": processing_time
        }
        
        file_path = await save_conversation_file(messages, response, metadata)
        
        # Upload to Dropbox in background if configured
        dropbox_link = None
        if file_path and config.DROPBOX_ACCESS_TOKEN:
            background_tasks.add_task(upload_to_dropbox_background, file_path)
        
        return ChatResponse(
            reply=response,
            provider_used="auto",
            file_path=file_path,
            dropbox_link=dropbox_link,
            processing_time=processing_time
        )
        
    except NoProvidersAvailableError as e:
        logger.error(f"No providers available: {e}")
        raise HTTPException(status_code=503, detail="No AI providers are currently available")
        
    except ProviderError as e:
        logger.error(f"Provider error: {e}")
        raise HTTPException(status_code=502, detail=f"AI provider error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    """Streaming chat endpoint"""
    try:
        body = await request.json()
        
        # Parse request
        messages = body.get("messages")
        if not messages and body.get("prompt"):
            messages = [{"role": "user", "content": body.get("prompt")}]
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages or prompt provided")
        
        template = body.get("template", "default")
        messages = ensure_system_message(messages, template)
        
        async def stream_generator():
            assembled_response = ""
            provider_used = "unknown"
            
            try:
                # Stream response
                async for chunk in stream_auto_messages(messages):
                    assembled_response += chunk
                    yield chunk
                
                # Save conversation after streaming
                try:
                    metadata = {
                        "template": template,
                        "provider": provider_used
                    }
                    
                    file_path = await save_conversation_file(messages, assembled_response, metadata)
                    
                    # Upload to Dropbox
                    if file_path and config.DROPBOX_ACCESS_TOKEN:
                        try:
                            dropbox_link = await upload_to_dropbox_background(file_path)
                            if dropbox_link:
                                yield f"\n\n[FILE_UPLOADED] {dropbox_link}"
                            else:
                                yield f"\n\n[FILE_UPLOADED] UPLOAD_FAILED"
                        except Exception as e:
                            logger.error(f"Dropbox upload error: {e}")
                            yield f"\n\n[FILE_UPLOADED] UPLOAD_ERROR"
                    else:
                        yield f"\n\n[FILE_SAVED] {file_path or 'LOCAL_SAVE_FAILED'}"
                        
                except Exception as e:
                    logger.error(f"File save error: {e}")
                    yield f"\n\n[FILE_ERROR] {str(e)}"
                    
            except NoProvidersAvailableError as e:
                yield f"\n\n[ERROR] No AI providers available: {str(e)}"
                
            except ProviderError as e:
                yield f"\n\n[ERROR] Provider error: {str(e)}"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"\n\n[ERROR] Unexpected error: {str(e)}"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Stream setup error: {e}")
        raise HTTPException(status_code=500, detail="Stream setup failed")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="http_error",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type="server_error", 
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """App startup initialization"""
    logger.info("🚀 HeinBot API starting up...")
    logger.info(f"📁 Chat storage: {config.CHAT_STORAGE_DIR}")
    logger.info(f"🌐 Frontend: {'✅ Found' if frontend_dir.exists() else '❌ Not found'}")
    logger.info(f"📦 Dropbox: {'✅ Configured' if config.DROPBOX_ACCESS_TOKEN else '❌ Not configured'}")
    
    available_providers = config.get_available_providers()
    logger.info(f"🤖 Available providers: {', '.join(available_providers) if available_providers else 'None'}")
    
    if not available_providers:
        logger.warning("⚠️  No AI providers configured! The app will not function properly.")

@app.on_event("shutdown")
async def shutdown_event():
    """App shutdown cleanup"""
    logger.info("🛑 HeinBot API shutting down...")

# Development endpoints (only in debug mode)
if config.DEBUG:
    @app.get("/api/debug/config")
    async def debug_config():
        """Debug configuration endpoint"""
        return {
            "dropbox_configured": bool(config.DROPBOX_ACCESS_TOKEN),
            "providers": {
                "openai": len(config.OPENAI_KEYS),
                "gemini": len(config.GEMINI_KEYS),
                "openrouter": len(config.OPENROUTER_KEYS),
                "anthropic": len(getattr(config, 'ANTHROPIC_KEYS', []))
            },
            "models": {
                "openai": config.OPENAI_MODEL,
                "gemini": config.GEMINI_MODEL,
                "openrouter": config.OPENROUTER_MODEL
            },
            "timeouts": {
                "request": config.REQUEST_TIMEOUT,
                "stream": config.STREAM_TIMEOUT
            }
        }
    
    @app.get("/api/debug/files")
    async def debug_files():
        """Debug saved files"""
        try:
            files = list(config.CHAT_STORAGE_DIR.glob("*.md"))
            return {
                "total_files": len(files),
                "recent_files": [f.name for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]]
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info" if config.DEBUG else "warning",
        access_log=config.DEBUG
    )

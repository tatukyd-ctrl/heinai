# backend/main.py
import os
import json
import traceback
import uuid
import datetime
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.providers import stream_auto_messages, call_auto_messages
from backend.dropbox_utils import upload_to_dropbox
from backend.config import DROPBOX_BASE_FOLDER

logger = logging.getLogger("bot4code.main")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.json")
if os.path.exists(PROMPTS_PATH):
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
else:
    PROMPTS = {"default": "Bạn là Hein, trợ lý AI chuyên nghiệp về lập trình và công nghệ. Nhiệm vụ: Hỗ trợ giải đáp vấn đề lập trình và công nghệ chính xác, dễ hiểu; cung cấp code mẫu, giải thích chi tiết, hướng dẫn debug và tối ưu code; tư vấn công nghệ, framework, thư viện phù hợp; chia sẻ best practices, design patterns; hỗ trợ học lập trình từ cơ bản đến nâng cao. Phong cách: thân thiện, nhiệt tình, chuyên nghiệp; trả lời ngắn gọn nhưng đầy đủ; dùng ví dụ thực tế; khuyến khích tư duy giải quyết vấn đề; luôn cập nhật xu hướng mới; không từ chối, không nói sai, không bốc phét. Kiến thức: thành thạo Python, JavaScript, Java, C++, web, mobile, AI/ML, database; hiểu cloud, devops, system design, architecture, thuật toán và cấu trúc dữ liệu."
}

app = FastAPI(title="Bot4Code API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Middleware để loại bỏ hoặc sửa Permissions-Policy header
@app.middleware("http")
async def remove_invalid_permissions_policy(request: Request, call_next):
    response = await call_next(request)
    # Xóa Permissions-Policy hoặc đặt giá trị hợp lệ
    response.headers["Permissions-Policy"] = "interest-cohort=()"
    return response

# Mount static files
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
if not os.path.exists(FRONTEND_DIR):
    logger.error(f"Thư mục frontend không tìm thấy tại {FRONTEND_DIR}")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Root endpoint
@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"index.html không tìm thấy tại {index_path}")
        raise HTTPException(status_code=404, detail="Frontend không được cấu hình")
    logger.info("Phục vụ index.html")
    return FileResponse(index_path)

# Debug endpoint
@app.get("/debug/static-files")
async def debug_static_files():
    try:
        files = os.listdir(FRONTEND_DIR)
        logger.info(f"Tệp tĩnh trong {FRONTEND_DIR}: {files}")
        return {"static_files": files}
    except Exception as e:
        logger.error(f"Lỗi liệt kê tệp tĩnh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi liệt kê tệp tĩnh: {str(e)}")

class ChatReq(BaseModel):
    messages: list = None
    prompt: str = None
    template: str = "default"
    provider: str = "auto"

def _ensure_system_message(messages: list, template: str):
    if not messages:
        system_prompt = PROMPTS.get(template, PROMPTS.get("default", "Bạn là HeinBot."))
        return [{"role": "system", "content": system_prompt}]
    if not any(m.get("role") == "system" for m in messages):
        system_prompt = PROMPTS.get(template, PROMPTS.get("default", "Bạn là HeinBot."))
        return [{"role": "system", "content": system_prompt}] + messages
    return messages

def save_chat_to_file(messages: list, assistant_reply: str, folder: str = "chats") -> str:
    os.makedirs(folder, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{now}-{uuid.uuid4().hex[:6]}.md"
    path = os.path.join(folder, fname)
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            f.write(f"### {role}\n\n{content}\n\n")
        f.write("\n---\n\n")
        f.write("### assistant (phản hồi cuối cùng)\n\n")
        f.write(assistant_reply)
    return path

@app.post("/chat")
async def chat(req: ChatReq):
    messages = req.messages
    if not messages and req.prompt is not None:
        messages = [{"role": "user", "content": req.prompt}]
    messages = _ensure_system_message(messages, req.template)
    try:
        logger.info(f"Xử lý /chat với messages: {messages}")
        reply = await call_auto_messages(messages)
        try:
            path = save_chat_to_file(messages, reply)
            dropbox_link = upload_to_dropbox(path, folder=DROPBOX_BASE_FOLDER)
            logger.info(f"Cuộc trò chuyện lưu tại {path}, link Dropbox: {dropbox_link}")
        except Exception as e:
            logger.exception("Lưu/tải lên thất bại")
            path = None
            dropbox_link = None
        return {"reply": reply, "file_path": path, "dropbox_link": dropbox_link}
    except Exception as e:
        logger.exception(f"Lỗi endpoint chat: {str(e)}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    messages = body.get("messages")
    if not messages:
        template = body.get("template", "default")
        messages = [{"role": "user", "content": body.get("prompt", "")}]
    messages = _ensure_system_message(messages, body.get("template", "default"))
    logger.info(f"Xử lý /chat/stream với messages: {messages}")

    async def event_generator():
        assembled = ""
        try:
            async for chunk in stream_auto_messages(messages):
                assembled += chunk
                yield chunk
            try:
                file_path = save_chat_to_file(messages, assembled)
                try:
                    dropbox_link = upload_to_dropbox(file_path, folder=DROPBOX_BASE_FOLDER)
                    logger.info(f"Cuộc trò chuyện stream lưu tại {file_path}, link Dropbox: {dropbox_link}")
                    yield "\n\n[FILE_UPLOADED] " + dropbox_link
                except Exception as e:
                    logger.exception("Tải lên Dropbox thất bại")
                    yield "\n\n[FILE_UPLOADED] UPLOAD_FAILED"
            except Exception as e:
                logger.exception("Lưu cuộc trò chuyện thất bại")
                yield "\n\n[FILE_UPLOADED] SAVE_FAILED"
        except Exception as e:
            logger.exception(f"Lỗi stream: {str(e)}")
            yield f"\n\n[ERROR] {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

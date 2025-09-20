import os, json, traceback, uuid, datetime, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
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
    PROMPTS = {"default": "You are CodeBot — help with code."}

app = FastAPI(title="Bot4Code API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatReq(BaseModel):
    messages: list = None
    prompt: str = None
    template: str = "default"
    provider: str = "auto"

def _ensure_system_message(messages: list, template: str):
    if not messages:
        system_prompt = PROMPTS.get(template, PROMPTS.get("default", "You are CodeBot."))
        return [{"role": "system", "content": system_prompt}]
    if not any(m.get("role") == "system" for m in messages):
        system_prompt = PROMPTS.get(template, PROMPTS.get("default", "You are CodeBot."))
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
        f.write("### assistant (final reply)\n\n")
        f.write(assistant_reply)
    return path

@app.post("/chat")
async def chat(req: ChatReq):
    messages = req.messages
    if not messages and req.prompt is not None:
        messages = [{"role": "user", "content": req.prompt}]
    messages = _ensure_system_message(messages, req.template)
    try:
        reply = await call_auto_messages(messages)
        try:
            path = save_chat_to_file(messages, reply)
            dropbox_link = upload_to_dropbox(path, folder=DROPBOX_BASE_FOLDER)
        except Exception:
            logger.exception("save/upload failed")
            path = None
            dropbox_link = None
        return {"reply": reply, "file_path": path, "dropbox_link": dropbox_link}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    messages = body.get("messages")
    if not messages:
        messages = [{"role": "user", "content": body.get("prompt", "")}]
    messages = _ensure_system_message(messages, body.get("template", "default"))

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
                    yield "\n\n[FILE_UPLOADED] " + dropbox_link
                except Exception:
                    logger.exception("Dropbox upload failed")
                    yield "\n\n[FILE_UPLOADED] UPLOAD_FAILED"
            except Exception:
                logger.exception("Saving chat to file failed")
                yield "\n\n[FILE_UPLOADED] SAVE_FAILED"
        except Exception as e:
            logger.exception("Stream error")
            yield f"\n\n[ERROR] {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

# ✅ Entry point cho local/Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)

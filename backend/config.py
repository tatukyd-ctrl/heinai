# backend/config.py
import os
import logging

# Thiết lập logging
logger = logging.getLogger("bot4code.config")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Lấy biến môi trường với giá trị mặc định
try:
    DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "")
    if not DROPBOX_ACCESS_TOKEN:
        logger.warning("Không tìm thấy DROPBOX_ACCESS_TOKEN trong biến môi trường")

    DROPBOX_BASE_FOLDER = os.getenv("DROPBOX_BASE_FOLDER", "/my-app-chats")
    logger.info(f"DROPBOX_BASE_FOLDER được đặt thành: {DROPBOX_BASE_FOLDER}")

    # Xử lý các khóa API, đảm bảo danh sách không rỗng
    OPENAI_KEYS = os.getenv("OPENAI_KEYS", "").split(",") if os.getenv("OPENAI_KEYS") else []
    OPENAI_KEYS = [key.strip() for key in OPENAI_KEYS if key.strip()]  # Loại bỏ khoảng trắng
    if not OPENAI_KEYS:
        logger.warning("Không tìm thấy OPENAI_KEYS hoặc danh sách rỗng")
    else:
        logger.info(f"Tìm thấy {len(OPENAI_KEYS)} khóa OpenAI")

    GEMINI_KEYS = os.getenv("GEMINI_KEYS", "").split(",") if os.getenv("GEMINI_KEYS") else []
    GEMINI_KEYS = [key.strip() for key in GEMINI_KEYS if key.strip()]
    if not GEMINI_KEYS:
        logger.warning("Không tìm thấy GEMINI_KEYS hoặc danh sách rỗng")
    else:
        logger.info(f"Tìm thấy {len(GEMINI_KEYS)} khóa Gemini")

    OPENROUTER_KEYS = os.getenv("OPENROUTER_KEYS", "").split(",") if os.getenv("OPENROUTER_KEYS") else []
    OPENROUTER_KEYS = [key.strip() for key in OPENROUTER_KEYS if key.strip()]
    if not OPENROUTER_KEYS:
        logger.warning("Không tìm thấy OPENROUTER_KEYS hoặc danh sách rỗng")
    else:
        logger.info(f"Tìm thấy {len(OPENROUTER_KEYS)} khóa OpenRouter")

    # Model mặc định cho OpenAI
    OPENAI_FINE_TUNE_MODEL = os.getenv("OPENAI_FINE_TUNE_MODEL", "gpt-4o-mini")
    logger.info(f"OPENAI_FINE_TUNE_MODEL được đặt thành: {OPENAI_FINE_TUNE_MODEL}")

    # Timeout cho các yêu cầu HTTP
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 60.0))
    logger.info(f"REQUEST_TIMEOUT được đặt thành: {REQUEST_TIMEOUT} giây")

except Exception as e:
    logger.error(f"Lỗi khi đọc biến môi trường: {str(e)}")
    raise RuntimeError(f"Không thể khởi tạo cấu hình: {str(e)}")

# run_bot.py
import os
import subprocess
import sys
import webbrowser
import time

def main():
    # Đảm bảo đã cài uvicorn
    try:
        import uvicorn
    except ImportError:
        print("Bạn chưa cài uvicorn. Cài bằng: pip install uvicorn")
        sys.exit(1)

    # Tìm đường dẫn frontend
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(frontend_path):
        print("Không tìm thấy frontend/index.html. Vui lòng kiểm tra.")
    else:
        # Mở giao diện frontend
        webbrowser.open_new_tab(frontend_path)

    # Chạy uvicorn server (backend/main.py)
    print("Khởi động server tại http://127.0.0.1:8000 ...")
    # bạn có thể đổi host, port tại đây
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()

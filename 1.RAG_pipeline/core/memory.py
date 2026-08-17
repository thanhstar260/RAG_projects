import os
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from config import MEMORY_TYPE, HISTORY_DIR

# Lưu lịch sử chat trên RAM (Dictionary với key là session_id)
_ram_store = {}

def get_chat_history(session_id: str):
    if MEMORY_TYPE == "FILE":
        file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        return FileChatMessageHistory(file_path)
    else:
        # Nếu chọn RAM, khởi tạo history mới nếu session chưa tồn tại
        if session_id not in _ram_store:
            _ram_store[session_id] = ChatMessageHistory()
        return _ram_store[session_id]
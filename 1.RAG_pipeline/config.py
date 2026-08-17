import os

# --- THIẾT LẬP ĐƯỜNG DẪN THƯ MỤC ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")

# Thêm đường dẫn tới thư mục models
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Tự động tạo thư mục nếu chưa tồn tại
for d in [DATA_DIR, DB_DIR, HISTORY_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- CẤU HÌNH CHUNKING ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- CẤU HÌNH EMBEDDING MODEL ---
# Các tùy chọn: "BGE", "E5", "NOMIC"
DEFAULT_EMBEDDING = "BGE"
EMBEDDING_MODELS = {
    "BGE": "BAAI/bge-m3",
    "E5": "intfloat/multilingual-e5-base",
    "NOMIC": "nomic-ai/nomic-embed-text-v1.5"
}

# --- CẤU HÌNH VECTOR DATABASE ---
# Các tùy chọn: "CHROMA", "FAISS", "QDRANT", "MILVUS", "WEAVIATE"
DEFAULT_VECTOR_DB = "CHROMA"

# --- CẤU HÌNH DEEPSEEK LLM ---
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash" # hoặc "deepseek-v4-pro"

# --- CẤU HÌNH LƯU TRỮ LỊCH SỬ CHAT ---
# Các tùy chọn: "RAM", "FILE"
MEMORY_TYPE = "RAM"
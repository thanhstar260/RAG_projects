import sys
import os

# Thêm BASE_DIR vào sys.path để Python hiểu được các module core, config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_loader import load_and_split_documents
from core.embeddings import get_embedding_model
from core.vector_stores import get_vector_store
from config import DEFAULT_VECTOR_DB, DEFAULT_EMBEDDING

def run_ingestion():
    print("=== BẮT ĐẦU QUÁ TRÌNH NẠP DỮ LIỆU (INGESTION) ===")
    
    # Bước 1: Load và chia nhỏ tài liệu
    print("\n[1/3] Đang load và chia nhỏ tài liệu...")
    chunks = load_and_split_documents()
    if not chunks:
        print("Không có tài liệu nào trong thư mục data/. Hủy bỏ.")
        return

    # Bước 2: Tải Embedding Model
    print(f"\n[2/3] Đang tải Embedding Model ({DEFAULT_EMBEDDING})...")
    embeddings = get_embedding_model()

    # Bước 3: Khởi tạo DB và lưu vector
    print(f"\n[3/3] Đang nhúng (embed) và lưu vào {DEFAULT_VECTOR_DB}...")
    # Khi truyền 'docs=chunks', DB sẽ thực hiện việc embed và persist xuống ổ cứng
    get_vector_store(embeddings=embeddings, docs=chunks)
    
    print("\n=== HOÀN TẤT NẠP DỮ LIỆU ===")
    print(f"Cơ sở dữ liệu {DEFAULT_VECTOR_DB} đã sẵn sàng trong thư mục db/")

if __name__ == "__main__":
    run_ingestion()
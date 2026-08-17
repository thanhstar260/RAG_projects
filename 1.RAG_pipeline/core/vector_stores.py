import os
from config import DB_DIR, DEFAULT_VECTOR_DB
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, Qdrant, Milvus, Weaviate

def get_vector_store(embeddings, db_type=DEFAULT_VECTOR_DB, docs=None):
    """
    Nếu docs (chunks) có dữ liệu -> Tạo DB mới và lưu.
    Nếu docs=None -> Load DB đã lưu từ thư mục để truy vấn.
    """
    persist_dir = os.path.join(DB_DIR, db_type.lower())
    
    if db_type == "CHROMA":
        if docs:
            return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
        else:
            return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            
    elif db_type == "FAISS":
        index_name = "index"
        if docs:
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(persist_dir, index_name)
            return db
        else:
            return FAISS.load_local(persist_dir, embeddings, index_name, allow_dangerous_deserialization=True)
            
    elif db_type == "QDRANT":
        if docs:
            return Qdrant.from_documents(docs, embeddings, path=persist_dir, collection_name="rag_collection")
        else:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=persist_dir)
            return Qdrant(client=client, collection_name="rag_collection", embeddings=embeddings)
    
    # Đối với Milvus và Weaviate, giả định bạn đang chạy server local ở port mặc định
    elif db_type == "MILVUS":
        connection_args = {"host": "127.0.0.1", "port": "19530"}
        if docs:
            return Milvus.from_documents(docs, embeddings, connection_args=connection_args)
        else:
            return Milvus(embedding_function=embeddings, connection_args=connection_args)
            
    elif db_type == "WEAVIATE":
        import weaviate
        client = weaviate.Client(url="http://localhost:8080")
        if docs:
            return Weaviate.from_documents(docs, embeddings, client=client, index_name="RagIndex")
        else:
            return Weaviate(client=client, index_name="RagIndex", text_key="text", embedding=embeddings)
    else:
        raise ValueError(f"Vector DB {db_type} không được hỗ trợ.")
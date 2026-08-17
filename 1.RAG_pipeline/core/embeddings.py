from langchain_huggingface import HuggingFaceEmbeddings
from config import DEFAULT_EMBEDDING, EMBEDDING_MODELS, MODELS_DIR

def get_embedding_model(model_name=DEFAULT_EMBEDDING):
    model_id = EMBEDDING_MODELS.get(model_name)
    if not model_id:
        raise ValueError(f"Model {model_name} không được hỗ trợ.")
    
    print(f"Đang khởi tạo Embedding Model: {model_name} ({model_id})...")
    
    # 'trust_remote_code=True' cần thiết cho một số model như Nomic
    model_kwargs = {
        'device': 'cpu', 
        'trust_remote_code': True,
    }
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        cache_folder=MODELS_DIR, # <--- Bắt buộc model tải về đây
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
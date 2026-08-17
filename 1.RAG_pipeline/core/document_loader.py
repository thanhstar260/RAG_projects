import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_documents():
    documents = []
    
    # Quét tất cả file trong thư mục data/
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        
        # Hỗ trợ PDF
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        # Hỗ trợ TXT
        elif file.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            
    print(f"Đã load thành công {len(documents)} trang/tài liệu.")
    
    # Cắt nhỏ tài liệu
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(chunks)} chunks.")
    return chunks
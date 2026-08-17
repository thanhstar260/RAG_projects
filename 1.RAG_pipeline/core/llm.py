import os
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_URL, DEEPSEEK_MODEL

def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Lỗi: DEEPSEEK_API_KEY chưa được khai báo trong file .env!")
    
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        base_url=DEEPSEEK_API_URL,
        temperature=0.1,  # Nhiệt độ thấp (0.1 - 0.3) giúp RAG trả lời bám sát Context hơn
        max_tokens=2048
    )
    return llm
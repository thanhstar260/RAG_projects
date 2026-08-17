import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env (chứa API KEY)
load_dotenv()
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


from pipeline.chat_chain import build_conversational_rag_chain

def main():
    print("Đang khởi tạo Hệ thống RAG... (Quá trình này có thể mất vài giây để load Model)")
    
    try:
        chat_chain = build_conversational_rag_chain()
    except Exception as e:
        print(f"\n[LỖI KHỞI TẠO]: {e}")
        print("Hãy chắc chắn bạn đã chạy file 'pipeline/ingest.py' để tạo Database trước!")
        return

    print("\n" + "="*50)
    print("Hệ thống RAG đã sẵn sàng! Gõ 'quit' hoặc 'exit' để thoát.")
    print("="*50 + "\n")

    # Giả định 1 session_id cho phiên làm việc này. 
    # Nếu bạn dùng bản có Web UI (Streamlit), session_id này sẽ sinh tự động cho từng user.
    session_id = "default_session"

    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Tạm biệt!")
            break
        if not user_input.strip():
            continue

        try:
            # Gọi Chain xử lý
            response = chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nAI: {response['answer']}")
        except Exception as e:
             print(f"\n[LỖI TRUY VẤN]: {e}")

if __name__ == "__main__":
    main()
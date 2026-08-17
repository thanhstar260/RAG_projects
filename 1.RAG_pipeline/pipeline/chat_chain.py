import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.embeddings import get_embedding_model
from core.vector_stores import get_vector_store
from core.llm import get_llm
from core.memory import get_chat_history

def build_conversational_rag_chain():
    # 1. Khởi tạo các thành phần lõi
    llm = get_llm()
    embeddings = get_embedding_model()
    # Truyền docs=None để chỉ LOAD database, không ghi đè
    vectorstore = get_vector_store(embeddings=embeddings, docs=None)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ==========================================
    # BƯỚC 1: XỬ LÝ LẠI CÂU HỎI (CONTEXTUALIZE)
    # ==========================================
    # Nhiệm vụ: Dùng lịch sử chat để biến câu hỏi hiện tại (có thể chứa đại từ 'nó', 'anh ấy') 
    # thành một câu hỏi độc lập đầy đủ nghĩa, trước khi mang đi tìm kiếm (retrieve).
    contextualize_q_system_prompt = (
        "Đưa ra lịch sử trò chuyện và câu hỏi mới nhất của người dùng "
        "(câu hỏi này có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện), "
        "hãy định dạng lại (rephrase) câu hỏi này thành một câu hỏi độc lập "
        "mà không cần xem lịch sử trò chuyện vẫn hiểu được. "
        "TUYỆT ĐỐI KHÔNG trả lời câu hỏi, chỉ định dạng lại nó nếu cần, nếu không cần thì trả về nguyên bản."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    # Tạo retriever nhận thức được lịch sử
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


    # ==========================================
    # BƯỚC 2: TẠO CÂU TRẢ LỜI RAG
    # ==========================================
    # Nhiệm vụ: Trả lời câu hỏi dựa trên Context (được History Aware Retriever tìm ra)
    qa_system_prompt = (
        "Bạn là một trợ lý ảo chuyên nghiệp và hữu ích.\n"
        "Nhiệm vụ của bạn là trả lời câu hỏi của người dùng CHỈ dựa trên các tài liệu ngữ cảnh (Context) được cung cấp dưới đây.\n"
        "- Nếu thông tin không có trong tài liệu ngữ cảnh, hãy nói rõ ràng rằng 'Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.' Tuyệt đối không tự sáng tạo hoặc bịa đặt thông tin.\n"
        "- Hãy trả lời một cách chi tiết, mạch lạc và bằng ngôn ngữ giống với ngôn ngữ mà người dùng đặt câu hỏi.\n"
        "\nNgữ cảnh được cung cấp:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    # Chain chuyên ghép nối các chunks vào prompt để LLM đọc
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Ghép Retriever và QA Chain lại với nhau
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # ==========================================
    # BƯỚC 3: TÍCH HỢP BỘ NHỚ (MEMORY)
    # ==========================================
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_chat_history, # Hàm chúng ta đã viết trong core/memory.py
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
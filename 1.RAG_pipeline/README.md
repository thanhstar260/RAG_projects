# 🤖 Project 1: Basic Conversational RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) system built with LangChain and DeepSeek. It supports conversational memory and allows easy switching between different vector databases and local embedding models via a central configuration file.

## ✨ Key Features
- **LLM**: DeepSeek API (`deepseek-v4-flash`).
- **Vector DBs**: Chroma (default), FAISS, Qdrant, Milvus, Weaviate.
- **Embeddings (Local)**: BGE-M3 (default), E5, Nomic.
- **Memory**: Context-aware chat history (RAM or File-based).
- **Architecture**: Separated data ingestion and chat inference pipelines.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Configure API Key

Create a `.env` file in the project root and add your DeepSeek API key:

```env
DEEPSEEK_API_KEY=sk-your_api_key_here

```

### 3. Ingest Data

Place your documents (`.txt` or `.pdf`) into the `data/` folder, then run the ingestion script.
*(Note: The first run will download the local embedding model to the `models/` directory).*

```bash
python pipeline/ingest.py

```

### 4. Start Chatting

Launch the terminal-based chatbot:

```bash
python main.py

```

## ⚙️ Configuration

Modify `config.py` to easily customize your pipeline:

* `DEFAULT_VECTOR_DB`: Switch between "CHROMA", "FAISS", "QDRANT", "MILVUS", or "WEAVIATE".
* `DEFAULT_EMBEDDING`: Switch between "BGE", "E5", or "NOMIC".
* `MEMORY_TYPE`: Choose "RAM" (temporary) or "FILE" (persistent).
* Adjust chunk size and overlap limits.

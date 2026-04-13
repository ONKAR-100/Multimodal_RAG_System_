📘 Multimodal RAG System (Ollama Edition)
🔍 Overview

The Multimodal Retrieval-Augmented Generation (RAG) System is an intelligent AI-powered platform that enables unified search and understanding across multiple data formats including PDFs, DOCX, images, and audio files,

It extracts, processes, and indexes multimodal data into a shared vector space, allowing users to perform natural language queries, and receive accurate, citation-based answers, using a local LLM powered by Ollama,

🚀 Key Features
📂 Multimodal Data Support
PDF (page & paragraph-level extraction),
DOCX (structured text extraction),
Images (OCR using Tesseract),
Audio (Speech-to-text using Whisper),

🔎 Hybrid Search
Semantic search (vector similarity),
Keyword search (text-based),
Combined using Reciprocal Rank Fusion (RRF),

🧠 Advanced RAG Pipeline
Embedding generation (SentenceTransformers),
Reranking (CrossEncoder),
Context-grounded response generation,

📊 Citation-Based Answers
Every answer is backed by sources,
Includes document references (page/time/paragraph),
Eliminates hallucination,

⚡ FastAPI Backend
File upload API,
Query API,
Health monitoring endpoint,

🏗️ System Architecture

Input Files (PDF / DOCX / Image / Audio),
↓
Preprocessing (OCR / ASR / Parsing),
↓
Text Chunking,
↓
Embedding Model (SentenceTransformers),
↓
Vector Database (Qdrant),
↓
Hybrid Retrieval (Vector + Keyword),
↓
Reranking (Cross Encoder),
↓
LLM (Ollama),
↓
Answer + Citations,

🛠️ Tech Stack
Backend: FastAPI,
LLM: Ollama (Llama3 / Phi3 / ChatQA models),
Embeddings: SentenceTransformers,
Reranker: CrossEncoder (MiniLM),
Vector DB: Qdrant (local storage),
OCR: Tesseract,
Speech-to-Text: Whisper,
Frontend: Static HTML (served via FastAPI),

📂 Supported File Types
PDF,
DOCX,
PNG, JPG, JPEG,
MP3, WAV, M4A, AAC,

⚙️ Installation Guide
1. Clone Repository

git clone <your-repo-url>,
cd multimodal-rag,

2. Install Python Dependencies

pip install -r requirements.txt,

3. Install System Dependencies
Install Tesseract OCR,
Install FFmpeg,
4. Install Ollama

ollama pull llama3-chatqa:8b,
or
ollama pull phi3,

▶️ Running the Application

python main.py,

Access: http://127.0.0.1:8000/
,

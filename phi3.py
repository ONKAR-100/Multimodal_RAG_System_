# http://127.0.0.1:8000/ run in browser
import os
import uuid
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import shutil

# File processing
import pypdf
import docx
import pytesseract
from PIL import Image
import whisper

# Core RAG components
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
import ollama  # <-- MODIFICATION: Replaced google.generativeai with ollama

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Utility
from tqdm import tqdm
import warnings
import json

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# NOTE: Before running, ensure you have the Ollama library installed:
# pip install ollama
# Also, make sure the Ollama application is running and you have pulled the model:
# ollama pull llama3-chatqa:8b

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
SOURCE_DIRECTORY = "source_documents"
VECTOR_DB_COLLECTION_NAME = "multimodal_rag_sih_advanced"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
WHISPER_MODEL_NAME = "base"
# <-- MODIFICATION: Swapped Gemini model for Ollama model -->
OLLAMA_MODEL_NAME = "phi3:latest"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- 2. INITIALIZE CORE SERVICES ---
print("--- Initializing Services (this may take a moment on first run) ---")

# <-- MODIFICATION: Check if Ollama service is running -->
try:
    ollama.list()
    logger.info(f"Ollama service is running. Default model: '{OLLAMA_MODEL_NAME}'")
except Exception as e:
    logger.error("Ollama service not found. Please ensure the Ollama application is running.")
    logger.error(f"Error details: {e}")
    # Exit if Ollama is not available, as it's a core component.
    raise RuntimeError("Could not connect to Ollama service.")

vector_db_client = QdrantClient(path="qdrant_storage")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

# Load whisper model once to avoid reloading
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("Whisper model loaded.")
except Exception as e:
    print(f"Could not load Whisper model: {e}")
    whisper_model = None

# Create Qdrant collection for hybrid search
try:
    if not vector_db_client.collection_exists(VECTOR_DB_COLLECTION_NAME):
        vector_db_client.create_collection(
            collection_name=VECTOR_DB_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )
        print("Vector DB collection created.")

        vector_db_client.create_payload_index(
            collection_name=VECTOR_DB_COLLECTION_NAME,
            field_name="text",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=15,
                lowercase=True,
            ),
        )
        print("Text index created on payload for hybrid search.")
        
        vector_db_client.create_payload_index(
            collection_name=VECTOR_DB_COLLECTION_NAME,
            field_name="doc_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Keyword index created on doc_type payload.")
    else:
        print(f"Collection '{VECTOR_DB_COLLECTION_NAME}' loaded from persistent storage.")

except Exception as e:
    print(f"Error during collection or index creation: {e}")
print("--- Services Initialized ---\n")

# --- 3. INGESTION AND PREPROCESSING ---
def chunk_text(text: str, source: str, doc_type: str = "unknown") -> List[Dict[str, Any]]:
    if not text: return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append({'text': chunk_text, 'metadata': {'source': source, 'doc_type': doc_type}})
    return chunks

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text from a PDF page by page, paragraph by paragraph."""
    all_chunks = []
    try:
        reader = pypdf.PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if not page_text:
                continue

            paragraphs = page_text.split('\n\n')
            if len(paragraphs) == 1:
                paragraphs = page_text.split('\n')

            for para_num, para_text in enumerate(paragraphs):
                if len(para_text.strip()) > 20:
                    source = f"{os.path.basename(file_path)}#page={page_num + 1},paragraph={para_num + 1}"
                    all_chunks.extend(chunk_text(para_text, source, "pdf"))
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
    return all_chunks

def process_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text from a DOCX file paragraph by paragraph."""
    all_chunks = []
    try:
        doc = docx.Document(file_path)
        for para_num, para in enumerate(doc.paragraphs):
            para_text = para.text.strip()
            if len(para_text) > 20:
                source = f"{os.path.basename(file_path)}#paragraph={para_num + 1}"
                all_chunks.extend(chunk_text(para_text, source, "docx"))
    except Exception as e:
        print(f"Error processing DOCX {file_path}: {e}")
    return all_chunks

def process_image(file_path: str) -> List[Dict[str, Any]]:
    all_chunks = []
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        if text:
            all_chunks.extend(chunk_text(text, f"{os.path.basename(file_path)} (OCR)", "image"))
    except Exception as e:
        print(f"Error processing Image {file_path}: {e}")
    return all_chunks

def process_audio(file_path: str) -> List[Dict[str, Any]]:
    """Transcribes audio using Whisper and creates chunks from segments."""
    ffmpeg_path = shutil.which("ffmpeg")
    logger.info(f"Checking for ffmpeg. Found at: {ffmpeg_path}")
    if not ffmpeg_path:
        logger.warning("FFmpeg not found in PATH. Audio processing may fail for non-WAV files.")

    if not whisper_model:
        print("Whisper model not available, skipping audio processing.")
        return []

    print(f"Processing Audio (Whisper): {os.path.basename(file_path)}")
    all_chunks = []
    try:
        result = whisper_model.transcribe(file_path, word_timestamps=True)
        segments = result.get('segments', [])
        for seg in segments:
            start_time = f"{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}"
            end_time = f"{int(seg['end'] // 60):02d}:{int(seg['end'] % 60):02d}"
            source = f"{os.path.basename(file_path)}#time={start_time}-{end_time}"
            all_chunks.append({'text': seg['text'].strip(), 'metadata': {'source': source, 'doc_type': 'audio'}})

    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        raise e
    return all_chunks

async def process_and_index_single_file(file_path: str, filename: str):
    """Processes, embeds, and indexes a single document."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at '{file_path}'")

    logger.info(f"Processing new file: {filename}")

    all_chunks = []
    ext_to_processor = {
        'pdf': process_pdf,
        'docx': process_docx,
        'png': process_image,
        'jpg': process_image,
        'jpeg': process_image,
        'mp3': process_audio,
        'wav': process_audio,
        'm4a': process_audio,
        'aac': process_audio,
    }

    file_ext = filename.lower().split('.')[-1]

    if file_ext in ext_to_processor:
        processor_func = ext_to_processor[file_ext]
        try:
            processed_chunks = await run_in_threadpool(processor_func, file_path)
            all_chunks.extend(processed_chunks)
        except Exception as e:
            logger.error(f"Error running processor for {filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: '.{file_ext}'")
    
    logger.info(f"📄 File type detected: {file_ext}")
    logger.info(f"📊 Total chunks extracted: {len(all_chunks)}")

    if not all_chunks:
        return {"message": f"File '{filename}' processed, but no text content was extracted.", "chunks_indexed": 0}

    logger.info(f"Extracted {len(all_chunks)} chunks. Now creating embeddings...")

    texts_to_embed = [str(chunk.get('text') or '') for chunk in all_chunks]
    embeddings = await run_in_threadpool(embedding_model.encode, texts_to_embed, show_progress_bar=True)

    logger.info("Indexing chunks into Vector DB...")
    vector_db_client.upsert(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk['text'], 
                    "source": chunk['metadata']['source'],
                    "doc_type": chunk['metadata'].get('doc_type', 'unknown')
                }
            ) for chunk, embedding in zip(all_chunks, embeddings)
        ],
        wait=True
    )

    return {"message": f"Successfully indexed {len(all_chunks)} chunks from {filename}", "chunks_indexed": len(all_chunks)}

def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """Builds the prompt for using the provided template."""

    evidence_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        evidence_str += f"[{i+1}] source: {chunk['payload']['source']}\n"
        evidence_str += f"content: \"{chunk['payload']['text']}\"\n\n"

    prompt_template = f"""
You are a highly precise and strict expert assistant. You must answer the user's query using **ONLY** the evidence provided below. 

STRICT RULES:
1. Grounding: You must formulate your answer based entirely on the provided evidence. Do not use outside knowledge.
2. Citations: Every single fact or claim you state MUST be followed by the bracketed citation number of the source document(s) you used, such as [1] or [3].
3. Refusal: If the provided evidence does not contain the answer, you are FORBIDDEN from guessing or hallucinating. You MUST exactly reply: "I cannot answer this question based on the provided documents."
4. Conflict Resolution: If you find conflicting information across different sources, explicitly state the conflict, explain what each source says, and then summarize the discrepancy. Do not say you cannot answer just because there is a conflict.

Query: "{query}"

Evidence:
{evidence_str}

Answer:
"""
    return prompt_template

async def ask_question(query: str, doc_type: str = None):
    """Performs the full RAG pipeline with Hybrid Search, Reranking, and RRF Score Fusion."""

    # --- Part 1: Hybrid Search (Vector + Keyword) ---
    logger.info("1. Performing Hybrid Search (Vector + Keyword)...")
    query_embedding = embedding_model.encode(query)

    query_filter = None
    if doc_type:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
        )

    # ✅ FIX 1: Correct Qdrant vector search
    vector_hits = vector_db_client.query_points(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=query_filter,
        limit=40
    ).points

    # Keyword search
    keyword_must = [models.FieldCondition(key="text", match=models.MatchText(text=query))]
    if doc_type:
        keyword_must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)))

    keyword_hits = vector_db_client.scroll(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        scroll_filter=models.Filter(must=keyword_must),
        limit=40
    )[0]

    # ✅ FIX 2: True Score Fusion using RRF
    k = 60
    vector_rank = {hit.id: rank for rank, hit in enumerate(vector_hits)}
    keyword_rank = {hit.id: rank for rank, hit in enumerate(keyword_hits)}

    combined_hits = {}

    for hit in vector_hits:
        v_rank = vector_rank[hit.id]
        k_rank = keyword_rank.get(hit.id, 1000)
        rrf_score = (1 / (k + v_rank)) + (1 / (k + k_rank))
        
        combined_hits[hit.id] = {
            "id": hit.id,
            "payload": hit.payload,
            "score": rrf_score
        }

    for hit in keyword_hits:
        if hit.id not in combined_hits:
            v_rank = 1000
            k_rank = keyword_rank[hit.id]
            rrf_score = (1 / (k + v_rank)) + (1 / (k + k_rank))
            
            combined_hits[hit.id] = {
                "id": hit.id,
                "payload": hit.payload,
                "score": rrf_score
            }

    if not combined_hits:
        return {
            "answer": "I cannot answer this question based on the provided documents.",
            "sources": []
        }

    # --- Part 2: Reranking ---
    logger.info(f"2. Reranking {len(combined_hits)} candidates with Cross-Encoder...")

    rerank_pairs = [[query, hit['payload']['text']] for hit in combined_hits.values()]
    scores = reranker_model.predict(rerank_pairs)

    for hit, score in zip(combined_hits.values(), scores):
        hit['score'] = score

    reranked_hits = sorted(combined_hits.values(), key=lambda x: x['score'], reverse=True)
    top_k_reranked = reranked_hits[:5]

    logger.info("3. Building prompt with top 5 reranked results...")
    prompt = build_prompt(query, top_k_reranked)

    logger.info(f"4. Generating answer with Ollama ({OLLAMA_MODEL_NAME})...")

    try:
        def generate_ollama_response(prompt_text: str):
            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt_text}]
            )
            return response['message']['content'].strip()

        answer = await run_in_threadpool(generate_ollama_response, prompt)

        sources = []
        for i, chunk in enumerate(top_k_reranked):
            sources.append({
                "rank": i + 1,
                "source": chunk['payload']['source'],
                "score": round(float(chunk['score']), 4),
                "snippet": chunk['payload']['text'][:150] + "..."
            })

        return {
            "answer": answer,
            "sources": sources,
            "total_sources_considered": len(top_k_reranked)
        }

    except Exception as e:
        logger.error(f"Error communicating with LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    
# --- 5. FASTAPI APPLICATION ---
app = FastAPI(title="Multimodal RAG System (Ollama Edition)", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Ingest documents from the DATA folder at startup."""
    logger.info("Initializing system with data from 'DATA' directory...")
    data_dir = "DATA"
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                logger.info(f"Startup ingestion for {filename}...")
                try:
                    await process_and_index_single_file(file_path, filename)
                except Exception as e:
                    logger.error(f"Failed to ingest {filename} during startup: {e}")
    else:
        logger.warning(f"Data directory '{data_dir}' not found.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Create required directories ---
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve static files (for frontend)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = None
    try:
        file_ext = file.filename.split('.')[-1].lower()

        if file_ext not in ['pdf', 'docx', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a']:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # ✅ Unique filename (fix overwrite issue)
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")

        # ✅ FIX: streaming write (no corruption / empty file issue)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"✅ File saved: {file_path}")
        logger.info(f"📦 File size: {os.path.getsize(file_path)} bytes")

        # ✅ Process file
        result = await process_and_index_single_file(file_path, file.filename)

        logger.info(f"🧠 Chunks indexed: {result['chunks_indexed']}")

        if result["chunks_indexed"] == 0:
            logger.warning("⚠️ WARNING: File processed but NO TEXT extracted!")

        # ✅ delete after processing
        os.remove(file_path)

        return result

    except Exception as e:
        logger.error(f"❌ UPLOAD ERROR: {e}")

        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    query: str
    doc_type: str = None

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query the document database."""
    try:
        query = request.query
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        result = await ask_question(query, request.doc_type)
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Multimodal RAG System"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
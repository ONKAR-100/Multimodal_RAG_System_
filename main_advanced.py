import os
import uuid
from typing import List, Dict, Any
import tkinter as tk
from tkinter import filedialog

# File processing
import pypdf
import docx
import pytesseract
from PIL import Image
import whisper

# Core RAG components
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
import ollama

# Utility
from tqdm import tqdm
import warnings
from collections import defaultdict

# Suppress verbose warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
SOURCE_DIRECTORY = "source_documents"
VECTOR_DB_COLLECTION_NAME = "multimodal_rag_sih_advanced"
# Switched to a multilingual model for embedding
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL_NAME = "mistral:latest"
WHISPER_MODEL_NAME = "base"  # "base" is good for multilingual, use "small" or "medium" for higher accuracy

CHUNK_SIZE = 300  # Reduced for more granular chunks
CHUNK_OVERLAP = 40

# --- 2. INITIALIZE CORE SERVICES ---
print("--- Initializing Services (this may take a moment on first run) ---")
vector_db_client = QdrantClient(":memory:")
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
    # Step 1: Create the collection without the payload_schema
    vector_db_client.recreate_collection(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    print("Vector DB collection created.")
    
    # Step 2: Create the payload index for the 'text' field in a separate step
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

except Exception as e:
    print(f"Error during collection or index creation: {e}")
print("--- Services Initialized ---\n")

# --- 3. INGESTION AND PREPROCESSING ---
def chunk_text(text: str, source: str) -> List[Dict[str, Any]]:
    # (Same chunking logic as before)
    if not text: return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append({'text': chunk_text, 'metadata': {'source': source}})
    return chunks

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extracts text from a PDF page by page, paragraph by paragraph."""
    all_chunks = []
    try:
        reader = pypdf.PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            # Extract text and split into paragraphs (assuming paragraphs are separated by double newlines)
            page_text = page.extract_text()
            if not page_text:
                continue
            
            paragraphs = page_text.split('\n \n') # A simple way to detect paragraphs
            if len(paragraphs) == 1: # If splitting by double newline fails, try single
                paragraphs = page_text.split('\n')

            for para_num, para_text in enumerate(paragraphs):
                if len(para_text.strip()) > 20: # Ignore very short paragraphs/stray lines
                    # More granular source tracking: page and paragraph number
                    source = f"{os.path.basename(file_path)}#page={page_num + 1},paragraph={para_num + 1}"
                    all_chunks.extend(chunk_text(para_text, source))
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
            if len(para_text) > 20: # Ignore very short paragraphs
                # More granular source tracking: paragraph number
                source = f"{os.path.basename(file_path)}#paragraph={para_num + 1}"
                all_chunks.extend(chunk_text(para_text, source))
    except Exception as e: 
        print(f"Error processing DOCX {file_path}: {e}")
    return all_chunks

def process_image(file_path: str) -> List[Dict[str, Any]]:
    # (Same logic as before)
    all_chunks = []
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        if text:
            all_chunks.extend(chunk_text(text, f"{os.path.basename(file_path)} (OCR)"))
    except Exception as e: print(f"Error processing Image {file_path}: {e}")
    return all_chunks

def process_audio(file_path: str) -> List[Dict[str, Any]]:
    """Transcribes audio using Whisper and creates chunks from segments."""
    if not whisper_model:
        print("Whisper model not available, skipping audio processing.")
        return []
        
    print(f"Processing Audio (Whisper): {os.path.basename(file_path)}")
    all_chunks = []
    try:
        # Transcribe with word-level timestamps for accurate chunking
        result = whisper_model.transcribe(file_path, word_timestamps=True)
        
        # Group words into sentence-like chunks based on timestamps
        segments = result.get('segments', [])
        for seg in segments:
            start_time = f"{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}"
            end_time = f"{int(seg['end'] // 60):02d}:{int(seg['end'] % 60):02d}"
            source = f"{os.path.basename(file_path)}#time={start_time}-{end_time}"
            # We treat each transcribed segment as a chunk
            all_chunks.append({'text': seg['text'].strip(), 'metadata': {'source': source}})
            
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
    return all_chunks

def process_and_index_single_file(file_path: str):
    """Processes, embeds, and indexes a single document provided by its file path."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"\n--- Processing new file: {os.path.basename(file_path)} ---")
    
    all_chunks = []
    filename = os.path.basename(file_path)
    # Use a dictionary to map extensions to functions for cleaner code
    ext_to_processor = {
        'pdf': process_pdf,
        'docx': process_docx,
        'png': process_image,
        'jpg': process_image,
        'jpeg': process_image,
        'mp3': process_audio,
        'wav': process_audio,
        'm4a': process_audio,
    }
    
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext in ext_to_processor:
        processor_func = ext_to_processor[file_ext]
        all_chunks.extend(processor_func(file_path))
    else:
        print(f"Unsupported file type: '.{file_ext}'. Skipping.")
        return

    if not all_chunks:
        print("No text could be extracted from the document. Nothing to index.")
        return
        
    print(f"\nExtracted {len(all_chunks)} chunks. Now creating embeddings...")

    texts_to_embed = [chunk['text'] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)

    print("Indexing chunks into Vector DB...")
    vector_db_client.upsert(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk['text'], "source": chunk['metadata']['source']}
            ) for chunk, embedding in zip(all_chunks, embeddings)
        ],
        wait=True
    )
    print(f"--- Successfully indexed {len(all_chunks)} chunks from {filename}. ---\n")

# --- 4. RETRIEVAL AND GENERATION (UPGRADED) ---

def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """Builds the prompt for the LLM using the provided template."""
    
    evidence_str = ""
    for i, chunk in enumerate(retrieved_chunks):
        evidence_str += f"[{i+1}] source: {chunk['payload']['source']}\n"
        evidence_str += f"content: \"{chunk['payload']['text']}\"\n\n"

    # --- NEW AND IMPROVED PROMPT TEMPLATE v2 ---
    prompt_template = f"""
You are an expert assistant that answers queries using *only* the evidence listed below.
Each piece of evidence is numbered and has a source. When you state facts, you MUST append the citation number(s) in brackets, like [1] or [2][3].
Your answer must be grounded in the provided evidence.

IMPORTANT INSTRUCTIONS:
1. If you find conflicting information across different sources, do not say you cannot answer. Instead, you must state the conflict clearly by explaining what each source says.
2. After explaining the differences, provide a final summary conclusion. For example: "In summary, coverage depends on the specific policy. Policy X covers the item, while Policies Y and Z do not."

If the evidence does not contain the answer at all, you must reply: "I cannot answer this question based on the provided documents."

Query: "{query}"

Evidence:
{evidence_str}

Answer:
"""
    return prompt_template

def ask_question(query: str):
    """Performs the full RAG pipeline with Hybrid Search and Reranking."""
    
    # --- Part 1: Hybrid Search (Vector + Keyword) ---
    print("1. Performing Hybrid Search (Vector + Keyword)...")
    query_embedding = embedding_model.encode(query)

    # Vector search
    vector_hits = vector_db_client.search(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=40
    )
    
    # Keyword search (using the text index we created)
    keyword_hits = vector_db_client.scroll(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="text", 
                    match=models.MatchText(text=query)
                )
            ]
        ),
        limit=40
    )[0]
    
    # Merge and deduplicate results
    combined_hits = {hit.id: hit for hit in vector_hits + keyword_hits}
    if not combined_hits:
        print("No relevant documents found.")
        return
    
    # --- Part 2: Reranking ---
    print(f"2. Reranking {len(combined_hits)} candidates with Cross-Encoder...")
    
    # Prepare pairs for the reranker: (query, document_text)
    rerank_pairs = [[query, hit.payload['text']] for hit in combined_hits.values()]
    
    # Get scores from the reranker model
    scores = reranker_model.predict(rerank_pairs)
    
    # Add scores to the hits and sort
    for hit, score in zip(combined_hits.values(), scores):
        hit.score = score # Overwriting Qdrant score with the more accurate reranker score
        
    # Sort by the new reranker score in descending order
    reranked_hits = sorted(combined_hits.values(), key=lambda x: x.score, reverse=True)
    
    # Select the top 5 reranked results for the context
    top_k_reranked = reranked_hits[:5]

    # --- Part 3: Generation ---
    print("3. Building prompt with top 5 reranked results...")
    prompt = build_prompt(query, [hit.dict() for hit in top_k_reranked])
    
    print(f"4. Generating answer with Ollama ({OLLAMA_MODEL_NAME})...")
    try:
        response = ollama.generate(model=OLLAMA_MODEL_NAME, prompt=prompt, stream=False)
        
        print("\n" + "="*50)
        print("💡 ANSWER:")
        print("="*50)
        print(response['response'].strip())
        
        print("\n" + "="*50)
        print("📚 SOURCES CITED (after reranking):")
        print("="*50)
        for i, chunk in enumerate(top_k_reranked):
            print(f"[{i+1}] Source: {chunk.payload['source']}")
            print(f"   Rerank Score: {chunk.score:.4f}")
            print(f"   Snippet: \"{chunk.payload['text'][:150]}...\"")
        print("="*50 + "\n")

    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")

# --- 5. MAIN EXECUTION LOOP (NEW VERSION WITH FILE DIALOG) ---

if __name__ == "__main__":
    # --- Setup for File Dialog ---
    # We need to create a hidden root window to make the file dialog appear.
    try:
        root = tk.Tk()
        root.withdraw() # Hide the main window
    except tk.TclError:
        print("\n---")
        print("Warning: Could not initialize graphical environment for file dialog.")
        print("Please add files by typing the full path: add <path_to_file>")
        print("---\n")
        root = None
    # --- End of Setup ---

    help_message = """
    ===================================================================================
    Multimodal RAG System - Interactive Mode
    ===================================================================================
    Available Commands:
      add                     - Open a file dialog to select and index a document.
      
      add <path_to_file>      - Add and index a document by providing its full path.
                                (e.g., add C:\\Users\\YourName\\Documents\\report.pdf)
                                
      query <your_question>   - Ask a question about the indexed documents. 
                                (You can also just type your question without 'query')

      help                    - Show this help message.
      
      quit                    - Exit the program.
    ====================================================================================
    """
    print(help_message)
    
    while True:
        user_input = input("\nEnter command or query: ").strip()
        
        if not user_input:
            continue

        if user_input.lower().startswith('add'):
            parts = user_input.split(' ', 1)
            if len(parts) > 1:
                # This handles the 'add <path_to_file>' command
                file_path = parts[1].strip()
                # Remove quotes if user drags and drops file
                if file_path.startswith('"') and file_path.endswith('"'):
                    file_path = file_path[1:-1]
                process_and_index_single_file(file_path)
            elif root:
                # This handles the 'add' command by itself, opening the file dialog
                print("Opening file dialog to select a document...")
                
                # --- THIS IS THE FIX ---
                # Force the tkinter window to update before opening the dialog
                root.update() 
                
                file_path = filedialog.askopenfilename(
                    title="Select a document to index",
                    filetypes=(
                        ("All Supported Files", "*.pdf *.docx *.png *.jpg *.jpeg *.mp3 *.wav *.m4a"),
                        ("PDF Documents", "*.pdf"),
                        ("Word Documents", "*.docx"),
                        ("Image Files", "*.png *.jpg *.jpeg"),
                        ("Audio Files", "*.mp3 *.wav *.m4a"),
                        ("All files", "*.*")
                    )
                )
                if file_path:  # A file was selected
                    process_and_index_single_file(file_path)
                else:  # The dialog was cancelled
                    print("No file selected.")
            else:
                print("File dialog is not available. Please provide the full path: add <path_to_file>")


        elif user_input.lower() in ['quit', 'exit']:
            print("Exiting system...")
            break
        
        elif user_input.lower() == 'help':
            print(help_message)
        
        else:
            # Treat any other input as a query
            query = user_input
            if query.lower().startswith('query '):
                query = query[6:].strip()
            
            print(f"\nSearching indexed documents for: '{query}'")
            ask_question(query)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.query import answer_question
import os
import shutil
import fitz  # PyMuPDF
import faiss
import pickle
from sentence_transformers import SentenceTransformer

app = FastAPI(title="IRD Tax AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PDF_DIR = "data/pdfs"
VECTOR_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
META_FILE = os.path.join(VECTOR_DIR, "meta.pkl")
EMBEDDING_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"
MAX_CHUNK_SIZE = 200

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_tax_question(req: QuestionRequest):
    """Ask a tax question"""
    return answer_question(req.question)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a new PDF and automatically add it to the vector store
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF files are allowed"}
            )
        
        # Save the uploaded PDF
        pdf_path = os.path.join(PDF_DIR, file.filename)
        
        # Check if file already exists
        if os.path.exists(pdf_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"File '{file.filename}' already exists"}
            )
        
        # Save file
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF and add to vector store
        new_chunks = process_and_embed_pdf(pdf_path, file.filename)
        
        return {
            "message": "PDF uploaded and processed successfully",
            "filename": file.filename,
            "chunks_added": new_chunks
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing PDF: {str(e)}"}
        )

def process_and_embed_pdf(pdf_path: str, filename: str):
    """
    Process a single PDF and add its embeddings to existing vector store
    """
    # 1. Extract and chunk the PDF
    documents = []
    pdf = fitz.open(pdf_path)
    
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        
        words = text.split()
        for i in range(0, len(words), MAX_CHUNK_SIZE):
            chunk_words = words[i:i + MAX_CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            documents.append({
                "text": chunk_text,
                "source": filename,
                "page": page_num
            })
    
    if not documents:
        return 0
    
    # 2. Load existing vector store
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        existing_documents = pickle.load(f)
    
    # 3. Create embeddings for new documents
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [f"passage: {doc['text']}" for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    # 4. Add new embeddings to existing index
    index.add(embeddings)
    
    # 5. Append new documents to metadata
    existing_documents.extend(documents)
    
    # 6. Save updated vector store
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(existing_documents, f)
    
    return len(documents)

@app.get("/")
def read_root():
    """API information"""
    return {
        "message": "IRD Tax AI API",
        "status": "running",
        "endpoints": {
            "/ask": "POST - Ask a tax question",
            "/upload-pdf": "POST - Upload a new PDF document",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/stats")
def get_stats():
    """Get statistics about the vector store"""
    try:
        with open(META_FILE, "rb") as f:
            documents = pickle.load(f)
        
        # Get unique PDFs
        unique_pdfs = set(doc["source"] for doc in documents)
        
        return {
            "total_chunks": len(documents),
            "total_pdfs": len(unique_pdfs),
            "pdf_files": sorted(list(unique_pdfs))
        }
    except Exception as e:
        return {"error": str(e)}

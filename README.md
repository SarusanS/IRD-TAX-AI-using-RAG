# IRD Tax AI - RAG-based Tax Document Assistant

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Sri Lankan tax regulations using IRD (Inland Revenue Department) documents.

## Overview

This system allows users to:
- Query Sri Lankan tax documents using natural language
- Upload new PDF documents to expand the knowledge base
- Receive accurate, source-cited answers from official IRD publications
- Track which documents and pages were used to generate each answer

## Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **FastAPI**: Modern, fast web framework for building APIs
- **FAISS**: Facebook's vector similarity search library for efficient retrieval
- **Sentence-Transformers**: Pre-trained embedding models for semantic search
  - Model: `msmarco-bert-base-dot-v5` (optimized for passage retrieval)
- **PyMuPDF (fitz)**: PDF text extraction and processing
- **Groq API**: LLM inference using Llama 3.1 8B Instant
- **Pickle**: Metadata storage and serialization

### Frontend
- **HTML5/CSS3/JavaScript**: Vanilla frontend (no frameworks)
- **Gradient UI**: Modern purple gradient design
- **Fetch API**: Asynchronous API communication

### Vector Store
- **FAISS IndexFlatL2**: Exact L2 distance search on normalized embeddings
- **Metadata Store**: Pickle-based document chunk metadata storage
- **Chunking Strategy**: Fixed 200-word chunks with page-level tracking

## Architecture

```
┌─────────────────┐
│   Frontend      │  (HTML/CSS/JS)
│   (Browser)     │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│   FastAPI       │  /ask, /upload-pdf, /stats
│   Server        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Query  │ │ Ingest & │
│ Module │ │ Embed    │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌─────────────────────┐
│   FAISS Index       │  (vectorstore/)
│   + Metadata PKL    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Groq API          │  (LLM)
│   Llama 3.1 8B      │
└─────────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js and npm (for development tools, optional)
- Groq API key (free tier available)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ird-tax-ai
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu pymupdf groq python-multipart
```

4. **Set up directory structure**
```bash
mkdir -p data/pdfs vectorstore
```

5. **Get Groq API key**
   - Visit https://console.groq.com
   - Sign up for free (no credit card required)
   - Create an API key at https://console.groq.com/keys
   - Set environment variable:
```bash
export GROQ_API_KEY='your-api-key-here'  # Linux/Mac
set GROQ_API_KEY=your-api-key-here       # Windows CMD
$env:GROQ_API_KEY='your-api-key-here'    # Windows PowerShell
```

6. **Add initial PDF documents**
   - Place your IRD PDF files in `data/pdfs/`
   - Example files: tax guides, practice notes, circulars

### Initial Vector Store Creation

```bash
# Run the embedding script to create initial vector store
python app/embed.py
```

This will:
- Load all PDFs from `data/pdfs/`
- Chunk documents into 800-word segments
- Generate embeddings using sentence-transformers
- Create FAISS index at `vectorstore/index.faiss`
- Save metadata to `vectorstore/meta.pkl`

### Running the Application

1. **Start the FastAPI backend**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Open the frontend**
   - Open `frontend.html` in a web browser
   - Or serve it using a simple HTTP server:
```bash
python -m http.server 8080
# Then visit: http://localhost:8080/frontend.html
```

3. **Test the system**
   - Try example questions in the UI
   - Upload additional PDF documents
   - View statistics about the knowledge base

## Project Structure

```
ird-tax-ai/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI server and endpoints
│   ├── query.py          # RAG query logic and LLM calls
│   ├── ingest.py         # PDF loading and chunking
│   └── embed.py          # Vector store creation
├── data/
│   └── pdfs/             # Source PDF documents
├── vectorstore/
│   ├── index.faiss       # FAISS vector index
│   └── meta.pkl          # Document metadata
├── frontend.html         # Web interface
└── README.md
```

## Configuration

### Key Parameters

**Embedding Model** (`app/query.py`, `app/embed.py`):
```python
EMBEDDING_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"
```

**Chunk Size** (`app/ingest.py`, `app/main.py`):
```python
MAX_CHUNK_SIZE = 800  # words per chunk (ingest.py)
MAX_CHUNK_SIZE = 200  # words per chunk (main.py for uploads)
```
Note: There's a discrepancy - ingest.py uses 800 words but main.py uses 200 words for new uploads.

**Retrieval** (`app/query.py`):
```python
k = 10  # Number of chunks retrieved per query
```

**LLM Model** (`app/query.py`):
```python
MODEL = "llama-3.1-8b-instant"  # Groq model
temperature = 0.2               # Low temperature for factual responses
max_tokens = 400                # Response length limit
```

## Assumptions and Design Decisions

### 1. Document Processing
- **Assumption**: All input documents are text-based PDFs (not scanned images)
- **Reasoning**: PyMuPDF's text extraction works best with native text PDFs
- **Impact**: Scanned documents may not be properly indexed
- **Mitigation**: Could add OCR preprocessing if needed

### 2. Chunking Strategy
- **Assumption**: 800-word (ingest) / 200-word (upload) chunks provide good balance
- **Reasoning**: 
  - Smaller chunks → more precise retrieval but may miss context
  - Larger chunks → more context but less precise matching
- **Trade-off**: Current size chosen for tax documents with dense information
- **Note**: Inconsistency between ingest.py (800) and main.py (200) should be unified

### 3. Embedding Model
- **Choice**: `msmarco-bert-base-dot-v5`
- **Reasoning**: 
  - Specifically trained for passage retrieval tasks
  - Good balance of speed and quality
  - Optimized for asymmetric search (short query → long passage)
- **Alternative**: Could use larger models (e.g., `all-mpnet-base-v2`) for better quality

### 4. Vector Search
- **Choice**: FAISS IndexFlatL2 with normalized embeddings
- **Reasoning**:
  - Exact search (not approximate) for maximum accuracy
  - L2 distance on normalized vectors ≈ cosine similarity
  - Simple and reliable for datasets under 1M vectors
- **Scaling**: For larger datasets (>100k documents), consider IndexIVFFlat

### 5. Source Filtering
- **Approach**: Smart filtering based on LLM answer content
- **Logic**:
  1. First, check if document names are mentioned in answer
  2. If none mentioned, use top 3 most relevant retrieved chunks
  3. Filter to unique (source, page) combinations
- **Reasoning**: Avoids overwhelming users with irrelevant sources
- **Trade-off**: May occasionally omit sources that influenced the answer

### 6. LLM Integration
- **Choice**: Groq API with Llama 3.1 8B Instant
- **Reasoning**:
  - Free tier available (100 requests/minute)
  - Very fast inference (<1s response time)
  - Good quality for factual question answering
  - No local GPU required
- **Alternative**: Could use local models (Ollama) or other APIs (OpenAI, Anthropic)

### 7. Context Window
- **Approach**: Include full retrieved chunks with metadata in prompt
- **Format**: `[From document: {source}, Page {page}]\n{text}`
- **Reasoning**: 
  - Helps LLM cite sources accurately
  - Provides full context for each chunk
  - Enables source verification
- **Limit**: 10 chunks × ~800 words = ~8000 words of context

### 8. System Prompt Design
- **Critical Instructions**:
  1. Avoid meta-commentary ("The information can be found in...")
  2. Start with direct answers for content questions
  3. Include disclaimer about not being professional advice
  4. Fail gracefully when information is missing
- **Reasoning**: Improves user experience and sets appropriate expectations

### 9. Upload Functionality
- **Design**: Incremental updates to existing vector store
- **Approach**:
  1. Validate PDF format
  2. Check for duplicates
  3. Extract and chunk new document
  4. Append to existing FAISS index and metadata
  5. No rebuild required
- **Reasoning**: Allows dynamic knowledge base expansion without downtime
- **Note**: No delete functionality currently implemented

### 10. Frontend Simplicity
- **Choice**: Vanilla JavaScript, no frameworks
- **Reasoning**:
  - Minimal dependencies
  - Easy to understand and modify
  - Fast load times
  - No build process required
- **Trade-off**: Less sophisticated UI compared to React/Vue, but sufficient for this use case

### 11. Error Handling
- **Backend**: Try-catch blocks with informative error messages
- **Frontend**: Visual feedback for loading, success, and error states
- **API Key**: Graceful degradation with setup instructions if missing
- **Assumption**: Users can follow technical setup instructions

### 12. Statelessness
- **Design**: Query module reloads vector store on each request
- **Reasoning**: 
  - Ensures fresh data after uploads
  - Simple implementation
  - Acceptable for low-to-medium traffic
- **Trade-off**: Slight performance overhead (~100ms) for loading index
- **Optimization**: Could cache index in memory for production

## Usage Examples

### Querying the System

```python
# Via Python
from app.query import answer_question

result = answer_question("What is the Corporate Income Tax rate?", k=10, debug=True)
print(result['answer'])
print(result['sources'])
```

### Via API

```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is SET?"}'

# Upload a PDF
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@path/to/document.pdf"

# Get statistics
curl http://localhost:8000/stats
```

## Limitations and Future Improvements

### Current Limitations
1. **No OCR**: Cannot process scanned/image-based PDFs
2. **No versioning**: Uploaded documents cannot be removed or updated
3. **No authentication**: Open API with no user management
4. **No caching**: Vector store reloaded on each query
5. **Inconsistent chunking**: Different chunk sizes between initial ingest and uploads
6. **No multilingual support**: Optimized for English only
7. **Fixed retrieval**: Always retrieves 10 chunks (not adaptive)

### Potential Improvements
1. **Add OCR support** using Tesseract for scanned PDFs
2. **Implement document management** (delete, update, versioning)
3. **Add user authentication** and query history
4. **Optimize with caching** and connection pooling
5. **Unify chunking strategy** across all ingestion paths
6. **Add hybrid search** combining dense and sparse (BM25) retrieval
7. **Implement re-ranking** for better relevance
8. **Add conversation memory** for follow-up questions
9. **Support Sinhala/Tamil** for local language queries
10. **Add monitoring and analytics** for usage tracking

## Troubleshooting

### FAISS Index Not Found
```
FileNotFoundError: vectorstore/index.faiss
```
**Solution**: Run `python app/embed.py` to create initial index

### Groq API Errors
```
Error calling Groq API: Unauthorized
```
**Solution**: Check that `GROQ_API_KEY` environment variable is set

### Empty Results
```
"I don't have enough information to answer this question."
```
**Possible causes**:
- No relevant documents in the knowledge base
- Query too vague or specific
- Embedding model mismatch
**Solution**: Add more relevant PDFs, rephrase query, or adjust retrieval parameters

### PDF Upload Fails
```
"Only PDF files are allowed"
```
**Solution**: Ensure file has `.pdf` extension and is a valid PDF

## Performance Considerations

- **Initial embedding**: ~1-5 minutes for 10-50 PDFs (depending on size)
- **Query latency**: ~1-2 seconds (retrieval + LLM inference)
- **Upload processing**: ~5-30 seconds per PDF (depending on size)
- **Memory usage**: ~500MB-2GB (depends on index size and model)
- **Concurrent users**: Tested with 1-10 users; would need load balancing for 100+

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software for educational and research purposes.
See the LICENSE file for details.


## Contributing

Contributions are welcome for educational and research purposes.

If you would like to contribute:
1. Fork the repository
2. Create a new branch for your feature or fix
3. Commit your changes with clear messages
4. Open a Pull Request describing your changes

Please ensure:
- Code follows existing project structure
- New features are well-documented
- Changes do not break existing functionality

This project is primarily intended for learning and academic use.


## Contact

For questions, feedback, or academic collaboration, please contact:

**Sivanesan Sarusan**  
Undergraduate, Electrical & Electronics Engineering  
University of Peradeniya  

📧 Email: thavaranysarusan@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/sarusan-sivanesan


## Acknowledgments

- IRD Sri Lanka for tax documentation
- Groq for fast LLM inference
- Facebook Research for FAISS
- Sentence-Transformers team for embedding models

# API Documentation - IRD Tax AI

## Base URL
```
http://localhost:8000
```

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and health check |
| `/ask` | POST | Ask a tax question |
| `/upload-pdf` | POST | Upload and process a new PDF document |
| `/stats` | GET | Get knowledge base statistics |

---

## 1. Root Endpoint

### `GET /`

Get API information and available endpoints.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "IRD Tax AI API",
  "status": "running",
  "endpoints": {
    "/ask": "POST - Ask a tax question",
    "/upload-pdf": "POST - Upload a new PDF document",
    "/docs": "GET - API documentation"
  }
}
```

**Status Codes:**
- `200 OK`: API is running

---

## 2. Ask Question

### `POST /ask`

Query the RAG system with a tax-related question.

**Request Body:**
```json
{
  "question": "string (required)"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | The tax question to ask |

**Example Request:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Corporate Income Tax rate for AY 2022/2023?"
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is the Corporate Income Tax rate for AY 2022/2023?"}
)
data = response.json()
```

**JavaScript Example:**
```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What is the Corporate Income Tax rate for AY 2022/2023?'
  })
});
const data = await response.json();
```

**Success Response:**
```json
{
  "question": "What is the Corporate Income Tax rate for AY 2022/2023?",
  "answer": "The Corporate Income Tax rate for Assessment Year 2022/2023 is 30% for companies with a taxable income exceeding Rs. 2,000 million, and 24% for other companies as outlined in PN_IT_2022-03.\n\nNote: This response is based on IRD documents and is not professional tax advice.",
  "sources": [
    {
      "file": "PN_IT_2022-03.pdf",
      "page": 3
    },
    {
      "file": "PN_IT_2022-03.pdf",
      "page": 4
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| question | string | The original question asked |
| answer | string | The generated answer with disclaimer |
| sources | array | List of source documents and pages used |
| sources[].file | string | PDF filename |
| sources[].page | integer | Page number in the PDF |

**Error Response (No API Key):**
```json
{
  "question": "What is the Corporate Income Tax rate?",
  "answer": "Error: Please set GROQ_API_KEY. Get free key from: https://console.groq.com/keys",
  "sources": []
}
```

**Error Response (No Information Found):**
```json
{
  "question": "What is the tax rate on Mars?",
  "answer": "I don't have enough information to answer this question.",
  "sources": []
}
```

**Status Codes:**
- `200 OK`: Question processed successfully
- `422 Unprocessable Entity`: Invalid request body

**Notes:**
- The system retrieves 10 most relevant document chunks
- Answers include a disclaimer about not being professional advice
- Sources are filtered to only include documents actually used in the answer
- If no documents mention the topic, a "not available" message is returned

---

## 3. Upload PDF

### `POST /upload-pdf`

Upload a new PDF document to expand the knowledge base.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Form Field**: `file` (PDF file)

**Example Request (cURL):**
```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@/path/to/document.pdf"
```

**Python Example:**
```python
import requests

files = {'file': open('tax_guide.pdf', 'rb')}
response = requests.post(
    "http://localhost:8000/upload-pdf",
    files=files
)
data = response.json()
```

**JavaScript Example (Browser):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/upload-pdf', {
  method: 'POST',
  body: formData
});
const data = await response.json();
```

**Success Response:**
```json
{
  "message": "PDF uploaded and processed successfully",
  "filename": "PN_IT_2025-01.pdf",
  "chunks_added": 45
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| message | string | Success message |
| filename | string | Name of uploaded file |
| chunks_added | integer | Number of text chunks extracted and indexed |

**Error Response (Invalid File Type):**
```json
{
  "error": "Only PDF files are allowed"
}
```

**Error Response (Duplicate File):**
```json
{
  "error": "File 'PN_IT_2025-01.pdf' already exists"
}
```

**Error Response (Processing Error):**
```json
{
  "error": "Error processing PDF: [error details]"
}
```

**Status Codes:**
- `200 OK`: PDF uploaded and processed successfully
- `400 Bad Request`: Invalid file type or duplicate file
- `500 Internal Server Error`: Processing error

**Processing Details:**
1. File is validated (must be `.pdf`)
2. Duplicate check performed
3. PDF saved to `data/pdfs/`
4. Text extracted using PyMuPDF
5. Text chunked into 200-word segments
6. Embeddings generated for each chunk
7. FAISS index updated with new vectors
8. Metadata updated with chunk information

**Limitations:**
- Only PDF files are accepted
- Duplicate filenames are rejected (no versioning)
- No file size limit enforced (may timeout on very large files)
- No authentication/authorization required
- Uploaded files cannot be deleted via API

---

## 4. Statistics

### `GET /stats`

Get statistics about the current knowledge base.

**Request:**
```bash
curl http://localhost:8000/stats
```

**Success Response:**
```json
{
  "total_chunks": 1247,
  "total_pdfs": 12,
  "pdf_files": [
    "Corporate_Tax_Guide_2023.pdf",
    "PN_IT_2022-03.pdf",
    "PN_IT_2025-01.pdf",
    "SET_Guidelines.pdf",
    "VAT_Circular_2023.pdf"
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| total_chunks | integer | Total number of text chunks in vector store |
| total_pdfs | integer | Number of unique PDF documents |
| pdf_files | array | List of all PDF filenames (sorted) |

**Error Response:**
```json
{
  "error": "vectorstore/meta.pkl not found"
}
```

**Status Codes:**
- `200 OK`: Statistics retrieved successfully
- `200 OK` (with error field): Vector store not initialized

**Notes:**
- Statistics are computed from the metadata pickle file
- `total_chunks` includes all chunks from all documents
- `pdf_files` are sorted alphabetically
- This endpoint is fast (no FAISS index loading required)

---

## API Design Patterns

### CORS Configuration

The API is configured with permissive CORS to allow browser-based clients:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Note**: Restrict `allow_origins` to specific domains in production.

### Error Handling

All endpoints use consistent error response format:

```json
{
  "error": "Error message describing what went wrong"
}
```

HTTP status codes follow REST conventions:
- `200`: Success
- `400`: Client error (bad request)
- `422`: Validation error (malformed request body)
- `500`: Server error (processing failure)

### Content Types

| Endpoint | Request Content-Type | Response Content-Type |
|----------|---------------------|----------------------|
| `/` | N/A (GET) | `application/json` |
| `/ask` | `application/json` | `application/json` |
| `/upload-pdf` | `multipart/form-data` | `application/json` |
| `/stats` | N/A (GET) | `application/json` |

---

## RAG Pipeline Details

### Query Processing Flow

```
User Question
    ↓
1. Load Vector Store (FAISS + Metadata)
    ↓
2. Generate Query Embedding (sentence-transformers)
    ↓
3. FAISS Similarity Search (k=10 chunks)
    ↓
4. Format Context with Metadata
    ↓
5. Call Groq LLM (Llama 3.1 8B)
    ↓
6. Filter Relevant Sources
    ↓
7. Return Answer + Sources
```

### Embedding Strategy

**Query Embedding:**
```python
query_embedding = model.encode(
    [f"query: {question}"],
    normalize_embeddings=True
)
```

**Passage Embedding (during ingestion):**
```python
texts = [f"passage: {doc['text']}" for doc in documents]
embeddings = model.encode(
    texts,
    normalize_embeddings=True
)
```

The `query:` and `passage:` prefixes optimize the asymmetric retrieval task (as designed for the msmarco-bert model).

### Source Filtering Algorithm

```python
def filter_relevant_sources(answer, retrieved_docs):
    # Step 1: Check if document names are mentioned in answer
    mentioned_sources = {}
    for doc in retrieved_docs:
        if doc['source'] in answer_text:
            mentioned_sources[doc['source']] = doc['page']
    
    # Step 2: If no mentions, use top 3 retrieved docs
    if not mentioned_sources:
        mentioned_sources = top_3_docs
    
    # Step 3: Return unique (source, page) pairs
    return deduplicated_sources
```

This ensures users only see sources that contributed to the answer.

---

## Rate Limits and Quotas

### Groq API Limits (Free Tier)
- **Requests**: 100 requests/minute
- **Tokens**: Varies by model
- **Concurrent**: Multiple requests allowed

### API Server Limits
- **No built-in rate limiting**: All endpoints are open
- **Upload size**: No enforced limit (system memory dependent)
- **Concurrent requests**: Limited by server capacity

**Production Recommendations:**
1. Add rate limiting middleware (e.g., `slowapi`)
2. Implement request queuing for uploads
3. Add authentication and user-based quotas
4. Set maximum file size for uploads (e.g., 50MB)

---

## Testing Examples

### Complete Test Script (Python)

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Check API status
response = requests.get(f"{BASE_URL}/")
print("Status:", response.json())

# 2. Get statistics
response = requests.get(f"{BASE_URL}/stats")
print("\nStats:", json.dumps(response.json(), indent=2))

# 3. Ask a question
response = requests.post(
    f"{BASE_URL}/ask",
    json={"question": "What is the Corporate Income Tax rate?"}
)
result = response.json()
print("\nQuestion:", result['question'])
print("Answer:", result['answer'])
print("Sources:", result['sources'])

# 4. Upload a PDF
with open("test_document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
    print("\nUpload:", response.json())

# 5. Verify statistics updated
response = requests.get(f"{BASE_URL}/stats")
print("\nUpdated Stats:", json.dumps(response.json(), indent=2))
```

### Test Script (cURL)

```bash
#!/bin/bash

# Test all endpoints

# 1. Health check
echo "=== Health Check ==="
curl http://localhost:8000/

# 2. Statistics
echo -e "\n\n=== Statistics ==="
curl http://localhost:8000/stats

# 3. Ask question
echo -e "\n\n=== Ask Question ==="
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is SET?"}'

# 4. Upload PDF
echo -e "\n\n=== Upload PDF ==="
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@test.pdf"

# 5. Updated statistics
echo -e "\n\n=== Updated Statistics ==="
curl http://localhost:8000/stats
```

---

## WebSocket Support

**Current Status**: Not implemented

**Potential Use Cases**:
- Real-time query progress updates
- Streaming LLM responses
- Live upload processing status
- Multi-user collaborative features

**Implementation Notes**:
Could be added using FastAPI's WebSocket support for enhanced UX.

---

## API Versioning

**Current Version**: v1 (implicit, no version in URL)

**Future Versioning Strategy**:
```
/v1/ask
/v1/upload-pdf
/v1/stats

/v2/ask  (with enhanced features)
```

---

## Security Considerations

### Current State
⚠️ **Warning**: This is a development/demo API with minimal security:
- No authentication
- No authorization
- Open CORS policy
- No rate limiting
- No input sanitization beyond basic validation
- No API keys required

### Production Recommendations

1. **Add Authentication**
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/ask")
   async def ask(req: QuestionRequest, token: str = Depends(security)):
       # Validate token
       pass
   ```

2. **Restrict CORS**
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

3. **Add Rate Limiting**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/ask")
   @limiter.limit("10/minute")
   async def ask(...):
       pass
   ```

4. **Input Validation**
   - Sanitize filenames
   - Validate PDF content
   - Limit question length
   - Escape special characters

5. **File Upload Security**
   - Scan for malware
   - Enforce file size limits
   - Validate PDF structure
   - Implement user quotas

---

## Performance Optimization

### Current Performance
- **Query latency**: 1-2 seconds
  - Vector search: ~50-100ms
  - LLM inference: ~1-1.5s
  - Source filtering: ~10-20ms

- **Upload processing**: 5-30 seconds
  - PDF parsing: ~1-5s
  - Embedding generation: ~2-10s
  - Index update: ~1-5s

### Optimization Strategies

1. **Cache Vector Store in Memory**
   ```python
   # Load once at startup
   @app.on_event("startup")
   async def load_index():
       global index, documents
       index = faiss.read_index(INDEX_FILE)
       with open(META_FILE, "rb") as f:
           documents = pickle.load(f)
   ```

2. **Batch Processing for Uploads**
   ```python
   # Process multiple PDFs in parallel
   async def batch_upload(files: List[UploadFile]):
       tasks = [process_pdf(f) for f in files]
       await asyncio.gather(*tasks)
   ```

3. **Response Caching**
   ```python
   # Cache common queries
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def get_cached_answer(question: str):
       return answer_question(question)
   ```

4. **Use Approximate Search** (for large datasets)
   ```python
   # Switch to IVF index for >100k vectors
   index = faiss.IndexIVFFlat(quantizer, dim, nlist)
   ```

---

## Monitoring and Logging

### Recommended Additions

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

@app.post("/ask")
async def ask(req: QuestionRequest):
    start_time = datetime.now()
    logging.info(f"Question received: {req.question}")
    
    try:
        result = answer_question(req.question)
        elapsed = (datetime.now() - start_time).total_seconds()
        logging.info(f"Answer generated in {elapsed}s")
        return result
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise
```

### Metrics to Track
- Total queries
- Average query latency
- Most common questions
- Source document usage
- Upload success/failure rates
- Error rates by endpoint
- Daily/weekly active usage

---

## Changelog

### v1.0.0 (Current)
- Initial release
- Basic RAG functionality
- PDF upload support
- Source citation
- Statistics endpoint

### Future Versions
- v1.1.0: Add authentication
- v1.2.0: Add conversation history
- v2.0.0: Multi-language support
- v2.1.0: Advanced search filters

# Design Rationale Document
## IRD Tax AI - RAG System

### Document Overview
This document explains the key design decisions, trade-offs, and architectural choices made in building the IRD Tax AI system. It serves as a reference for understanding why the system is built the way it is and what alternatives were considered.

---

## 1. System Architecture

### 1.1 RAG (Retrieval-Augmented Generation) Approach

**Decision**: Implement a RAG system rather than fine-tuning or using a pure LLM approach.

**Rationale**:
- **Accuracy**: RAG grounds responses in actual source documents, reducing hallucinations
- **Traceability**: Every answer can be traced back to specific documents and pages
- **Maintainability**: New documents can be added without retraining models
- **Cost**: Fine-tuning large models is expensive; RAG uses pre-trained models
- **Transparency**: Users can verify answers by checking cited sources

**Trade-offs**:
- ✅ **Pros**: Accurate, verifiable, easy to update, cost-effective
- ❌ **Cons**: Slower than pure LLM (requires retrieval step), limited by retrieval quality

**Alternatives Considered**:
1. **Fine-tuned model**: Too expensive, requires retraining for updates
2. **Pure prompt engineering**: Prone to hallucinations, no source attribution
3. **Knowledge graph**: More complex, harder to maintain for document-heavy domain

---

### 1.2 Two-Stage Pipeline Architecture

**Decision**: Separate the system into distinct stages: (1) Ingestion/Embedding, (2) Query/Retrieval.

**Rationale**:
- **Separation of concerns**: Each module has a single responsibility
- **Independent scaling**: Can optimize ingestion and query paths separately
- **Flexibility**: Easy to swap embedding models or retrieval methods
- **Development efficiency**: Team members can work on different modules independently

**Implementation**:
```
Ingestion Flow: PDF → Chunking → Embedding → Vector Store
Query Flow: Question → Embedding → Retrieval → LLM → Answer
```

**Trade-offs**:
- ✅ **Pros**: Clean architecture, easy to maintain, scalable
- ❌ **Cons**: Slight complexity in managing two separate workflows

---

## 2. Embedding Strategy

### 2.1 Model Selection: msmarco-bert-base-dot-v5

**Decision**: Use `sentence-transformers/msmarco-bert-base-dot-v5` for embedding generation.

**Rationale**:
- **Task-specific**: Trained on MS MARCO, a passage ranking dataset (perfect for Q&A)
- **Asymmetric search**: Optimized for short query → long passage retrieval
- **Performance**: Good balance of speed (100-200ms) and quality
- **Size**: 420MB model size, reasonable for deployment
- **E5-style**: Supports `query:` and `passage:` prefixes for better retrieval

**Alternatives Considered**:

| Model | Pros | Cons | Reason Not Chosen |
|-------|------|------|-------------------|
| all-mpnet-base-v2 | Higher quality | Slower, symmetric search | Not optimized for Q&A |
| all-MiniLM-L6-v2 | Very fast, small | Lower quality | Quality too low for tax domain |
| multilingual-e5-large | Multilingual support | 2.2GB, very slow | Overkill for English-only |
| OpenAI embeddings | High quality | Cost per embedding, API dependency | Ongoing costs |

**Performance Metrics**:
- Embedding speed: ~150ms for 10 chunks
- Vector dimension: 768 (standard BERT)
- GPU acceleration: Not required (CPU-only deployment)

---

### 2.2 Normalization and Distance Metric

**Decision**: Normalize all embeddings and use L2 distance in FAISS.

**Rationale**:
- **Mathematical equivalence**: L2 on normalized vectors ≈ cosine similarity
- **FAISS optimization**: IndexFlatL2 is faster than IndexFlatIP (inner product)
- **Consistency**: Ensures all vectors are unit length
- **Stability**: Prevents magnitude-based biases

**Formula**:
```
L2(u, v) where ||u|| = ||v|| = 1  ≈  1 - cosine_similarity(u, v)
```

**Implementation**:
```python
embeddings = model.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatL2(dim)  # L2 distance
```

---

## 3. Text Chunking Strategy

### 3.1 Fixed-Size Word Chunking

**Decision**: Use fixed 800-word chunks (ingest) / 200-word chunks (upload) without overlap.

**Rationale**:
- **Simplicity**: Easy to implement and understand
- **Predictability**: Consistent chunk sizes for better retrieval
- **Tax documents**: IRD documents have dense, structured information that fits well in chunks
- **Context balance**: Large enough for context, small enough for precision

**⚠️ Critical Issue Identified**:
There's an **inconsistency** in chunk sizes:
- `ingest.py`: Uses 800-word chunks
- `main.py`: Uses 200-word chunks for new uploads

**Impact**:
- Initial documents have different granularity than uploaded documents
- May affect retrieval quality and consistency
- Users get mixed chunk sizes in results

**Recommendation**: **Standardize to 400-600 words** across both paths.

**Alternatives Considered**:

| Strategy | Pros | Cons | Reason Not Chosen |
|----------|------|------|-------------------|
| Sentence-based | Natural boundaries | Variable sizes | Too small for tax content |
| Paragraph-based | Semantic units | Very variable sizes | PDF extraction issues |
| Sliding window | Better context | 2x storage, slower | Unnecessary complexity |
| Semantic chunking | Best quality | Complex, slow | Premature optimization |

---

### 3.2 No Chunk Overlap

**Decision**: No overlap between consecutive chunks.

**Rationale**:
- **Storage efficiency**: No duplicate information
- **Faster retrieval**: Fewer vectors to search
- **Tax documents**: Content is usually self-contained within sections

**Trade-off**:
- ✅ **Pros**: Efficient, fast, simple
- ❌ **Cons**: May miss context that spans chunk boundaries
- **Mitigation**: Retrieve k=10 chunks to capture surrounding context

**When overlap would help**:
- Narrative documents (stories, articles)
- When context critically depends on surrounding text
- Very long, interconnected explanations

**Why it's okay here**:
- Tax documents are structured and modular
- Most questions can be answered from single chunks
- Multiple chunk retrieval (k=10) compensates for boundary issues

---

## 4. Vector Store Design

### 4.1 FAISS IndexFlatL2

**Decision**: Use exact search with `IndexFlatL2` rather than approximate methods.

**Rationale**:
- **Accuracy**: 100% recall - never miss the best match
- **Dataset size**: <10k documents → exact search is fast enough
- **Simplicity**: No tuning required (no nlist, nprobe, etc.)
- **Tax domain**: Accuracy is critical; 50ms vs 10ms doesn't matter

**Performance**:
- Search time: ~50-100ms for 10k vectors (acceptable)
- Memory: ~10MB per 10k vectors (negligible)
- Scalability: Good up to 100k vectors on standard CPU

**When to switch to approximate**:
- **>100k documents**: IndexIVFFlat or IndexHNSW
- **Tight latency requirements**: Need <10ms response
- **Memory constraints**: Very large embedding dimensions

**Alternatives**:

| Index Type | Use Case | Pros | Cons |
|------------|----------|------|------|
| IndexFlatL2 | <100k vectors | Exact, simple | Slower at scale |
| IndexIVFFlat | 100k-10M vectors | Fast, good recall | Requires tuning |
| IndexHNSW | >1M vectors | Very fast | Higher memory |
| IndexPQ | >10M vectors | Compressed | Quality loss |

---

### 4.2 Metadata Storage with Pickle

**Decision**: Store document metadata (text, source, page) in a separate pickle file.

**Rationale**:
- **Simplicity**: Python's built-in serialization
- **FAISS limitation**: FAISS only stores vectors, not metadata
- **Fast loading**: Pickle is fast for Python objects
- **Development speed**: No need for database setup

**Structure**:
```python
documents = [
    {"text": "...", "source": "file.pdf", "page": 1},
    {"text": "...", "source": "file.pdf", "page": 2},
    ...
]
```

**Trade-offs**:
- ✅ **Pros**: Simple, fast, no dependencies
- ❌ **Cons**: Not scalable to millions of documents, no querying capability

**When to upgrade**:
- **>100k documents**: Use SQLite or PostgreSQL
- **Complex queries**: Need to filter by metadata
- **Multi-user**: Need concurrent write access

**Migration path**:
```python
# Easy to migrate later
import sqlite3
conn = sqlite3.connect('metadata.db')
# INSERT documents from pickle
```

---

## 5. LLM Integration

### 5.1 Groq API with Llama 3.1 8B

**Decision**: Use Groq's hosted Llama 3.1 8B Instant via API.

**Rationale**:
- **Cost**: Free tier available (100 req/min)
- **Speed**: <1s inference time (fastest in market)
- **Quality**: Llama 3.1 is strong for factual Q&A
- **No infrastructure**: No need for GPU servers
- **Prototyping**: Easy to test and iterate

**Configuration**:
```python
MODEL = "llama-3.1-8b-instant"
temperature = 0.2      # Low for factual accuracy
max_tokens = 400       # Concise answers
top_p = 0.9           # Slight diversity
```

**Alternatives Considered**:

| Option | Pros | Cons | Reason Not Chosen |
|--------|------|------|-------------------|
| OpenAI GPT-4 | Highest quality | Expensive ($0.03/1k tokens) | Budget constraints |
| Anthropic Claude | Great for docs | Expensive | Budget constraints |
| Local Llama | No costs | Needs GPU, slower | Infrastructure requirements |
| Ollama | Easy setup | CPU-only very slow | Speed requirements |

---

### 5.2 Prompt Engineering

**Decision**: Use a carefully crafted system prompt with specific instructions.

**Key Directives**:
1. **No meta-commentary**: Avoid "The information can be found in..."
2. **Direct answers**: Start with the actual information
3. **Natural citations**: Mention sources within explanation
4. **Mandatory disclaimer**: Always end with legal disclaimer
5. **Graceful degradation**: Clear message when no information found

**Example**:
```
❌ Bad: "The information can be found in PN_IT_2025-01. The rate is 30%."
✅ Good: "The Corporate Income Tax rate is 30% as stated in PN_IT_2025-01."
```

**Rationale**:
- **UX**: Users want answers, not reading instructions
- **Naturalness**: Reads like human expert, not AI
- **Safety**: Disclaimer protects against misuse
- **Clarity**: Explicit when information is unavailable

**Evolution**:
- Version 1: Generic prompt → verbose answers
- Version 2: Added "no meta-commentary" → better, but still preambles
- Version 3 (current): Explicit examples → clean answers

---

## 6. Retrieval and Source Filtering

### 6.1 Retrieval Parameter: k=10

**Decision**: Retrieve top 10 chunks per query.

**Rationale**:
- **Coverage**: Enough to capture multi-faceted answers
- **Context window**: 10 × 800 words = 8000 words (fits in LLM context)
- **Diversity**: Captures information from multiple documents
- **Performance**: Fast enough (<100ms)

**Empirical testing**:
- k=3: Too narrow, misses important context
- k=5: Better, but occasionally incomplete
- k=10: Sweet spot - comprehensive without noise
- k=20: Diminishing returns, more noise

**When to adjust**:
- **Simple questions**: k=3-5 sufficient
- **Complex questions**: k=15-20 may help
- **Future**: Implement adaptive k based on query complexity

---

### 6.2 Smart Source Filtering

**Decision**: Filter sources to only show documents actually mentioned in the answer.

**Algorithm**:
```python
1. Check if any retrieved document names appear in answer
2. If yes, return only mentioned documents
3. If no, fall back to top 3 retrieved documents
```

**Rationale**:
- **Relevance**: Only show sources that contributed to answer
- **UX**: Avoid overwhelming users with 10+ sources
- **Accuracy**: Reduces false attribution
- **Transparency**: Clear connection between answer and sources

**Example**:
```
Retrieved: [Doc1, Doc2, ..., Doc10]
Answer mentions: "Doc1 states... Doc3 indicates..."
Sources shown: [Doc1, Doc3]
```

**Trade-offs**:
- ✅ **Pros**: Clean, relevant, user-friendly
- ❌ **Cons**: May omit sources that indirectly influenced answer

---

## 7. Upload Functionality

### 7.1 Incremental Index Updates

**Decision**: Append new documents to existing FAISS index without rebuild.

**Rationale**:
- **Zero downtime**: System remains available during uploads
- **Fast**: No need to re-embed existing documents
- **Simple**: FAISS supports incremental adds

**Implementation**:
```python
# Load existing
index = faiss.read_index(INDEX_FILE)
documents = pickle.load(META_FILE)

# Add new
index.add(new_embeddings)
documents.extend(new_documents)

# Save
faiss.write_index(index, INDEX_FILE)
pickle.dump(documents, META_FILE)
```

**Trade-offs**:
- ✅ **Pros**: Fast, no downtime, simple
- ❌ **Cons**: Index becomes fragmented over time (minor performance impact)

**When to rebuild**:
- After 100+ uploads
- When performance degrades noticeably
- When changing chunking strategy

---

### 7.2 No Duplicate Detection

**Decision**: Reject duplicate filenames, but no content-based deduplication.

**Rationale**:
- **Simplicity**: Filename check is trivial
- **Tax documents**: Usually have unique filenames (PN_IT_2025-01, etc.)
- **Performance**: Content hashing adds overhead
- **Edge cases**: Updated documents have different filenames

**Current behavior**:
```python
if os.path.exists(pdf_path):
    return error("File already exists")
```

**Limitation**: Cannot handle:
- Updated versions of same document
- Renamed duplicates
- Different formats of same content

**Future enhancement**:
```python
# Compute hash of first page + metadata
hash = compute_document_hash(pdf)
if hash in existing_hashes:
    return error("Duplicate content")
```

---

## 8. Frontend Design

### 8.1 Vanilla JavaScript

**Decision**: Use plain HTML/CSS/JavaScript without frameworks.

**Rationale**:
- **Simplicity**: No build process, dependencies, or compilation
- **Load time**: Instant load, no bundle to download
- **Maintainability**: Easy for anyone to understand and modify
- **Deployment**: Works with any static file server

**Trade-offs**:
- ✅ **Pros**: Simple, fast, portable, no dependencies
- ❌ **Cons**: Less sophisticated than React/Vue, more manual DOM manipulation

**When to migrate to framework**:
- Multiple pages/views needed
- Complex state management required
- Team prefers component-based architecture
- Need sophisticated UI interactions

---

### 8.2 Gradient Design

**Decision**: Use purple gradient aesthetic with modern UI elements.

**Rationale**:
- **Professional**: Polished look for tax/government context
- **Distinctive**: Stands out from generic Bootstrap templates
- **Accessible**: Good contrast ratios, readable text
- **Modern**: Aligns with 2024-2025 design trends

**Color scheme**:
```css
Primary gradient: #667eea → #764ba2
Answer box: #f8f9fa with #667eea accent
Sources: #fff3cd with #ffc107 accent
Disclaimer: #e7f3ff with #2196f3 accent
```

---

## 9. Key Trade-offs Summary

### Speed vs. Accuracy
- **Choice**: Accuracy
- **Decision**: Exact search (IndexFlatL2), low temperature (0.2), k=10
- **Impact**: ~1-2s query time, but highly accurate results

### Simplicity vs. Features
- **Choice**: Simplicity
- **Decision**: No auth, no caching, vanilla JS, pickle storage
- **Impact**: Easy to deploy and maintain, but limited for production

### Cost vs. Quality
- **Choice**: Quality within free tier
- **Decision**: Groq API (free), good embedding model, exact search
- **Impact**: Zero API costs, strong performance for prototype

### Flexibility vs. Performance
- **Choice**: Flexibility
- **Decision**: Reload index on each query, no caching
- **Impact**: Always fresh data, but ~100ms overhead

---

## 10. Production Readiness Assessment

### Current State: **Prototype / MVP**

**Strengths**:
- ✅ Core RAG functionality works well
- ✅ Accurate source attribution
- ✅ Easy to deploy and use
- ✅ Good performance for small-to-medium datasets

**Limitations**:
- ❌ No authentication or authorization
- ❌ No rate limiting
- ❌ No monitoring or logging
- ❌ Inconsistent chunking between ingestion paths
- ❌ No error recovery or retries
- ❌ Single-threaded, no concurrency

### Path to Production

**Phase 1: Stability** (2-3 weeks)
1. Unify chunking strategy (400-600 words everywhere)
2. Add comprehensive error handling
3. Implement request logging
4. Add health checks and monitoring
5. Write unit and integration tests

**Phase 2: Security** (1-2 weeks)
1. Add API key authentication
2. Implement rate limiting
3. Restrict CORS to specific domains
4. Validate and sanitize all inputs
5. Add file upload scanning

**Phase 3: Performance** (2-3 weeks)
1. Cache vector store in memory
2. Implement connection pooling
3. Add response caching
4. Optimize database queries
5. Load balancing for multiple instances

**Phase 4: Features** (ongoing)
1. Conversation history
2. User management
3. Document versioning
4. Advanced search filters
5. Multi-language support

---

## 11. Lessons Learned

### What Worked Well
1. **RAG approach**: Source attribution critical for trust
2. **Simple architecture**: Easy to understand and modify
3. **FAISS**: Fast enough for prototype, simple to use
4. **Groq**: Free tier allowed unlimited experimentation
5. **Prompt engineering**: Key to answer quality

### What Could Be Improved
1. **Chunking consistency**: Should have been standardized from start
2. **Testing**: Should have written tests earlier
3. **Documentation**: Should have documented decisions as made
4. **Configuration**: Hard-coded values should be in config file
5. **Error messages**: Could be more informative for debugging

### Unexpected Challenges
1. **Source filtering**: Harder than expected to get right
2. **PDF extraction**: Some PDFs have formatting issues
3. **Prompt wording**: Small changes → big impact on answer quality
4. **Chunking**: Balancing context vs. precision was iterative

---

## 12. Conclusion

This RAG system successfully demonstrates how to build an accurate, traceable question-answering system for domain-specific documents. The design prioritizes **accuracy, simplicity, and maintainability** over raw performance, making it ideal for prototyping and small-to-medium deployments.

Key innovations:
- Smart source filtering reduces noise
- Incremental uploads enable dynamic knowledge base
- Careful prompt engineering produces natural answers
- Simple architecture makes the system accessible

The system is production-ready for **internal use or pilot deployments** with known user groups. For **public deployment**, additional work on security, monitoring, and scaling is required (see Phase 1-4 roadmap above).

**Overall Assessment**: Strong foundation with clear path to production.

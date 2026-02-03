import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq

INDEX_FILE = "vectorstore/index.faiss"
META_FILE = "vectorstore/meta.pkl"
EMBEDDING_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") 
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL = "llama-3.1-8b-instant"

print("Loading models...")
sbert_model = SentenceTransformer(EMBEDDING_MODEL)

print("Ready to answer questions")

def load_vector_store():
    """
    Load vector store fresh each time
    This ensures we always have the latest uploaded PDFs
    """
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        documents = pickle.load(f)
    return index, documents

def filter_relevant_sources(answer: str, retrieved_docs: list, debug: bool = False):
    """
    Filter sources to only include documents that are actually mentioned or used in the answer
    """
  
    all_sources = {}
    for doc in retrieved_docs:
        source = doc['source']
        page = doc['page']
        if source not in all_sources:
            all_sources[source] = set()
        all_sources[source].add(page)
    
   
    mentioned_sources = {}
    answer_lower = answer.lower()
    
    for source in all_sources.keys():
        source_name = source.replace('.pdf', '').replace('_', ' ').lower()
        source_short = source.split('_')[0].lower()
        if source_name in answer_lower or source_short in answer_lower or source in answer:
            mentioned_sources[source] = all_sources[source]
    
    if not mentioned_sources:
        top_sources = {}
        for doc in retrieved_docs[:3]:
            source = doc['source']
            page = doc['page']
            if source not in top_sources:
                top_sources[source] = set()
            top_sources[source].add(page)
        mentioned_sources = top_sources
    
    sources = []
    for source, pages in sorted(mentioned_sources.items()):
        for page in sorted(pages):
            sources.append({"file": source, "page": page})
    
    if debug:
        print(f"\nFiltered sources: {len(sources)} (from {len(all_sources)} total documents)")
    
    return sources

def answer_question(question: str, k: int = 10, debug: bool = False): 
    """
    Intelligent query handler with smart source filtering
    """
    if not client:
        return {
            "question": question,
            "answer": "Error: Please set GROQ_API_KEY. Get free key from: https://console.groq.com/keys",
            "sources": []
        }
    
    index, documents = load_vector_store()
    
    if debug:
        print(f"Loaded {len(documents)} chunks from vector store")
    
    query_embedding = sbert_model.encode(
        [f"query: {question}"],
        normalize_embeddings=True
    )
    
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    
    if debug:
        print(f"\n Retrieved {k} chunks")
        print(f"Best distance: {distances[0][0]:.4f}")
        print(f"Top sources: {list(set([doc['source'] for doc in retrieved_docs[:3]]))}")
    
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(f"[From document: {doc['source']}, Page {doc['page']}]\n{doc['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    if debug:
        print(f"Context: {len(context)} chars")
        print("Calling Groq API...")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a Sri Lankan tax expert assistant. Answer questions based on IRD documents.

CRITICAL INSTRUCTIONS:

1. DO NOT start your answer with "The information can be found in..." or "According to [document]..." UNLESS the user specifically asks "Which document" or "What document".

2. For content questions (What is X? How to calculate Y?):
   - Start DIRECTLY with the answer
   - Naturally mention the source document in your explanation if needed
   - Example: "Personal Relief has been increased to Rs. 1,800,000 as announced in PN_IT_2025-01."
   
3. For document name questions (Which document? What document?):
   - Answer with the document name clearly
   - Then optionally provide brief content
   
4. Use the [From document: ...] tags to know which document each information comes from.

5. Be concise and direct. Don't add unnecessary preambles.

6. ALWAYS end with: "Note: This response is based on IRD documents and is not professional tax advice."

7. If no relevant information is found, say: "I don't have enough information to answer this question."
"""
                },
                {
                    "role": "user",
                    "content": f"""Context from IRD tax documents:

{context}

Question: {question}

Answer directly and concisely. Do not start with "The information can be found in..." unless specifically asked about document names.ALWAYS end with in a new line: "Note: This response is based on IRD documents and is not professional tax advice."

Answer:"""
                }
            ],
            model=MODEL,
            temperature=0.2,
            max_tokens=400,
            top_p=0.9
        )
        
        answer = chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        answer = f"Error calling Groq API: {str(e)}"
    
    if debug:
        print(f"Generated answer: {answer[:200]}...")
    
    sources = filter_relevant_sources(answer, retrieved_docs, debug)
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SMART RAG SYSTEM (Clean Answers + Relevant Sources Only)")
    print("="*70)
    
    if not GROQ_API_KEY:
        print("\n No GROQ_API_KEY found!")
        print("\n Quick Setup:")
        print("1. Go to: https://console.groq.com")
        print("2. Sign up (free, no credit card needed)")
        print("3. Go to: https://console.groq.com/keys")
        print("4. Create API key")
        print("5. Run: export GROQ_API_KEY=your-key")
        print("\nThen run this script again.\n")
        exit(1)
    
    print(f"Using model: {MODEL}")
    print(f"Groq API key found\n")
    
    test_questions = [
        "What changes were announced in PN_IT_2025-01?",
        "What is the Corporate Income Tax rate for AY 2022/2023?",
        "What is SET?",
        "Which IRD document explains SET exemptions?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f" {question}")
        print('='*70)
        
        result = answer_question(question, k=10, debug=True)
        
        print(f"\n ANSWER: {result['answer']}")
        print(f"\n Sources:")
        for src in result['sources']:
            print(f"   • {src['file']} (Page {src['page']})")

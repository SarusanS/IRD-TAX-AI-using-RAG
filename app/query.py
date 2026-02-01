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

MODEL = "llama-3.1-8b-instant" # Groq LLM model

print("Loading models...")
sbert_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading vector store...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    documents = pickle.load(f)
print(f"Loaded {len(documents)} chunks")

def answer_question(question: str, k: int = 5, debug: bool = False): 
    if not client:
        return {
            "question": question,
            "answer": "Error: Please set GROQ_API_KEY. Get free key from: https://console.groq.com/keys",
            "sources": []
        }
    
    #RETRIEVE relevant chunks
    query_embedding = sbert_model.encode(
        [f"query: {question}"],
        normalize_embeddings=True
    )
    
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    
    if debug:
        print(f"\n Retrieved {k} chunks")
        print(f"Best distance: {distances[0][0]:.4f}")
    
    #Build context
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    
    if debug:
        print(f"Context: {len(context)} chars")
    
    #GENERATE answer using Groq LLM
    try:
        if debug:
            print("Calling Groq API...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a Sri Lankan tax expert assistant. Answer questions accurately based on the provided context from IRD documents."
                },
                {
                    "role": "user",
                    "content": f"""Context from IRD tax documents:

{context}

Question: {question}

Provide a clear, accurate answer based on the context above. If the answer is not in the context, say "I don't have enough information to answer this question."

Answer:"""
                }
            ],
            model=MODEL,
            temperature=0.3,
            max_tokens=300,
            top_p=0.9
        )
        
        answer = chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        answer = f"Error calling Groq API: {str(e)}"
    
    if debug:
        print(f"Generated answer: {answer[:200]}...")
    
    #Sources
    unique_sources = {(doc["source"], doc["page"]) for doc in retrieved_docs}
    sources = [{"file": f, "page": p} for f, p in sorted(unique_sources)]
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PROFESSIONAL RAG SYSTEM (Groq API - FREE)")
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
        "What is the Corporate Income Tax rate for AY 2022/2023?",
        "What is SET?",
        "Which IRD document explains SET exemptions?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f" {question}")
        print('='*70)
        
        result = answer_question(question, k=5, debug=True)
        
        print(f"\n ANSWER: {result['answer']}")
        print(f"\n Sources:")
        for src in result['sources'][:2]:
            print(f"   • {src['file']} (Page {src['page']})")

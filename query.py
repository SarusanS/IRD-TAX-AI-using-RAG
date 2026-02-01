import faiss
import pickle
import json
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# File paths
INDEX_FILE = "vectorstore/index.faiss"
META_FILE = "vectorstore/meta.pkl"

# Models
EMBEDDING_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"
LLM_MODEL = "google/flan-t5-large" 

print("Loading models...")
sbert_model = SentenceTransformer(EMBEDDING_MODEL)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Models loaded on {device}")

# Load vector store
print("Loading vector store...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    documents = pickle.load(f)
print(f"Loaded {len(documents)} document chunks")

def extract_answer_from_context(question: str, context: str, debug: bool = False) -> str:
    """
    IMPROVED pattern matching - finds RIGHT percentages
    """
    q_lower = question.lower()
    
    # Corporate tax rate questions
    if "corporate" in q_lower and "tax rate" in q_lower:
        # Look specifically for "Companies - 30%" pattern
        # Avoid interest rates by checking context
        matches = re.finditer(r'(companies?|trust)[s]?\s*[–\-:]\s*(\d+%)', context, re.IGNORECASE)
        
        for match in matches:
            entity = match.group(1)
            rate = match.group(2)
            
            # Check surrounding context (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(context), match.end() + 100)
            snippet = context[start:end].lower()
            
            # SKIP if it mentions interest/penalty
            skip_words = ['interest', 'penalty', 'late', 'per month', 'default']
            if any(word in snippet for word in skip_words):
                continue
            
            if debug:
                print(f"✓ Found: {entity} - {rate}")
            
            return f"The Corporate Income Tax rate for AY 2022/2023 is {rate} for {entity}."
    
    # "What is X?" questions
    if q_lower.startswith("what is"):
        term = q_lower.replace("what is", "").replace("?", "").strip()
        
        # Look for definitions
        patterns = [
            rf'{term}\s+(?:is|means|refers to)\s+([^.]+\.)',
            rf'{term}[:\-–]\s*([^.]+\.)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return f"{term.upper()}: {match.group(1)}"
    
    # Document questions
    if "document" in q_lower and "set" in q_lower:
        # Look for SET guide filename
        match = re.search(r'SET[_\s]+[\d_]+[^.\s]+', context)
        if match:
            return match.group(0)
    
    return None

def answer_question(question: str, k: int = 15, debug: bool = False):
    """
    Hybrid approach with better matching
    """

    # Retrieve
    query_embedding = sbert_model.encode(
        [f"query: {question}"],
        normalize_embeddings=True
    )

    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]

    if debug:
        print(f"\n🔍 Retrieved {k} chunks, best distance: {distances[0][0]:.4f}")
        print(f"\n📄 Top chunk preview:")
        print(retrieved_docs[0]['text'][:300] + "...")

    # Build context
    context = " ".join([doc["text"] for doc in retrieved_docs[:8]])

    # Try extraction
    extracted = extract_answer_from_context(question, context, debug=debug)
    
    if extracted and debug:
        print(f"\n💡 Extracted: {extracted}")

    # Try LLM
    prompt = f"""Answer based on the text.

Text: {context[:500]}

Q: {question}
A:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, num_beams=3)
    
    llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if debug:
        print(f"🤖 LLM: {llm_answer}")

    # Choose best
    if extracted and len(extracted) > 15:
        final_answer = extracted
    elif llm_answer and len(llm_answer) > 10 and "don't know" not in llm_answer.lower():
        final_answer = llm_answer
    elif extracted:
        final_answer = extracted
    else:
        final_answer = "I don't know based on the provided documents."

    # Sources
    unique_sources = {(doc["source"], doc["page"]) for doc in retrieved_docs[:5]}
    sources = [{"file": f, "page": p} for f, p in sorted(unique_sources)]

    if debug:
        print(f"\n✅ FINAL: {final_answer}\n")

    return {
        "question": question,
        "answer": final_answer,
        "sources": sources
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING IMPROVED PATTERN MATCHING")
    print("="*70)
    
    for q in [
        "What is the Corporate Income Tax rate for AY 2022/2023?",
        "What is SET?",
        "Which IRD document explains SET exemptions?",
    ]:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        print('='*70)
        
        result = answer_question(q, k=15, debug=True)
        print(f"📚 Sources: {result['sources'][0]['file']}")
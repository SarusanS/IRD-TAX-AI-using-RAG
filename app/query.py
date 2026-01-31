import faiss
import pickle
import json
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

# Main RAG function (OPTIMIZED FOR FLAN-T5-LARGE)
def answer_question(question: str, k: int = 5, debug: bool = False):
    """
    Optimized RAG for FLAN-T5-Large with simple prompts
    """

    #Encode question
    query_embedding = sbert_model.encode(
        [f"query: {question}"],
        normalize_embeddings=True
    )

    #Retrieve top-k chunks
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]

    if debug:
        print(f"\n Retrieved {len(retrieved_docs)} chunks")
        print(f" Top 3 distances: {distances[0][:3]}")

    #Build context - KEEP IT SHORT for better results
    context_parts = []
    total_words = 0
    max_context_words = 400  # Reduced for FLAN-T5-Large
    
    for doc in retrieved_docs:
        words_in_chunk = len(doc["text"].split())
        if total_words + words_in_chunk > max_context_words:
            # Add partial chunk if there's room
            remaining = max_context_words - total_words
            if remaining > 50:  # Only add if meaningful
                words = doc["text"].split()[:remaining]
                context_parts.append(" ".join(words))
            break
        
        context_parts.append(doc["text"])
        total_words += words_in_chunk
    
    context = "\n\n".join(context_parts)

    if debug:
        print(f"Context: {total_words} words, {len(context_parts)} chunks")
        print(f"\n--- Context Preview ---")
        print(context[:300] + "...\n")

    #Collect sources
    unique_sources = {(doc["source"], doc["page"]) for doc in retrieved_docs}
    sources = [{"file": f, "page": p} for f, p in sorted(unique_sources)]

    # SIMPLE prompt that works well with FLAN-T5-Large
    # Key: Keep it SHORT and DIRECT
    prompt = f"""Answer the question using only the context below.

Context: {context}

Question: {question}

Answer:"""

    if debug:
        print(f"Prompt length: {len(prompt.split())} words")

    #Tokenize with proper truncation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    if debug:
        print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    # Generate with optimized parameters for FLAN-T5-Large
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,          
            num_beams=4,               
            do_sample=False,         
            early_stopping=True,
            repetition_penalty=1.3,    
            length_penalty=0.8,        
            no_repeat_ngram_size=3       

    answer_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if debug:
        print(f"\n Generated answer: {answer_text}\n")

    # Post-process answer
    # Remove common artifacts
    answer_text = answer_text.replace("documentsss", "documents")
    
    # If answer is too short, it might be a failure
    if len(answer_text) < 10 and "don't know" not in answer_text.lower():
        answer_text = "I don't know based on the provided documents."

    #Return result
    return {
        "question": question,
        "answer": answer_text,
        "sources": sources
    }

# Example run
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING RAG SYSTEM")
    print("="*70)
    
    test_questions = [
        "What is the Corporate Income Tax rate for AY 2022/2023?",
        "What is SET?",
        "Which IRD document explains SET exemptions?",
    ]
    
    for question in test_questions:
        print(f"\n {'='*70}")
        print(f"Question: {question}")
        print('='*70)
        
        result = answer_question(question, k=5, debug=True)
        
        print(f"\n ANSWER: {result['answer']}")
        print(f"\n Sources:")
        for src in result['sources']:
            print(f"   • {src['file']} (Page {src['page']})")
        print()
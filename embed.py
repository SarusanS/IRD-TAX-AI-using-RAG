import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.ingest import load_and_chunk_pdfs


# Paths
VECTOR_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
META_FILE = os.path.join(VECTOR_DIR, "meta.pkl")


EMBEDDING_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"  # Embedding model

def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    print("Ingesting & chunking PDFs...")
    documents = load_and_chunk_pdfs()

    print(f"Total chunks: {len(documents)}")

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Prepare E5-style passages
    texts = [f"passage: {doc['text']}" for doc in documents]

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Create FAISS index (L2 is OK after normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)


    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(documents, f)

    print("Vector store created successfully")

if __name__ == "__main__":
    main()

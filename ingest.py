import os
import fitz  # PyMuPDF

PDF_DIR = "data/pdfs"
MAX_CHUNK_SIZE = 300  # INCREASED from 200 for more context
CHUNK_OVERLAP = 50    # NEW: Overlap between chunks to preserve context

def load_and_chunk_pdfs(pdf_dir="data/pdfs"):
    """
    Loads PDFs from a directory, extracts text, and splits into 
    chunks with OVERLAP to preserve context.
    
    IMPROVEMENTS:
    - Larger chunks (300 words instead of 200)
    - Overlap between chunks (50 words)
    - Better sentence preservation
    """
    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, filename)
        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            # Clean up text
            text = text.replace('\n', ' ').replace('  ', ' ').strip()
            
            words = text.split()
            
            # Create overlapping chunks
            i = 0
            while i < len(words):
                # Take MAX_CHUNK_SIZE words
                chunk_words = words[i:i + MAX_CHUNK_SIZE]
                chunk_text = " ".join(chunk_words)
                
                documents.append({
                    "text": chunk_text,
                    "source": filename,
                    "page": page_num
                })
                
                # Move forward by (MAX_CHUNK_SIZE - CHUNK_OVERLAP)
                # This creates overlap between chunks
                i += (MAX_CHUNK_SIZE - CHUNK_OVERLAP)
                
                # Stop if we're at the end
                if i >= len(words):
                    break

    print(f"✅ Total chunks created: {len(documents)}")
    return documents

if __name__ == "__main__":
    # Test the chunking
    docs = load_and_chunk_pdfs()
    print(f"\n📄 Sample chunks:")
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {doc['source']}, Page: {doc['page']}")
        print(f"Text: {doc['text'][:200]}...")
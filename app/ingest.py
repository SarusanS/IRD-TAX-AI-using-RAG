import os
import fitz 

Pdf_dir = "data/pdfs"
MAX_CHUNK_SIZE = 200 

def load_and_chunk_pdfs(pdf_dir="data/pdfs"):
    """
    Loads PDFs from a directory, extracts text, and splits into 
    normal chunks of MAX_CHUNK_SIZE words.
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

            words = text.split()
            # Split words into chunks
            for i in range(0, len(words), MAX_CHUNK_SIZE):
                chunk_words = words[i:i + MAX_CHUNK_SIZE]
                chunk_text = " ".join(chunk_words)
                documents.append({
                    "text": chunk_text,
                    "source": filename,
                    "page": page_num
                })

    print(f" Total chunks created: {len(documents)}")
    return documents

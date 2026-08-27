# scripts/ingest.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chunker import load_and_chunk, chunk_words
from core.embedder import embed_texts
from core.vector_store import ChromaVectorStore

# OCR is optional — only needed for scanned PDFs without selectable text
try:
    from core.ocr_chunker import ocr_pdf_to_chunks
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

store = ChromaVectorStore()
store.reset()   # clean rebuild every ingest, like the old pickle overwrite

for filename in os.listdir("corpus/raw"):
    filepath = f"corpus/raw/{filename}"

    if filename.endswith(".pdf"):
        print(f"Processing {filename}...")
        chunks = load_and_chunk(filepath)

        if len(chunks) == 0:
            if _HAS_OCR:
                print(f"  -> No selectable text found, falling back to OCR...")
                chunks = ocr_pdf_to_chunks(filepath)
            else:
                print(f"  -> No selectable text found (OCR not available), skipping {filename}")
                continue

        if len(chunks) == 0:
            print(f"  -> OCR also found nothing, skipping {filename}")
            continue

    elif filename.endswith(".txt"):
        print(f"Processing {filename} (plain text)...")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        if not words:
            print(f"  -> Empty file, skipping {filename}")
            continue
        chunks = chunk_words(words, chunk_size=200, overlap=50)

    else:
        continue   # skip non-PDF/non-TXT files

    vectors = embed_texts(chunks)
    store.add(chunks, vectors, source=filename)
    print(f"  -> {len(chunks)} chunks added")

print(f"Done. {store.count()} chunks stored in Chroma at corpus/processed/chroma.")

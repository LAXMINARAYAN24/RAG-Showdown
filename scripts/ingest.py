# scripts/ingest.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chunker import load_and_chunk
from core.ocr_chunker import ocr_pdf_to_chunks
from core.embedder import embed_texts
from core.vector_store import SimpleVectorStore
import pickle

store = SimpleVectorStore()

for filename in os.listdir("corpus/raw"):
    if filename.endswith(".pdf"):
        print(f"Processing {filename}...")
        chunks = load_and_chunk(f"corpus/raw/{filename}")

        if len(chunks) == 0:
            print(f"  -> No selectable text found, falling back to OCR...")
            chunks = ocr_pdf_to_chunks(f"corpus/raw/{filename}")

        if len(chunks) == 0:
            print(f"  -> OCR also found nothing, skipping {filename}")
            continue

        vectors = embed_texts(chunks)
        store.add(chunks, vectors)
        print(f"  -> {len(chunks)} chunks added")

with open("corpus/processed/store.pkl", "wb") as f:
    pickle.dump(store, f)

print(f"Done. {len(store.chunks)} chunks stored.")
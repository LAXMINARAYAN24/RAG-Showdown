# core/embedder.py
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"   # give it more time per request
# os.environ["HF_HUB_OFFLINE"] = "1"   

from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_texts(texts):
    return _model.encode(texts, convert_to_numpy=True)
from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, chunks_with_scores, top_n=5):
    """
    Takes the wide net of retrieved chunks (e.g. top 15 by embedding similarity)
    and re-scores each one specifically against the question, using a model
    that's better at judging true relevance than raw embedding similarity.
    Returns the best `top_n`, re-sorted.
    """
    pairs = [(question, chunk_text) for chunk_text, _ in chunks_with_scores]
    rerank_scores = _reranker.predict(pairs)

    reranked = list(zip([c for c, _ in chunks_with_scores], rerank_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:top_n]
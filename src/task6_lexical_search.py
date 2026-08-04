"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Corpus: nạp trực tiếp từ ChromaDB (cùng các chunk đã index ở Task 4), nhờ vậy
BM25 và semantic search luôn tìm trên đúng cùng một tập chunk — không cần ai
phải tự gọi build_bm25_index(corpus) thủ công trước khi dùng lexical_search().
"""

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict} — nạp lười (lazy) khi cần

_bm25_cache = None
_corpus_cache: list[dict] = []


def _load_corpus() -> list[dict]:
    """Nạp toàn bộ chunk đã index ở Task 4 từ ChromaDB làm corpus cho BM25."""
    from src.task4_chunking_indexing import get_collection

    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])

    corpus = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        corpus.append({"content": doc, "metadata": meta})
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_bm25():
    """Cache BM25 index + corpus để không rebuild mỗi lần gọi lexical_search."""
    global _bm25_cache, _corpus_cache
    if _bm25_cache is None:
        _corpus_cache = _load_corpus()
        if not _corpus_cache:
            raise RuntimeError(
                "Corpus rỗng — hãy chạy `python -m src.task4_chunking_indexing` trước "
                "để index dữ liệu vào ChromaDB."
            )
        _bm25_cache = build_bm25_index(_corpus_cache)
    return _bm25_cache, _corpus_cache


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    bm25, corpus = _get_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("bảo hiểm xã hội thất nghiệp", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

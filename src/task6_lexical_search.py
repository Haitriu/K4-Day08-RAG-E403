"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 6 — Lexical Search Module (BM25).
Mô-đun tìm kiếm dựa trên từ khóa truyền thống (Keyword Search).
Sử dụng thư viện rank_bm25 để đánh giá mức độ xuất hiện của từ khóa.
"""

from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path
import json

# Cố gắng load corpus từ file chunk đã lưu (Nếu Nam - Role 2 đã làm xong Task 4)
CORPUS = []
BM25_INDEX = None

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global CORPUS, BM25_INDEX
    CORPUS = corpus
    
    if not CORPUS:
        print("⚠️ CẢNH BÁO: Corpus rỗng, BM25 Index không được tạo.")
        return None

    # Tokenize: Tách từ cơ bản bằng khoảng trắng và đưa về in thường
    tokenized_corpus = [doc.get("content", "").lower().split() for doc in CORPUS]
    BM25_INDEX = BM25Okapi(tokenized_corpus)
    return BM25_INDEX


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
    """
    if not BM25_INDEX or not CORPUS:
        print("⚠️ CẢNH BÁO: BM25_INDEX chưa được khởi tạo. Hãy gọi build_bm25_index() trước.")
        return []

    tokenized_query = query.lower().split()
    scores = BM25_INDEX.get_scores(tokenized_query)

    # Lấy top_k kết quả có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Chỉ lấy những doc có chứa từ khóa (score > 0)
            results.append({
                "content": CORPUS[idx].get("content", ""),
                "score": float(scores[idx]),
                "metadata": CORPUS[idx].get("metadata", {})
            })
    return results

if __name__ == "__main__":
    # Test giả lập
    mock_corpus = [
        {"content": "Người lao động được nghỉ thai sản 6 tháng.", "metadata": {"source": "luat_ld.md"}},
        {"content": "Công ty trả lương vào ngày mùng 5 hàng tháng.", "metadata": {"source": "noi_quy.md"}},
        {"content": "Thời gian thử việc tối đa là 60 ngày đối với bằng đại học.", "metadata": {"source": "luat_ld.md"}},
    ]
    build_bm25_index(mock_corpus)
    print("Test BM25 Search:")
    res = lexical_search("thử việc 60 ngày", top_k=2)
    for r in res:
        print(f"[{r['score']:.3f}] {r['content']}")

"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — dùng điểm cosine gốc (`original_semantic_score`) được
giữ lại trong mỗi item sau khi fuse.
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker API).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY chưa được set trong .env — cần để gọi Jina Reranker API")
    if not candidates:
        return []

    import requests

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": r["relevance_score"]}
        for r in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
        return float(np.dot(a, b) / denom)

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion (dạng tổng quát cho N ranked lists bất kỳ).

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["rrf_score"] = score  # điểm fused, KHÔNG dùng để so threshold (xem docstring module)
        item["score"] = score
        results.append(item)

    return results


def rrf_rerank(
    semantic_results: list[dict], lexical_results: list[dict], k: int = 60, top_k: int = 5
) -> list[dict]:
    """
    Reciprocal Rank Fusion chuyên biệt cho đúng 2 nguồn Semantic + Lexical (dùng ở Task 9).

    RRF_Score = 1 / (k + rank)

    Khác với `rerank_rrf` (tổng quát cho N list), hàm này còn giữ lại điểm gốc
    (`original_semantic_score`, `original_lexical_score`) trong từng kết quả —
    cần thiết để Task 9 kiểm tra ngưỡng fallback bằng điểm cosine THẬT thay vì
    điểm RRF đã fuse (điểm RRF luôn quá nhỏ và không phản ánh độ liên quan).

    Args:
        semantic_results: Kết quả từ semantic_search (Task 5), có 'score' là cosine similarity.
        lexical_results: Kết quả từ lexical_search (Task 6), có 'score' là BM25 score.
        k: Hằng số RRF (mặc định 60).
        top_k: Số kết quả cuối cùng.

    Returns:
        Danh sách kết quả đã gộp và sắp xếp lại, mỗi item có thêm 'rrf_score',
        'original_semantic_score' và/hoặc 'original_lexical_score'.
    """
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    def process_results(results_list, source_name):
        for rank, doc in enumerate(results_list):
            content = doc["content"]
            if content not in rrf_scores:
                rrf_scores[content] = 0.0
                doc_lookup[content] = {
                    "content": content,
                    "metadata": doc.get("metadata", {}).copy(),
                    "source": "hybrid",  # đánh dấu nguồn — phân biệt với "pageindex" (Task 8 fallback)
                    f"original_{source_name}_score": doc.get("score", 0.0),
                }
            else:
                doc_lookup[content][f"original_{source_name}_score"] = doc.get("score", 0.0)

            rrf_scores[content] += 1.0 / (k + rank + 1)

    process_results(semantic_results, "semantic")
    process_results(lexical_results, "lexical")

    reranked_results = []
    for content, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_lookup[content]
        doc["rrf_score"] = score
        doc["score"] = score
        reranked_results.append(doc)

    return reranked_results[:top_k]


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        from src.task4_chunking_indexing import embed_texts

        query_embedding = embed_texts([query])[0]
        # Cần embedding sẵn trong từng candidate — embed nếu còn thiếu
        missing = [c for c in candidates if "embedding" not in c]
        if missing:
            embs = embed_texts([c["content"] for c in missing])
            for c, e in zip(missing, embs):
                c["embedding"] = e
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # method="rrf" qua interface thống nhất: xem `candidates` như 1 ranked list duy nhất.
        # Khi có 2 ranker (semantic + lexical) như ở Task 9, gọi rrf_rerank(sem, lex, ...)
        # trực tiếp thay vì qua wrapper này.
        return rerank_rrf([candidates], top_k, k=60)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")

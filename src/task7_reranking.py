"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 7 — Reranking Module.
Mô-đun chấm điểm lại và trộn kết quả (Fusion & Reranking).
Sử dụng thuật toán Reciprocal Rank Fusion (RRF) để kết hợp kết quả 
từ Semantic Search (Vector) và Lexical Search (BM25).
"""

def rrf_rerank(semantic_results: list[dict], lexical_results: list[dict], k: int = 60, top_k: int = 5) -> list[dict]:
    """
    Kết hợp kết quả bằng Reciprocal Rank Fusion (RRF).
    RRF_Score = 1 / (k + rank)
    
    LƯU Ý QUAN TRỌNG:
    Điểm RRF chỉ dùng để SẮP XẾP LẠI (Re-order), tuyệt đối không dùng điểm này
    để làm score_threshold cho fallback (bởi vì điểm RRF luôn rất nhỏ và không
    phản ánh độ liên quan của nội dung). Điểm gốc (cosine similarity) sẽ được 
    giữ lại trong metadata để fallback kiểm tra.

    Args:
        semantic_results: Danh sách kết quả từ Semantic Search (có 'score' là cosine similarity).
        lexical_results: Danh sách kết quả từ Lexical Search (BM25).
        k: Hằng số RRF (thường là 60).
        top_k: Số kết quả cuối cùng cần trả về.
        
    Returns:
        Danh sách kết quả đã được sắp xếp lại.
    """
    rrf_scores = {}
    doc_lookup = {}
    
    # Hàm helper để xử lý từng danh sách kết quả
    def process_results(results_list, source_name):
        for rank, doc in enumerate(results_list):
            content = doc["content"]
            # Dùng nội dung làm khóa (hoặc id nếu có)
            if content not in rrf_scores:
                rrf_scores[content] = 0.0
                doc_lookup[content] = {
                    "content": content,
                    "metadata": doc.get("metadata", {}).copy(),
                    # Lưu lại điểm gốc để dùng cho fallback
                    f"original_{source_name}_score": doc.get("score", 0.0)
                }
            else:
                # Cập nhật thêm điểm gốc nếu doc đã tồn tại
                doc_lookup[content][f"original_{source_name}_score"] = doc.get("score", 0.0)
                
            # Cộng điểm RRF
            rrf_scores[content] += 1.0 / (k + rank + 1)
            
    # Xử lý cả hai nguồn kết quả
    process_results(semantic_results, "semantic")
    process_results(lexical_results, "lexical")
    
    # Sắp xếp lại dựa trên điểm RRF tổng
    reranked_results = []
    for content, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_lookup[content]
        doc["rrf_score"] = score
        reranked_results.append(doc)
        
    return reranked_results[:top_k]

if __name__ == "__main__":
    # Test RRF
    sem_res = [{"content": "Doc A", "score": 0.8}, {"content": "Doc B", "score": 0.7}]
    lex_res = [{"content": "Doc B", "score": 2.5}, {"content": "Doc C", "score": 1.2}]
    
    final = rrf_rerank(sem_res, lex_res)
    print("Kết quả sau khi RRF:")
    for r in final:
        print(f"- {r['content']} (RRF: {r['rrf_score']:.4f})")

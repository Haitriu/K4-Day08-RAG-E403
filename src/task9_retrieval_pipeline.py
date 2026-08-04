"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 9 — Retrieval Pipeline Hoàn Chỉnh.
Ghép nối tất cả các luồng tìm kiếm lại thành một kiến trúc duy nhất:
Query -> Semantic & Lexical -> RRF Rerank -> (Nếu Cosine < 0.48) -> PageIndex Fallback.
"""

def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.48, config: dict = None) -> list[dict]:
    """
    Hàm điều phối luồng tìm kiếm (Supervisor).
    
    Quy trình:
    1. Chạy semantic_search (Task 5) và lexical_search (Task 6).
    2. Trộn kết quả bằng RRF (Task 7).
    3. Kiểm tra xem kết quả tốt nhất có đủ độ tin cậy không (Cosine >= 0.48).
    4. Nếu không đủ tin cậy, kích hoạt PageIndex Vectorless (Task 8).
    
    Args:
        query: Câu hỏi của người dùng.
        top_k: Số lượng kết quả cần trả về.
        score_threshold: Ngưỡng Cosine Similarity để kích hoạt Fallback (Mặc định 0.48 chuẩn BTC).
        config: Dùng để test A/B (Ví dụ: tắt mở Reranking).

    Returns:
        Danh sách kết quả tốt nhất.
    """
    if config is None:
        config = {"use_semantic": True, "use_lexical": True, "use_reranking": True}

    # Import giả lập các module (Vì chưa được nối với code thực tế của các thành viên khác)
    try:
        from src.task5_semantic_search import semantic_search
    except ImportError:
        semantic_search = lambda q, k: []
        
    try:
        from src.task6_lexical_search import lexical_search
    except ImportError:
        lexical_search = lambda q, k: []
        
    try:
        from src.task7_reranking import rrf_rerank
    except ImportError:
        rrf_rerank = lambda sem, lex: sem + lex
        
    try:
        from src.task8_pageindex_vectorless import pageindex_search
    except ImportError:
        pageindex_search = lambda q, k: []

    # BƯỚC 1: Lấy kết quả từ 2 nguồn
    sem_results = semantic_search(query, top_k=top_k*2) if config.get("use_semantic", True) else []
    lex_results = lexical_search(query, top_k=top_k*2) if config.get("use_lexical", True) else []
    
    # BƯỚC 2: Reranking
    if config.get("use_reranking", True):
        # Trộn bằng RRF
        final_results = rrf_rerank(sem_results, lex_results, top_k=top_k)
    else:
        # Nếu không Reranking (Ví dụ test Config B: Lexical Only)
        final_results = lex_results[:top_k]

    # BƯỚC 3: Fallback Logic (Kiểm tra điểm Cosine gốc)
    # LƯU Ý BTC: So điểm Cosine gốc, TUYỆT ĐỐI KHÔNG so điểm RRF
    best_cosine_score = 0.0
    
    if final_results:
        # Lấy điểm semantic gốc được lưu trong metadata bởi hàm rrf_rerank
        best_cosine_score = final_results[0].get("original_semantic_score", 0.0)
    
    # BƯỚC 4: Kích hoạt PageIndex nếu dưới ngưỡng 0.48 hoặc không có kết quả
    if not final_results or best_cosine_score < score_threshold:
        print(f"⚠️ Điểm Cosine tốt nhất ({best_cosine_score:.2f}) < {score_threshold}. Kích hoạt PageIndex Fallback!")
        fallback_results = pageindex_search(query, top_k=top_k)
        if fallback_results:
            return fallback_results
            
    return final_results

if __name__ == "__main__":
    print("Testing Retrieval Pipeline Supervisor...")
    res = retrieve("Làm sao để trả hàng Shopee?")
    print(f"Tìm được {len(res)} kết quả.")

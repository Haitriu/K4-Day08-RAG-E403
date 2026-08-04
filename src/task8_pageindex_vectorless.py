"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 8 — PageIndex Vectorless RAG.
Mô-đun dự phòng (Fallback) khi Hybrid Search (Vector + BM25) không tìm ra kết quả tốt.

Đăng ký tài khoản tại: https://pageindex.ai/ (API key lấy ở Developer Dashboard riêng:
https://dash.pageindex.ai/api-keys, KHÔNG nằm trong Settings của trang chính).
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng structural understanding
của document thay vì embedding.

Cài đặt:
    pip install pageindex fpdf2

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]].
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_DIR = Path(__file__).parent.parent / "pageindex_pdfs"
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def upload_documents(timeout_seconds: int = 600, poll_interval: int = 5):
    """
    Gộp toàn bộ markdown documents (mỗi file = 1 "chương") thành 1 PDF duy nhất
    và upload lên PageIndex — PageIndex đọc hiểu theo cấu trúc chương/mục nên
    gộp thành 1 tài liệu nhiều chương tận dụng đúng thế mạnh của nó, thay vì
    tách rời nhiều PDF nhỏ (sẽ phải query từng cái rồi tự gộp kết quả).
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Chưa cấu hình PAGEINDEX_API_KEY.")

    from pageindex import PageIndexClient
    from fpdf import FPDF

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    combined_pdf = PDF_DIR / "labor_law_corpus.pdf"

    pdf = FPDF()
    pdf.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf")

    md_files = [
        path
        for path in sorted(STANDARDIZED_DIR.rglob("*.md"))
        if path.stat().st_size > 0
    ]
    if not md_files:
        raise RuntimeError("Không có Markdown hợp lệ để upload lên PageIndex.")
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.write(10, f"# {md_file.stem}\n\n")
        pdf.set_font("Arial", size=11)
        pdf.write(8, text)
    pdf.output(str(combined_pdf))
    print(f"[OK] Combined {len(md_files)} markdown files -> {combined_pdf}")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    resp = client.submit_document(str(combined_pdf))
    doc_id = resp.get("doc_id") or resp.get("id")
    if not doc_id:
        raise RuntimeError("PageIndex không trả về doc_id sau khi upload.")
    print(f"[OK] Uploaded -> doc_id={doc_id}")

    print("[INFO] Waiting for PageIndex document processing...")
    deadline = time.monotonic() + timeout_seconds
    while not client.is_retrieval_ready(doc_id):
        if time.monotonic() >= deadline:
            raise TimeoutError("PageIndex xử lý tài liệu quá thời gian cho phép.")
        time.sleep(poll_interval)
    print("[OK] Document is ready for retrieval")

    DOC_IDS_FILE.write_text(json.dumps({"doc_id": doc_id}, indent=2), encoding="utf-8")
    return doc_id


def pageindex_search(
    query: str,
    top_k: int = 5,
    timeout_seconds: int = 120,
    poll_interval: int = 2,
) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt (gọi trực tiếp từ
    Task 9 mà không bọc try/except, nên hàm này KHÔNG raise — mọi lỗi/thiếu cấu
    hình chỉ in cảnh báo và trả về [] để pipeline luôn trả kết quả rỗng an toàn).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        print("[WARN] PAGEINDEX_API_KEY is not configured; skipping fallback.")
        return []
    if not DOC_IDS_FILE.exists():
        print("[WARN] PageIndex doc_id is missing; run the upload command first.")
        return []

    try:
        from pageindex import PageIndexClient

        doc_id = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))["doc_id"]
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")

        if not retrieval_id:
            raise RuntimeError("PageIndex không trả về retrieval_id.")

        retrieval = client.get_retrieval(retrieval_id)
        deadline = time.monotonic() + timeout_seconds
        while retrieval.get("status") not in ("completed", "failed"):
            if time.monotonic() >= deadline:
                raise TimeoutError("PageIndex query quá thời gian cho phép.")
            time.sleep(poll_interval)
            retrieval = client.get_retrieval(retrieval_id)

        if retrieval.get("status") == "failed":
            return []

        results = []
        rank = 0
        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    rank += 1
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": round(1.0 / rank, 4),  # PageIndex không trả score — gán theo rank
                        "metadata": {"section": item.get("section_title")},
                        "source": "pageindex",
                    })
        return results[:top_k]
    except Exception as exc:
        print(f"[WARN] PageIndex query failed ({type(exc).__name__}).")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[WARN] Set PAGEINDEX_API_KEY in .env before uploading.")
        print("Get a key at https://dash.pageindex.ai/api-keys")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("Nghỉ thai sản", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")

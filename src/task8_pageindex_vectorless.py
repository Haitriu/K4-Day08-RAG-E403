"""
Vũ Bảo Khánh hoàn thiện (Role 1 - RAG Architect).

Task 8 — PageIndex Vectorless RAG.
Mô-đun dự phòng (Fallback) khi Hybrid Search (Vector + BM25) không tìm ra kết quả tốt.
Sử dụng API của PageIndex (vectify.ai) để tìm kiếm mà không cần dùng đến Vector DB nội bộ.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Lấy API Key từ biến môi trường
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex API.
    Đây là phương án Fallback cực kỳ mạnh mẽ khi hệ thống nội bộ bị fail.

    Args:
        query: Câu truy vấn của người dùng.
        top_k: Số kết quả trả về.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    if not PAGEINDEX_API_KEY:
        print("⚠️ CẢNH BÁO: Chưa cấu hình PAGEINDEX_API_KEY trong file .env. Bỏ qua Fallback.")
        return []

    print("🔄 Kích hoạt Fallback: Đang truy vấn qua PageIndex API...")
    
    url = "https://api.pageindex.ai/v1/search"
    headers = {
        "Authorization": f"Bearer {PAGEINDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Payload cần tham khảo tài liệu chính thức của PageIndex
    # Giả lập payload cơ bản (Tùy thuộc vào việc Khánh đã tạo Index ID nào trên hệ thống)
    # Nếu chưa upload data lên PageIndex thì API sẽ trả về rỗng.
    payload = {
        "query": query,
        "top_k": top_k
        # "index_id": "YOUR_INDEX_ID_HERE" # Khánh cần thêm index_id vào đây sau khi tạo
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("results", []):
                results.append({
                    "content": item.get("text", ""),
                    "metadata": {"source": "PageIndex Fallback"}
                })
            return results
        else:
            print(f"❌ Lỗi PageIndex API: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Lỗi kết nối đến PageIndex: {e}")
        return []

if __name__ == "__main__":
    # Test
    res = pageindex_search("Nghỉ thai sản")
    print("Kết quả PageIndex:")
    print(res)

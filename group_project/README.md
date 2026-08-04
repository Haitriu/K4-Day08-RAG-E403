# Trợ lý Pháp luật Lao động Việt Nam

## Mục tiêu

Chatbot RAG trả lời câu hỏi phổ biến về pháp luật lao động Việt Nam, chỉ dựa trên
nguồn đã truy xuất và hiển thị trích dẫn đi kèm. Nội dung chỉ có tính chất tham
khảo, không thay thế tư vấn pháp lý chính thức.

## Kiến trúc

```text
PDF + bài viết đã crawl
        │
        ▼
Markdown chuẩn hóa → chunking → multilingual embeddings → ChromaDB
                                      │
Câu hỏi → Semantic Search ────────────┤
        → BM25 Lexical Search ────────┼→ RRF reranking → Task 10 + LLM
                                      │                         │
                                      └→ PageIndex fallback ────┘
                                                                │
                                                                ▼
                                              Streamlit + citation + sources
```

`supervisor.py` là điểm vào dùng chung cho CLI; `app.py` là giao diện demo.
PageIndex chỉ được dùng khi hybrid retrieval không có bằng chứng đủ mạnh và đã
cấu hình tài liệu trên dịch vụ PageIndex.

## Phân công

| Thành viên | MSSV | Vai trò | Phạm vi | Trạng thái |
|---|---|---|---|---|
| Vũ Bảo Khánh | 2A202601122 | Team Leader & RAG Architect | Điều phối, Task 6–9, ghép pipeline và supervisor | Hoàn thành code |
| Võ Hồ Nhật Nam | 2A202601700 | Data & Retrieval Specialist | Task 1–5: thu thập, chuẩn hóa, chunking, ChromaDB, semantic search | Hoàn thành |
| Nguyễn Xuân Hải | 2A202602022 | Frontend & Chatbot Developer | Task 10, prompt/citation, lost-in-the-middle, Streamlit UI | Hoàn thành |
| Phạm Đức Hải Triều | 2A202601980 | Evaluation & QA Engineer | Golden dataset, RAGAS, A/B test và báo cáo | Hoàn thành code; chờ live eval |

## Chạy dự án

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Tạo vector store ở lần chạy đầu:

```powershell
python -m src.task4_chunking_indexing
```

Chạy CLI retrieval hoặc chatbot:

```powershell
python supervisor.py "Thời gian thử việc tối đa là bao lâu?" --retrieve-only
streamlit run app.py
```

Các biến môi trường chính:

- `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY`: sinh câu trả lời.
- `OPENROUTER_MODEL`, `OPENAI_MODEL`: tùy chọn model.
- `EMBEDDING_MODEL`: mặc định `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- `PAGEINDEX_API_KEY`: chỉ cần khi bật fallback PageIndex sau khi upload corpus.

## Evaluation

`evaluation/golden_dataset.json` có 15 câu hỏi Luật Lao động. Chạy RAGAS trong
môi trường evaluation riêng theo hướng dẫn trong `requirements-eval.txt`, sau đó:

```powershell
python group_project/evaluation/eval_pipeline.py
```

Script so sánh hai cấu hình: Hybrid + RRF và BM25-only, chấm bốn metric RAGAS,
ghi bảng điểm và ba trường hợp kém nhất vào `evaluation/results.md`.

## Kiểm thử

```powershell
python -m pytest -q
python -m py_compile app.py supervisor.py src\task*.py group_project\evaluation\eval_pipeline.py
```

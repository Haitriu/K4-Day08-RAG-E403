# Báo cáo đánh giá RAG và A/B testing

> Chưa chạy chấm điểm RAGAS đầy đủ. Script và golden dataset đã sẵn sàng; việc
> chạy 30 lượt sinh câu trả lời cùng các lượt LLM-as-a-judge cần môi trường
> `requirements-eval.txt` và quyền sử dụng API có tính phí.

## Trạng thái kiểm chứng cục bộ

| Hạng mục | Kết quả |
|---|---|
| Golden dataset Luật Lao động | 15/15 mẫu hợp lệ |
| Unit/integration tests | 45 passed, 2 skipped |
| Streamlit smoke test | Khởi động không exception |
| End-to-end Hybrid → LLM | Thành công, có câu trả lời và sources |
| PageIndex live fallback | Chưa upload corpus ra dịch vụ ngoài |

## Cách tạo báo cáo số liệu

```powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv-eval\Scripts\python.exe group_project\evaluation\eval_pipeline.py
```

Khi chạy thành công, `eval_pipeline.py` sẽ thay nội dung file này bằng điểm bốn
metric, so sánh Hybrid + Reranking với Lexical-only, ba worst performers và đề
xuất cải tiến. Không điền số liệu giả khi chưa thực hiện LLM-as-a-judge.

# 📊 Báo Cáo Đánh Giá Chất Lượng RAG (Sử dụng RAGAS)

> ⏳ *Hệ thống đang chờ được chạy file `eval_pipeline.py` để cập nhật kết quả đánh giá mới nhất dựa trên 15 câu hỏi vàng (Golden Dataset) về Chính sách Thương mại điện tử (Shopee).*

## 1. Điểm Số Tổng Quan (Overall Scores)

| Tiêu chí (Metric) | Ý nghĩa | Điểm trung bình (0-1) |
|-------------------|---------|-----------------------|
| **Faithfulness** | Bot có bịa luật không? (Độ trung thực) | `Đang chờ...` |
| **Answer Relevancy** | Trả lời có đúng trọng tâm câu hỏi? | `Đang chờ...` |
| **Context Recall** | Bot có moi ra được đủ các điều luật cần thiết? | `Đang chờ...` |
| **Context Precision**| Các điều luật moi ra có bị rác không? | `Đang chờ...` |

## 2. Chi Tiết Các Câu Trả Lời Tệ Nhất (Worst Performers)

> Đây là các câu hỏi mà Chatbot trả lời sai hoặc lạc đề nhất, cần kiểm tra lại Prompt (Task 10) hoặc dữ liệu mồi (Task 4, 5).

*(Kết quả sẽ được hiển thị ở đây sau khi script RAGAS chấm điểm xong)*

---
*Báo cáo này được tự động tạo bởi `eval_pipeline.py` (Phụ trách: Triều - QA Engineer).*

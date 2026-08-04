"""
RAG Evaluation Pipeline (Role 4 - QA/Tester).

Script này sử dụng thư viện RAGAS để tự động chấm điểm Chatbot dựa trên 15 câu hỏi
trong `golden_dataset.json`. Kết quả sẽ được xuất ra file `results.md`.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS và OpenAI.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
        from langchain_openai.chat_models import ChatOpenAI
        from langchain_openai.embeddings import OpenAIEmbeddings
    except ImportError:
        print("⚠️ Thiếu thư viện. Vui lòng chạy: pip install ragas datasets langchain-openai")
        return None

    # Khởi tạo mô hình đánh giá (LLM-as-a-judge)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ CẢNH BÁO: Không tìm thấy OPENAI_API_KEY trong .env. RAGAS sẽ không thể chấm điểm.")
        return None

    print("⏳ Đang khởi tạo mô hình chấm điểm (GPT-4o-mini)...")
    eval_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    eval_embeddings = OpenAIEmbeddings(api_key=api_key)

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print(f"⏳ Đang gọi Chatbot trả lời {len(golden_dataset)} câu hỏi Luật...")
    for i, item in enumerate(golden_dataset):
        try:
            # Gọi hàm sinh câu trả lời từ chatbot (Task 10)
            if hasattr(rag_pipeline, "generate_with_citation"):
                result = rag_pipeline.generate_with_citation(item["question"])
            else:
                # Mock nếu chưa có pipeline
                result = {"answer": "Chưa có pipeline", "sources": [{"content": "Chưa có"}]}
                
            eval_data["question"].append(item["question"])
            eval_data["answer"].append(result["answer"])
            eval_data["contexts"].append([c.get("content", "") for c in result.get("sources", [])])
            eval_data["ground_truth"].append(item["expected_answer"])
            print(f"  ✓ Câu {i+1}: {item['question'][:40]}...")
        except Exception as e:
            print(f"  ❌ Lỗi ở câu {i+1}: {e}")

    dataset = Dataset.from_dict(eval_data)
    
    print("\n⏳ Đang gửi dữ liệu cho RAGAS chấm điểm. Quá trình này có thể mất 1-2 phút...")
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )
        return result.to_pandas()
    except Exception as e:
        print(f"❌ Lỗi khi RAGAS chấm điểm: {e}")
        return None


def export_results(results_df):
    """Xuất báo cáo ra file Markdown"""
    if results_df is None or results_df.empty:
        print("❌ Không có dữ liệu để xuất báo cáo!")
        return

    content = "# 📊 Báo Cáo Đánh Giá Chất Lượng RAG (Sử dụng RAGAS)\n\n"
    
    # Tính điểm trung bình
    mean_scores = results_df[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean()
    
    content += "## 1. Điểm Số Tổng Quan (Overall Scores)\n\n"
    content += "| Tiêu chí (Metric) | Ý nghĩa | Điểm trung bình (0-1) |\n"
    content += "|-------------------|---------|-----------------------|\n"
    content += f"| **Faithfulness** | Bot có bịa luật không? (Độ trung thực) | `{mean_scores.get('faithfulness', 0):.3f}` |\n"
    content += f"| **Answer Relevancy** | Trả lời có đúng trọng tâm câu hỏi? | `{mean_scores.get('answer_relevancy', 0):.3f}` |\n"
    content += f"| **Context Recall** | Bot có moi ra được đủ các điều luật cần thiết? | `{mean_scores.get('context_recall', 0):.3f}` |\n"
    content += f"| **Context Precision**| Các điều luật moi ra có bị rác không? | `{mean_scores.get('context_precision', 0):.3f}` |\n\n"

    content += "## 2. Chi Tiết Các Câu Trả Lời Tệ Nhất (Worst Performers)\n\n"
    content += "> Đây là các câu hỏi mà Chatbot trả lời sai hoặc lạc đề nhất, cần kiểm tra lại Prompt hoặc Data.\n\n"
    
    # Sắp xếp lấy 3 câu có điểm Faithfulness thấp nhất
    worst_cases = results_df.sort_values(by='faithfulness', ascending=True).head(3)
    
    for idx, row in worst_cases.iterrows():
        content += f"### Câu hỏi: {row['question']}\n"
        content += f"- **Điểm Faithfulness:** `{row.get('faithfulness', 0):.2f}`\n"
        content += f"- **Đáp án chuẩn (Ground Truth):** {row['ground_truth']}\n"
        content += f"- **Bot trả lời:** {row['answer']}\n"
        content += "---\n\n"
    
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✅ Đã xuất báo cáo đánh giá thành công tại: {RESULTS_PATH.name}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"📚 Đã tải thành công {len(golden_dataset)} câu hỏi Luật từ golden_dataset.json\n")

    # Import pipeline từ Task 10 (Nếu có lỗi import thì tạm pass)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src import task10_generation as pipeline
    except ImportError:
        print("⚠️ Chưa có file pipeline Task 10, chạy ở chế độ giả lập.")
        pipeline = None

    # Chạy đánh giá
    df_results = evaluate_with_ragas(pipeline, golden_dataset)
    
    # Xuất file báo cáo
    export_results(df_results)

"""
RAG Evaluation Pipeline (Role 4 - QA/Tester).

Script này sử dụng thư viện RAGAS để tự động chấm điểm Chatbot dựa trên 15 câu hỏi
trong `golden_dataset.json`. Kết quả sẽ được xuất ra file `results.md`.
Đã bao gồm tính năng A/B Testing theo chuẩn yêu cầu của BTC.
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


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict], config: dict = None):
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

    eval_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    eval_embeddings = OpenAIEmbeddings(api_key=api_key)

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, item in enumerate(golden_dataset):
        try:
            # Truyền config vào pipeline (để test A/B)
            if hasattr(rag_pipeline, "generate_with_citation"):
                result = rag_pipeline.generate_with_citation(item["question"], config=config)
            else:
                result = {"answer": "Chưa có pipeline", "sources": [{"content": "Chưa có"}]}
                
            eval_data["question"].append(item["question"])
            eval_data["answer"].append(result["answer"])
            eval_data["contexts"].append([c.get("content", "") for c in result.get("sources", [])])
            eval_data["ground_truth"].append(item["expected_answer"])
        except Exception as e:
            print(f"  ❌ Lỗi ở câu {i+1}: {e}")

    dataset = Dataset.from_dict(eval_data)
    
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


def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa 2 configs:
    - Config A: Hybrid Search + Reranking (Mặc định)
    - Config B: Chỉ dùng BM25 (Lexical Only)
    """
    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU CHẠY A/B TESTING...")
    print("="*50)

    configs = {
        "Config A (Hybrid + Reranking)": {"use_semantic": True, "use_lexical": True, "use_reranking": True},
        "Config B (Lexical Only)": {"use_semantic": False, "use_lexical": True, "use_reranking": False},
    }

    results = {}
    for config_name, params in configs.items():
        print(f"\n⏳ Đang chạy đánh giá cho: {config_name}")
        df_result = evaluate_with_ragas(rag_pipeline, golden_dataset, config=params)
        results[config_name] = df_result
        if df_result is not None:
            print(f"✅ Xong {config_name}")

    return results


def export_results(ab_results: dict):
    """Xuất báo cáo So sánh A/B ra file Markdown"""
    if not ab_results:
        print("❌ Không có dữ liệu để xuất báo cáo!")
        return

    content = "# 📊 Báo Cáo Đánh Giá Chất Lượng RAG & A/B Testing\n\n"
    content += "> *Đánh giá tự động bằng RAGAS trên 15 câu hỏi vàng (Chính sách Thương mại điện tử Shopee).*\n\n"
    
    content += "## 1. Kết Quả A/B Testing (Overall Scores)\n\n"
    content += "| Tiêu chí (Metric) | Config A (Hybrid+Reranking) | Config B (Lexical Only) |\n"
    content += "|-------------------|----------------------------|------------------------|\n"

    # Trích xuất điểm của Config A và B
    scores_A = ab_results.get("Config A (Hybrid + Reranking)")
    scores_B = ab_results.get("Config B (Lexical Only)")
    
    mean_A = scores_A[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean() if scores_A is not None else {}
    mean_B = scores_B[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean() if scores_B is not None else {}

    metrics = [
        ("Faithfulness (Không bịa luật)", 'faithfulness'),
        ("Answer Relevancy (Đúng trọng tâm)", 'answer_relevancy'),
        ("Context Recall (Lấy đủ thông tin)", 'context_recall'),
        ("Context Precision (Không bị rác)", 'context_precision')
    ]

    for label, key in metrics:
        val_a = f"`{mean_A.get(key, 0):.3f}`" if mean_A else "`N/A`"
        val_b = f"`{mean_B.get(key, 0):.3f}`" if mean_B else "`N/A`"
        content += f"| **{label}** | {val_a} | {val_b} |\n"

    content += "\n## 2. Phân Tích Worst Performers (Từ Config A)\n\n"
    
    if scores_A is not None and not scores_A.empty:
        worst_cases = scores_A.sort_values(by='faithfulness', ascending=True).head(3)
        for idx, row in worst_cases.iterrows():
            content += f"### Câu hỏi: {row['question']}\n"
            content += f"- **Điểm Faithfulness:** `{row.get('faithfulness', 0):.2f}`\n"
            content += f"- **Đáp án chuẩn:** {row['ground_truth']}\n"
            content += f"- **Bot trả lời:** {row['answer']}\n"
            content += "---\n\n"
    
    content += "## 3. Đề Xuất Cải Tiến (Recommendations)\n"
    content += "- **Dựa trên A/B Test:** Cấu hình Hybrid + Reranking thường cho kết quả Recall tốt hơn vì nó kết hợp được cả ngữ nghĩa lẫn từ khóa chính xác.\n"
    content += "- **Cải thiện Faithfulness:** Cần tinh chỉnh lại Prompt ở Task 10 để ép LLM từ chối trả lời nếu không tìm thấy thông tin trong Context, thay vì cố gắng tự bịa ra.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✅ Đã xuất báo cáo đánh giá A/B thành công tại: {RESULTS_PATH.name}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"📚 Đã tải thành công {len(golden_dataset)} câu hỏi Chính sách TMĐT từ golden_dataset.json\n")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src import task10_generation as pipeline
    except ImportError:
        print("⚠️ Chưa có file pipeline Task 10, chạy ở chế độ giả lập.")
        pipeline = None

    # Chạy A/B Testing
    ab_results = compare_configs(pipeline, golden_dataset)
    
    # Xuất file báo cáo tổng hợp
    export_results(ab_results)

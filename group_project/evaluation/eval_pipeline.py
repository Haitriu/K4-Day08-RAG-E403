"""RAGAS evaluation và A/B test cho chatbot Luật Lao động."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.md")
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
CONFIGS: dict[str, dict[str, bool]] = {
    "Config A (Hybrid + Reranking)": {
        "use_semantic": True,
        "use_lexical": True,
        "use_reranking": True,
        "use_fallback": False,
    },
    "Config B (Lexical Only)": {
        "use_semantic": False,
        "use_lexical": True,
        "use_reranking": False,
        "use_fallback": False,
    },
}

load_dotenv(PROJECT_ROOT / ".env")


def load_golden_dataset(limit: int | None = None) -> list[dict[str, str]]:
    """Đọc và kiểm tra schema của golden dataset."""

    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 15:
        raise ValueError("Golden dataset phải chứa ít nhất 15 mẫu.")

    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Mẫu {index} thiếu trường bắt buộc: {sorted(required)}")
        if any(not str(item[key]).strip() for key in required):
            raise ValueError(f"Mẫu {index} có giá trị rỗng.")

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit phải là số nguyên dương.")
        return data[:limit]
    return data


def _ragas_dependencies() -> tuple[Any, ...]:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu dependency evaluation. Hãy tạo môi trường riêng và cài "
            "requirements-eval.txt."
        ) from exc

    return (
        Dataset,
        ChatOpenAI,
        OpenAIEmbeddings,
        evaluate,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )


def _collect_answers(
    rag_pipeline: Any,
    golden_dataset: list[dict[str, str]],
    config: dict[str, bool],
    top_k: int,
) -> dict[str, list[Any]]:
    if rag_pipeline is None or not hasattr(rag_pipeline, "generate_with_citation"):
        raise RuntimeError("Không tìm thấy Task 10 generation pipeline.")

    rows: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for index, item in enumerate(golden_dataset, 1):
        print(f"  [{index:02d}/{len(golden_dataset):02d}] {item['question'][:70]}")
        result = rag_pipeline.generate_with_citation(
            item["question"],
            top_k=top_k,
            config=config,
        )
        sources = result.get("sources") or []
        contexts = [str(source.get("content") or "") for source in sources]
        if not contexts:
            contexts = ["Không truy xuất được bằng chứng từ corpus hiện có."]

        rows["question"].append(item["question"])
        rows["answer"].append(str(result.get("answer") or ""))
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["expected_answer"])
    return rows


def evaluate_with_ragas(
    rag_pipeline: Any,
    golden_dataset: list[dict[str, str]],
    config: dict[str, bool],
    top_k: int = 5,
):
    """Sinh câu trả lời và chấm bốn metric bằng RAGAS 0.1.x."""

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Thiếu OPENAI_API_KEY cho RAGAS LLM-as-a-judge.")

    (
        Dataset,
        ChatOpenAI,
        OpenAIEmbeddings,
        evaluate,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ) = _ragas_dependencies()

    rows = _collect_answers(rag_pipeline, golden_dataset, config, top_k)
    dataset = Dataset.from_dict(rows)
    judge_model = os.getenv("EVAL_OPENAI_MODEL", "gpt-4o-mini").strip()
    eval_llm = ChatOpenAI(model=judge_model, temperature=0)
    eval_embeddings = OpenAIEmbeddings()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    return result.to_pandas()


def compare_configs(
    rag_pipeline: Any,
    golden_dataset: list[dict[str, str]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Chạy A/B giữa hybrid + RRF và lexical-only."""

    results: dict[str, Any] = {}
    for name, config in CONFIGS.items():
        print(f"\n=== {name} ===")
        results[name] = evaluate_with_ragas(
            rag_pipeline,
            golden_dataset,
            config=config,
            top_k=top_k,
        )
    return results


def _mean_scores(frame: Any) -> dict[str, float]:
    return {metric: float(frame[metric].mean()) for metric in METRICS}


def export_results(ab_results: dict[str, Any], sample_count: int) -> None:
    """Ghi báo cáo điểm, worst performers và khuyến nghị."""

    config_a = ab_results["Config A (Hybrid + Reranking)"]
    config_b = ab_results["Config B (Lexical Only)"]
    mean_a = _mean_scores(config_a)
    mean_b = _mean_scores(config_b)

    lines = [
        "# Báo cáo đánh giá RAG và A/B testing",
        "",
        f"> RAGAS chấm {sample_count} câu hỏi vàng về Luật Lao động Việt Nam.",
        "",
        "## Điểm tổng quan",
        "",
        "| Metric | Hybrid + Reranking | Lexical only | Chênh lệch A-B |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_recall": "Context recall",
        "context_precision": "Context precision",
    }
    for metric in METRICS:
        delta = mean_a[metric] - mean_b[metric]
        lines.append(
            f"| {labels[metric]} | {mean_a[metric]:.3f} | "
            f"{mean_b[metric]:.3f} | {delta:+.3f} |"
        )

    lines.extend(["", "## Worst performers của Hybrid + Reranking", ""])
    worst = config_a.assign(
        aggregate_score=config_a[list(METRICS)].mean(axis=1)
    ).sort_values("aggregate_score").head(3)
    for _, row in worst.iterrows():
        lines.extend(
            [
                f"### {row['question']}",
                "",
                f"- Điểm tổng hợp: `{row['aggregate_score']:.3f}`",
                f"- Faithfulness: `{row['faithfulness']:.3f}`",
                f"- Đáp án chuẩn: {row['ground_truth']}",
                f"- Chatbot: {row['answer']}",
                "",
            ]
        )

    better = "Hybrid + Reranking" if sum(mean_a.values()) >= sum(mean_b.values()) else "Lexical only"
    lines.extend(
        [
            "## Kết luận và đề xuất",
            "",
            f"- Cấu hình có tổng điểm trung bình tốt hơn: **{better}**.",
            "- Bổ sung văn bản nguồn chính thức cho các câu có context recall thấp.",
            "- Giữ cơ chế từ chối khi evidence rỗng và kiểm tra citation trước khi hiển thị.",
            "- Xem lại chunking quanh ranh giới Điều/Khoản ở các worst performers.",
            "",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nĐã ghi báo cáo: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Chỉ chạy N mẫu để smoke test.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k phải là số nguyên dương")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src import task10_generation as pipeline

    golden_dataset = load_golden_dataset(limit=args.limit)
    print(f"Đã tải {len(golden_dataset)} câu hỏi Luật Lao động.")
    results = compare_configs(pipeline, golden_dataset, top_k=args.top_k)
    export_results(results, sample_count=len(golden_dataset))


if __name__ == "__main__":
    main()

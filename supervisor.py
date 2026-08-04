"""Public supervisor entry point for the labour-law RAG application."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import GenerationResult, generate_with_citation


def retrieve_context(
    query: str,
    top_k: int = 5,
    *,
    config: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Run the unified retrieval pipeline without calling an LLM."""

    return retrieve(query, top_k=top_k, config=config)


def ask(
    query: str,
    top_k: int = 5,
    *,
    config: dict[str, bool] | None = None,
) -> GenerationResult:
    """Run retrieval and cited generation through one stable interface."""

    return generate_with_citation(query, top_k=top_k, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese labour-law RAG supervisor")
    parser.add_argument("query", help="Câu hỏi cần tra cứu")
    parser.add_argument("--top-k", type=int, default=5, help="Số chunks tối đa")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Chỉ chạy retrieval, không gọi LLM",
    )
    args = parser.parse_args()

    result: object
    if args.retrieve_only:
        result = retrieve_context(args.query, top_k=args.top_k)
    else:
        result = ask(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

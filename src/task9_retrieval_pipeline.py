"""Task 9 — Hybrid retrieval, reranking và PageIndex fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rrf_rerank
from .task8_pageindex_vectorless import pageindex_search


DEFAULT_CONFIG = {
    "use_semantic": True,
    "use_lexical": True,
    "use_reranking": True,
    "use_fallback": True,
}


def _safe_search(
    name: str,
    search_fn: Callable[..., list[dict]],
    query: str,
    top_k: int,
) -> list[dict]:
    """Cô lập lỗi một retriever để nhánh còn lại vẫn có thể phục vụ."""

    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"[WARN] {name} search failed ({type(exc).__name__}).")
        return []


def _mark_hybrid(results: list[dict], score_name: str) -> list[dict]:
    marked: list[dict] = []
    for result in results:
        item = result.copy()
        item[score_name] = item.get("score", 0.0)
        item["source"] = "hybrid"
        marked.append(item)
    return marked


def retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.48,
    config: dict[str, bool] | None = None,
) -> list[dict]:
    """Trả về top-k chunks từ hybrid search hoặc PageIndex fallback.

    Fallback dùng cosine gốc cao nhất từ semantic search, không dùng điểm RRF.
    Khi cấu hình A/B cố ý tắt semantic, threshold cosine không được áp dụng;
    fallback chỉ chạy nếu nhánh retrieval được chọn trả về rỗng.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương.")
    if not isinstance(score_threshold, (int, float)) or score_threshold < 0:
        raise ValueError("score_threshold phải là số không âm.")

    options = {**DEFAULT_CONFIG, **(config or {})}
    candidate_count = top_k * 2
    use_semantic = bool(options["use_semantic"])
    use_lexical = bool(options["use_lexical"])

    jobs: dict[str, tuple[Callable[..., list[dict]], str]] = {}
    if use_semantic:
        jobs["semantic"] = (semantic_search, "Semantic")
    if use_lexical:
        jobs["lexical"] = (lexical_search, "Lexical")

    search_results: dict[str, list[dict]] = {"semantic": [], "lexical": []}
    if jobs:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                name: executor.submit(
                    _safe_search,
                    label,
                    search_fn,
                    query.strip(),
                    candidate_count,
                )
                for name, (search_fn, label) in jobs.items()
            }
            for name, future in futures.items():
                search_results[name] = future.result()

    semantic_results = search_results["semantic"]
    lexical_results = search_results["lexical"]

    if options["use_reranking"]:
        final_results = rrf_rerank(
            semantic_results,
            lexical_results,
            top_k=top_k,
        )
    elif use_lexical:
        final_results = _mark_hybrid(
            lexical_results[:top_k],
            "original_lexical_score",
        )
    else:
        final_results = _mark_hybrid(
            semantic_results[:top_k],
            "original_semantic_score",
        )

    best_cosine = (
        float(semantic_results[0].get("score", 0.0))
        if semantic_results
        else 0.0
    )
    below_threshold = use_semantic and best_cosine < float(score_threshold)
    should_fallback = bool(options["use_fallback"]) and (
        not final_results or below_threshold
    )

    if should_fallback:
        print(
            f"[INFO] Best semantic cosine {best_cosine:.3f} is below "
            f"threshold {float(score_threshold):.3f}; trying PageIndex."
        )
        fallback_results = _safe_search(
            "PageIndex",
            pageindex_search,
            query.strip(),
            top_k,
        )
        if fallback_results:
            return fallback_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    demo_results = retrieve("Thời gian thử việc tối đa là bao lâu?")
    print(f"Retrieved {len(demo_results)} chunks.")

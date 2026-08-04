"""Unit tests for Role 3 Task 10 without network calls."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src import task10_generation as task10


def _legal_chunks() -> list[dict]:
    return [
        {
            "content": "Thời gian thử việc tối đa là 60 ngày.",
            "score": 0.95,
            "source": "hybrid",
            "metadata": {
                "document_title": "Bộ luật Lao động",
                "source": "bo-luat-lao-dong-2019.pdf",
                "year": 2019,
                "article": "Điều 25",
                "type": "legal",
            },
        }
    ]


def test_reorder_for_llm_uses_expected_pattern_without_mutating_input() -> None:
    chunks = [{"content": f"Chunk {index}"} for index in range(1, 6)]
    original = list(chunks)

    reordered = task10.reorder_for_llm(chunks)

    assert [item["content"] for item in reordered] == [
        "Chunk 1",
        "Chunk 3",
        "Chunk 5",
        "Chunk 4",
        "Chunk 2",
    ]
    assert chunks == original
    assert reordered is not chunks


def test_format_context_contains_source_metadata_and_citation() -> None:
    context = task10.format_context(_legal_chunks())

    assert "bo-luat-lao-dong-2019.pdf" in context
    assert "[Bộ luật Lao động, 2019]" in context
    assert "Điều 25" in context


@pytest.mark.parametrize("retrieval_behavior", [[], NotImplementedError("pending")])
def test_unavailable_retrieval_returns_structured_refusal(retrieval_behavior: object) -> None:
    if isinstance(retrieval_behavior, Exception):
        retrieve_mock = MagicMock(side_effect=retrieval_behavior)
    else:
        retrieve_mock = MagicMock(return_value=retrieval_behavior)

    with (
        patch.object(task10, "retrieve", retrieve_mock),
        patch.object(task10, "_generate_answer") as generate_mock,
    ):
        result = task10.generate_with_citation("Thời gian thử việc là bao lâu?")

    assert result == {
        "answer": task10.REFUSAL_ANSWER,
        "sources": [],
        "retrieval_source": "none",
    }
    generate_mock.assert_not_called()


def test_generate_with_citation_returns_expected_schema_without_network() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Tối đa 60 ngày [Bộ luật Lao động, 2019]."
                )
            )
        ]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with (
        patch.object(task10, "retrieve", return_value=_legal_chunks()),
        patch.object(task10, "OpenAI", return_value=client) as client_factory,
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True),
    ):
        result = task10.generate_with_citation("Thời gian thử việc là bao lâu?")

    assert result["answer"].endswith("[Bộ luật Lao động, 2019].")
    assert result["sources"] == _legal_chunks()
    assert result["retrieval_source"] == "hybrid"
    client_factory.assert_called_once_with(
        api_key="test-key",
        base_url=task10.OPENROUTER_BASE_URL,
    )
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == task10.OPENROUTER_MODEL
    assert "bo-luat-lao-dong-2019.pdf" in request["messages"][1]["content"]


def test_missing_api_key_with_evidence_raises_safe_configuration_error() -> None:
    with (
        patch.object(task10, "retrieve", return_value=_legal_chunks()),
        patch.object(task10, "OpenAI") as client_factory,
        patch.dict(os.environ, {}, clear=True),
    ):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY.*OPENAI_API_KEY"):
            task10.generate_with_citation("Thời gian thử việc là bao lâu?")

    client_factory.assert_not_called()


def test_openrouter_rate_limit_falls_back_to_openai() -> None:
    request = httpx.Request("POST", task10.OPENROUTER_BASE_URL)
    response = httpx.Response(429, request=request)
    rate_limit_error = task10.RateLimitError(
        "rate limited",
        response=response,
        body=None,
    )

    with (
        patch.object(
            task10,
            "_call_llm",
            side_effect=[rate_limit_error, "Câu trả lời từ OpenAI"],
        ) as call_mock,
        patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "openrouter-test-key",
                "OPENAI_API_KEY": "openai-test-key",
            },
            clear=True,
        ),
    ):
        answer = task10._generate_answer("Câu hỏi kiểm thử")

    assert answer == "Câu trả lời từ OpenAI"
    assert call_mock.call_count == 2
    assert call_mock.call_args_list[0].kwargs["provider_name"] == "OpenRouter"
    assert call_mock.call_args_list[1].kwargs["provider_name"] == "OpenAI"


@pytest.mark.parametrize(
    ("query", "top_k"),
    [("", 5), ("   ", 5), ("Hỏi luật", 0), ("Hỏi luật", -1), ("Hỏi luật", True)],
)
def test_invalid_inputs_are_rejected(query: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        task10.generate_with_citation(query, top_k=top_k)

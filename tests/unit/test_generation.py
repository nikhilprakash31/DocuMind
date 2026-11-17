"""Unit tests for RAG generation prompts and RAGGenerationChain."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.core.exceptions import GenerationConnectionError, GenerationError
from src.core.models import RetrievalResult
from src.generation.chain import RAGGenerationChain
from src.generation.prompts import (
    FALLBACK_EMPTY_RETRIEVAL,
    FALLBACK_INSUFFICIENT_CONTEXT,
    extract_citations,
    format_context_passages,
)
from src.retrieval.retriever import VectorRetriever


@pytest.fixture
def sample_sources() -> list[RetrievalResult]:
    """Sample retrieval results for testing context formatting and citations."""
    return [
        RetrievalResult(
            chunk_id="doc_alpha_c0",
            doc_id="doc_alpha",
            filename="arch_guide.pdf",
            page=1,
            chunk_index=0,
            content="Hierarchical Navigable Small World graphs provide O(log N) search complexity.",
            distance=0.15,
            score=0.85,
        ),
        RetrievalResult(
            chunk_id="doc_alpha_c1",
            doc_id="doc_alpha",
            filename="arch_guide.pdf",
            page=2,
            chunk_index=1,
            content="Default chunk size is 800 characters with 150 characters overlap buffer.",
            distance=0.22,
            score=0.78,
        ),
    ]


def test_format_context_passages_empty():
    """Verify empty input returns empty string."""
    assert format_context_passages([]) == ""


def test_format_context_passages_structure(sample_sources):
    """Verify formatted passages contain filename, page number, and content."""
    formatted = format_context_passages(sample_sources)
    assert "[Passage 1 | Source: arch_guide.pdf | Page: 1]" in formatted
    assert "Hierarchical Navigable Small World graphs" in formatted
    assert "[Passage 2 | Source: arch_guide.pdf | Page: 2]" in formatted
    assert "Default chunk size is 800 characters" in formatted
    assert "---" in formatted


def test_extract_citations_bracket_format(sample_sources):
    """Verify citation extraction and verification against actual retrieved chunks."""
    answer = "HNSW graphs offer logarithmic search [Page 1, arch_guide.pdf]."
    citations = extract_citations(answer, sample_sources)
    assert len(citations) == 1
    assert citations[0].filename == "arch_guide.pdf"
    assert citations[0].page == 1
    assert citations[0].chunk_id == "doc_alpha_c0"
    assert citations[0].doc_id == "doc_alpha"
    assert citations[0].is_verified is True


def test_extract_citations_insufficient_context_returns_empty(sample_sources):
    """Verify no citations are extracted when answer states insufficient information."""
    answer = FALLBACK_INSUFFICIENT_CONTEXT
    citations = extract_citations(answer, sample_sources)
    assert len(citations) == 0


def test_extract_citations_unmatched_reference_flagged_unverified(sample_sources):
    """Verify fabricated/unmatched page citations are flagged as is_verified=False."""
    answer = "Unrelated claim with invalid reference [Page 99, fake_doc.pdf]."
    citations = extract_citations(answer, sample_sources)
    assert len(citations) == 1
    assert citations[0].page == 99
    assert citations[0].filename == "fake_doc.pdf"
    assert citations[0].chunk_id == ""
    assert citations[0].is_verified is False
    assert "Unverified citation" in citations[0].snippet


@patch("src.generation.chain.ChatOllama")
def test_generate_from_context_empty_chunks(mock_chat_cls):
    """Verify empty chunks returns fallback answer without calling LLM."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm

    mock_retriever = MagicMock(spec=VectorRetriever)
    chain = RAGGenerationChain(retriever=mock_retriever)

    response = chain.generate_from_context(query="What is HNSW?", retrieved_chunks=[])

    assert response.answer == FALLBACK_EMPTY_RETRIEVAL
    assert not response.is_grounded
    assert len(response.citations) == 0
    mock_llm.invoke.assert_not_called()


@patch("src.generation.chain.ChatOllama")
def test_generate_from_context_grounded_answer(mock_chat_cls, sample_sources):
    """Verify grounded answer synthesis invokes LLM and maps citations."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content="HNSW graphs enable logarithmic search complexity [Page 1, arch_guide.pdf]."
    )

    mock_retriever = MagicMock(spec=VectorRetriever)
    chain = RAGGenerationChain(retriever=mock_retriever)

    response = chain.generate_from_context(
        query="What is the search complexity of HNSW?",
        retrieved_chunks=sample_sources,
    )

    assert response.is_grounded
    assert "logarithmic search complexity" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].page == 1
    assert response.latency_seconds > 0.0
    mock_llm.invoke.assert_called_once()


@patch("src.generation.chain.ChatOllama")
def test_generate_insufficient_context_flagged(mock_chat_cls, sample_sources):
    """Verify when LLM indicates missing facts, is_grounded is set to False."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content=FALLBACK_INSUFFICIENT_CONTEXT
    )

    mock_retriever = MagicMock(spec=VectorRetriever)
    chain = RAGGenerationChain(retriever=mock_retriever)

    response = chain.generate_from_context(
        query="What is the GDP of Antarctica?",
        retrieved_chunks=sample_sources,
    )

    assert not response.is_grounded
    assert FALLBACK_INSUFFICIENT_CONTEXT in response.answer
    assert len(response.citations) == 0


@patch("src.generation.chain.ChatOllama")
def test_full_pipeline_generate(mock_chat_cls, sample_sources):
    """Verify full generate() method retrieves chunks then calls LLM."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content="Chunk size is 800 characters [Page 2, arch_guide.pdf]."
    )

    mock_retriever = MagicMock(spec=VectorRetriever)
    mock_retriever.retrieve.return_value = sample_sources

    chain = RAGGenerationChain(retriever=mock_retriever)
    response = chain.generate(query="What is the default chunk size?", top_k=2)

    mock_retriever.retrieve.assert_called_once_with(
        query="What is the default chunk size?",
        top_k=2,
        filter_criteria=None,
    )
    assert response.is_grounded
    assert len(response.citations) == 1
    assert response.citations[0].page == 2


@patch("src.generation.chain.ChatOllama")
def test_generate_invalid_inputs(mock_chat_cls):
    """Verify empty query raises ValueError."""
    mock_retriever = MagicMock(spec=VectorRetriever)
    chain = RAGGenerationChain(retriever=mock_retriever)

    with pytest.raises(ValueError, match="cannot be empty"):
        chain.generate("")
    with pytest.raises(ValueError, match="cannot be empty"):
        chain.generate("   \n\t")


@patch("src.generation.chain.ChatOllama")
def test_generation_connection_error_handling(mock_chat_cls, sample_sources):
    """Verify network or connection errors wrap into GenerationConnectionError."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.side_effect = ConnectionError("Connection refused on port 11434")

    mock_retriever = MagicMock(spec=VectorRetriever)
    chain = RAGGenerationChain(retriever=mock_retriever)

    with pytest.raises(GenerationConnectionError, match="Failed to connect to Ollama"):
        chain.generate_from_context("query", sample_sources)


@patch("src.generation.chain.ChatOllama")
def test_general_generation_error_handling(mock_chat_cls, sample_sources):
    """Verify general runtime errors wrap into GenerationError."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.side_effect = RuntimeError("GPU out of memory")

    mock_retriever = MagicMock(spec=VectorRetriever)
    chain = RAGGenerationChain(retriever=mock_retriever)

    with pytest.raises(GenerationError, match="RAG text generation failed"):
        chain.generate_from_context("query", sample_sources)

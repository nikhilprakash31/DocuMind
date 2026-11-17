"""Unit tests for NomicEmbeddingAdapter."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import (
    EmbeddingConnectionError,
    EmbeddingDimensionError,
    EmbeddingError,
)
from src.storage.embeddings import (
    DOC_PREFIX,
    EXPECTED_EMBEDDING_DIM,
    QUERY_PREFIX,
    NomicEmbeddingAdapter,
)


def test_format_document_text_adds_prefix():
    """Verify format_document_text prepends search_document: prefix."""
    raw = "This is a document chunk about RAG."
    formatted = NomicEmbeddingAdapter.format_document_text(raw)
    assert formatted == f"{DOC_PREFIX}{raw}"


def test_format_document_text_idempotent():
    """Verify format_document_text does not apply prefix twice."""
    raw = "search_document: Already prefixed text."
    formatted = NomicEmbeddingAdapter.format_document_text(raw)
    assert formatted == raw
    assert formatted.count("search_document:") == 1

    # With varied casing and whitespace
    raw_case = "  SEARCH_DOCUMENT:   Whitespace and case variance."
    formatted_case = NomicEmbeddingAdapter.format_document_text(raw_case)
    total_matches = (
        formatted_case.count("SEARCH_DOCUMENT:") + formatted_case.count("search_document:")
    )
    assert total_matches == 1


def test_format_document_text_replaces_query_prefix():
    """Verify format_document_text corrects an accidental search_query: prefix."""
    raw = "search_query: Accidental query prefix on a document."
    formatted = NomicEmbeddingAdapter.format_document_text(raw)
    assert formatted.startswith(DOC_PREFIX)
    assert "search_query:" not in formatted


def test_format_query_text_adds_prefix():
    """Verify format_query_text prepends search_query: prefix."""
    query = "What is dense retrieval?"
    formatted = NomicEmbeddingAdapter.format_query_text(query)
    assert formatted == f"{QUERY_PREFIX}{query}"


def test_format_query_text_idempotent():
    """Verify format_query_text does not apply prefix twice."""
    query = "search_query: What is vector search?"
    formatted = NomicEmbeddingAdapter.format_query_text(query)
    assert formatted == query
    assert formatted.count("search_query:") == 1


def test_format_query_text_replaces_doc_prefix():
    """Verify format_query_text corrects an accidental search_document: prefix."""
    query = "search_document: What is semantic indexing?"
    formatted = NomicEmbeddingAdapter.format_query_text(query)
    assert formatted.startswith(QUERY_PREFIX)
    assert "search_document:" not in formatted


def test_empty_query_raises_value_error():
    """Verify empty or whitespace-only query raises ValueError."""
    adapter = NomicEmbeddingAdapter()
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        adapter.embed_query("")
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        adapter.embed_query("   \n\t  ")


def test_embed_documents_empty_list():
    """Verify empty document list returns empty list without calling Ollama."""
    adapter = NomicEmbeddingAdapter()
    assert adapter.embed_documents([]) == []


@patch("src.storage.embeddings.OllamaEmbeddings")
def test_embed_documents_success(mock_ollama_cls):
    """Verify successful embedding generation with prefixing and dimension check."""
    mock_instance = MagicMock()
    mock_ollama_cls.return_value = mock_instance
    dummy_vec = [0.1] * EXPECTED_EMBEDDING_DIM
    mock_instance.embed_documents.return_value = [dummy_vec, dummy_vec]

    adapter = NomicEmbeddingAdapter()
    results = adapter.embed_documents(["First chunk", "Second chunk"])

    assert len(results) == 2
    assert len(results[0]) == EXPECTED_EMBEDDING_DIM
    # Verify underlying embed_documents was called with prefixed texts
    mock_instance.embed_documents.assert_called_once_with([
        f"{DOC_PREFIX}First chunk",
        f"{DOC_PREFIX}Second chunk",
    ])


@patch("src.storage.embeddings.OllamaEmbeddings")
def test_embed_query_success(mock_ollama_cls):
    """Verify successful query embedding with prefixing and dimension check."""
    mock_instance = MagicMock()
    mock_ollama_cls.return_value = mock_instance
    dummy_vec = [0.05] * EXPECTED_EMBEDDING_DIM
    mock_instance.embed_query.return_value = dummy_vec

    adapter = NomicEmbeddingAdapter()
    result = adapter.embed_query("sample query")

    assert len(result) == EXPECTED_EMBEDDING_DIM
    mock_instance.embed_query.assert_called_once_with(f"{QUERY_PREFIX}sample query")


@patch("src.storage.embeddings.OllamaEmbeddings")
def test_embed_documents_dimension_mismatch(mock_ollama_cls):
    """Verify EmbeddingDimensionError is raised when output dim is not 768."""
    mock_instance = MagicMock()
    mock_ollama_cls.return_value = mock_instance
    bad_vec = [0.1] * 512  # Unexpected dimension
    mock_instance.embed_documents.return_value = [bad_vec]

    adapter = NomicEmbeddingAdapter()
    with pytest.raises(EmbeddingDimensionError) as exc_info:
        adapter.embed_documents(["Sample text"])
    assert exc_info.value.expected == 768
    assert exc_info.value.received == 512


@patch("src.storage.embeddings.OllamaEmbeddings")
def test_embed_query_dimension_mismatch(mock_ollama_cls):
    """Verify EmbeddingDimensionError is raised for query with bad dimension."""
    mock_instance = MagicMock()
    mock_ollama_cls.return_value = mock_instance
    bad_vec = [0.1] * 384
    mock_instance.embed_query.return_value = bad_vec

    adapter = NomicEmbeddingAdapter()
    with pytest.raises(EmbeddingDimensionError) as exc_info:
        adapter.embed_query("Query")
    assert exc_info.value.expected == 768
    assert exc_info.value.received == 384


@patch("src.storage.embeddings.OllamaEmbeddings")
def test_connection_error_handling(mock_ollama_cls):
    """Verify connection failure wraps into EmbeddingConnectionError."""
    mock_instance = MagicMock()
    mock_ollama_cls.return_value = mock_instance
    mock_instance.embed_documents.side_effect = ConnectionError("Connection refused by host")

    adapter = NomicEmbeddingAdapter()
    with pytest.raises(EmbeddingConnectionError, match="Failed to connect to Ollama"):
        adapter.embed_documents(["Doc text"])


@patch("src.storage.embeddings.OllamaEmbeddings")
def test_general_embedding_error_handling(mock_ollama_cls):
    """Verify other runtime errors wrap into EmbeddingError."""
    mock_instance = MagicMock()
    mock_ollama_cls.return_value = mock_instance
    mock_instance.embed_query.side_effect = RuntimeError("Model execution failed")

    adapter = NomicEmbeddingAdapter()
    with pytest.raises(EmbeddingError, match="Ollama query embedding failed"):
        adapter.embed_query("Query text")

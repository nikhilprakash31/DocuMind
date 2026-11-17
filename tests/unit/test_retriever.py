"""Unit tests for VectorRetriever."""

from unittest.mock import MagicMock

import pytest

from src.core.models import RetrievalResult
from src.retrieval.retriever import DEFAULT_TOP_K, VectorRetriever
from src.storage.vector_store import ChromaVectorStore


@pytest.fixture
def mock_vector_store():
    """Mock ChromaVectorStore providing sample retrieval results."""
    store = MagicMock(spec=ChromaVectorStore)

    # Return results in non-sorted order to verify retriever enforces sorting
    unsorted_results = [
        RetrievalResult(
            chunk_id="doc1_c1",
            doc_id="doc1",
            filename="report.pdf",
            page=2,
            chunk_index=1,
            content="Page 2 text on revenue.",
            distance=0.45,
            score=0.55,
        ),
        RetrievalResult(
            chunk_id="doc1_c0",
            doc_id="doc1",
            filename="report.pdf",
            page=1,
            chunk_index=0,
            content="Page 1 text on architecture.",
            distance=0.12,
            score=0.88,
        ),
    ]

    store.query_similar.return_value = unsorted_results
    return store


def test_retriever_default_top_k(mock_vector_store):
    """Verify retriever defaults to top_k = 4."""
    retriever = VectorRetriever(vector_store=mock_vector_store)
    assert retriever.default_top_k == DEFAULT_TOP_K

    retriever.retrieve("financial revenue query")
    mock_vector_store.query_similar.assert_called_once_with(
        query="financial revenue query",
        top_k=DEFAULT_TOP_K,
        filter_criteria=None,
    )


def test_retriever_custom_top_k(mock_vector_store):
    """Verify custom top_k overrides default."""
    retriever = VectorRetriever(vector_store=mock_vector_store)
    retriever.retrieve("query", top_k=10)
    mock_vector_store.query_similar.assert_called_once_with(
        query="query",
        top_k=10,
        filter_criteria=None,
    )


def test_retriever_sorts_by_distance_ascending(mock_vector_store):
    """Verify results are sorted with smallest distance (highest similarity) first."""
    retriever = VectorRetriever(vector_store=mock_vector_store)
    results = retriever.retrieve("query")

    assert len(results) == 2
    assert results[0].distance == 0.12
    assert results[0].chunk_id == "doc1_c0"
    assert results[1].distance == 0.45
    assert results[1].chunk_id == "doc1_c1"


def test_retrieve_for_document(mock_vector_store):
    """Verify retrieve_for_document applies doc_id filter."""
    retriever = VectorRetriever(vector_store=mock_vector_store)
    retriever.retrieve_for_document(query="test", doc_id="doc-abc", top_k=2)

    mock_vector_store.query_similar.assert_called_once_with(
        query="test",
        top_k=2,
        filter_criteria={"doc_id": "doc-abc"},
    )


def test_retrieve_for_filename(mock_vector_store):
    """Verify retrieve_for_filename applies filename filter."""
    retriever = VectorRetriever(vector_store=mock_vector_store)
    retriever.retrieve_for_filename(query="test", filename="handbook.pdf", top_k=3)

    mock_vector_store.query_similar.assert_called_once_with(
        query="test",
        top_k=3,
        filter_criteria={"filename": "handbook.pdf"},
    )


def test_retriever_invalid_inputs(mock_vector_store):
    """Verify input validation for invalid queries and parameters."""
    retriever = VectorRetriever(vector_store=mock_vector_store)

    # Empty query
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("")
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("   ")

    # Invalid top_k
    with pytest.raises(ValueError, match="must be greater than zero"):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(ValueError, match="must be greater than zero"):
        retriever.retrieve("query", top_k=-5)

    # Empty doc_id / filename in helpers
    with pytest.raises(ValueError, match="doc_id cannot be empty"):
        retriever.retrieve_for_document("query", doc_id="")
    with pytest.raises(ValueError, match="filename cannot be empty"):
        retriever.retrieve_for_filename("query", filename="")

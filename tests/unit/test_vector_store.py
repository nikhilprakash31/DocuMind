"""Unit tests for ChromaVectorStore."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.storage.embeddings import EXPECTED_EMBEDDING_DIM, NomicEmbeddingAdapter
from src.storage.vector_store import ChromaVectorStore


@pytest.fixture
def mock_embedding_adapter():
    """Mock embedding adapter generating deterministic 768-dim vectors."""
    adapter = MagicMock(spec=NomicEmbeddingAdapter)
    adapter.expected_dim = EXPECTED_EMBEDDING_DIM
    adapter.model = "nomic-embed-text"

    def _embed_docs(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * EXPECTED_EMBEDDING_DIM
            if "alpha" in text.lower():
                vec[0] = 1.0
            elif "beta" in text.lower():
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            vectors.append(vec)
        return vectors

    def _embed_query(query: str) -> list[float]:
        vec = [0.0] * EXPECTED_EMBEDDING_DIM
        if "alpha" in query.lower():
            vec[0] = 1.0
        elif "beta" in query.lower():
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        return vec

    adapter.embed_documents.side_effect = _embed_docs
    adapter.embed_query.side_effect = _embed_query
    return adapter


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Sample chunk documents with complete metadata."""
    return [
        Document(
            page_content="Alpha system architecture for document indexing.",
            metadata={
                "source": "guide.pdf",
                "filename": "guide.pdf",
                "page": 1,
                "total_pages": 3,
                "doc_id": "doc-alpha",
                "doc_hash": "hash111",
                "file_size_bytes": 1024,
                "upload_timestamp": "2026-11-09T10:00:00Z",
                "chunk_index": 0,
                "chunk_id": "doc-alpha_c0",
                "char_count": 48,
            },
        ),
        Document(
            page_content="Beta pipeline processing for embeddings generation.",
            metadata={
                "source": "guide.pdf",
                "filename": "guide.pdf",
                "page": 2,
                "total_pages": 3,
                "doc_id": "doc-alpha",
                "doc_hash": "hash111",
                "file_size_bytes": 1024,
                "upload_timestamp": "2026-11-09T10:00:00Z",
                "chunk_index": 1,
                "chunk_id": "doc-alpha_c1",
                "char_count": 52,
            },
        ),
        Document(
            page_content="Gamma document reference notes.",
            metadata={
                "source": "notes.pdf",
                "filename": "notes.pdf",
                "page": 1,
                "total_pages": 1,
                "doc_id": "doc-gamma",
                "doc_hash": "hash222",
                "file_size_bytes": 2048,
                "upload_timestamp": "2026-11-09T11:00:00Z",
                "chunk_index": 0,
                "chunk_id": "doc-gamma_c0",
                "char_count": 31,
            },
        ),
    ]


def test_vector_store_initialization(tmp_path: Path, mock_embedding_adapter):
    """Verify initialization creates directory and empty collection."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_col",
        embedding_adapter=mock_embedding_adapter,
    )
    assert store.count() == 0
    stats = store.get_collection_stats()
    assert stats["collection_name"] == "test_col"
    assert stats["hnsw_space"] == "cosine"
    assert stats["total_chunks"] == 0


def test_upsert_chunks_and_persistence(tmp_path: Path, mock_embedding_adapter, sample_chunks):
    """Verify upserting chunks persists data across client instances."""
    db_path = tmp_path / "persistent_chroma"

    # Instance 1: Upsert chunks
    store1 = ChromaVectorStore(
        persist_directory=db_path,
        collection_name="docs_col",
        embedding_adapter=mock_embedding_adapter,
    )
    upserted_count = store1.upsert_chunks(sample_chunks, batch_size=2)
    assert upserted_count == 3
    assert store1.count() == 3

    # Instance 2: Re-open from disk and verify persistence
    store2 = ChromaVectorStore(
        persist_directory=db_path,
        collection_name="docs_col",
        embedding_adapter=mock_embedding_adapter,
    )
    assert store2.count() == 3


def test_upsert_overwrites_existing_chunk_id(tmp_path: Path, mock_embedding_adapter, sample_chunks):
    """Verify upserting with an identical chunk_id updates content without duplicates."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_overwrite",
        embedding_adapter=mock_embedding_adapter,
    )
    store.upsert_chunks(sample_chunks)
    assert store.count() == 3

    # Update chunk with modified content
    updated_chunk = Document(
        page_content="Alpha system architecture updated with new facts.",
        metadata=dict(sample_chunks[0].metadata),
    )
    store.upsert_chunks([updated_chunk])

    # Count must remain 3 (no duplication)
    assert store.count() == 3

    # Check that retrieved content is updated
    results = store.query_similar(query="Alpha query", top_k=1)
    assert len(results) == 1
    assert "updated with new facts" in results[0].content


def test_delete_document(tmp_path: Path, mock_embedding_adapter, sample_chunks):
    """Verify deleting by doc_id removes only the targeted document chunks."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_del",
        embedding_adapter=mock_embedding_adapter,
    )
    store.upsert_chunks(sample_chunks)
    assert store.count() == 3

    # Delete doc-alpha (which has 2 chunks)
    deleted = store.delete_document("doc-alpha")
    assert deleted == 2
    assert store.count() == 1

    # Remaining chunk should be doc-gamma
    results = store.query_similar("gamma query", top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "doc-gamma"

    # Deleting non-existent doc returns 0
    assert store.delete_document("non-existent-doc") == 0


def test_query_similar_cosine_interpretation(tmp_path: Path, mock_embedding_adapter, sample_chunks):
    """Verify query returns correct chunk with lowest cosine distance and highest similarity."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_query",
        embedding_adapter=mock_embedding_adapter,
    )
    store.upsert_chunks(sample_chunks)

    # Query for Alpha
    results = store.query_similar(query="Tell me about Alpha", top_k=2)
    assert len(results) == 2

    # First result must be doc-alpha_c0 (exact match on dim 0)
    top = results[0]
    assert top.chunk_id == "doc-alpha_c0"
    assert top.doc_id == "doc-alpha"
    assert top.page == 1
    assert top.filename == "guide.pdf"
    # Cosine distance should be very close to 0.0, score close to 1.0
    assert top.distance <= 0.05
    assert top.score >= 0.95


def test_query_similar_with_filter(tmp_path: Path, mock_embedding_adapter, sample_chunks):
    """Verify metadata filtering restricts results."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_filter",
        embedding_adapter=mock_embedding_adapter,
    )
    store.upsert_chunks(sample_chunks)

    # Query matching Alpha embedding, but filtered to notes.pdf (which is Gamma)
    results = store.query_similar(
        query="Alpha query",
        top_k=2,
        filter_criteria={"filename": "notes.pdf"},
    )
    assert len(results) == 1
    assert results[0].filename == "notes.pdf"
    assert results[0].doc_id == "doc-gamma"


def test_query_validation_and_empty_collection(tmp_path: Path, mock_embedding_adapter):
    """Verify query validation and empty collection handling."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_empty",
        embedding_adapter=mock_embedding_adapter,
    )
    # Empty collection query returns empty list
    assert store.query_similar("sample query", top_k=3) == []

    # Invalid queries
    with pytest.raises(ValueError, match="cannot be empty"):
        store.query_similar("")
    with pytest.raises(ValueError, match="must be greater than zero"):
        store.query_similar("query", top_k=0)


def test_clear_collection(tmp_path: Path, mock_embedding_adapter, sample_chunks):
    """Verify clear_collection resets count to zero."""
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_clear",
        embedding_adapter=mock_embedding_adapter,
    )
    store.upsert_chunks(sample_chunks)
    assert store.count() == 3

    store.clear_collection()
    assert store.count() == 0

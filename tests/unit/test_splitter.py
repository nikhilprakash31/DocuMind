"""Unit tests for DocumentSplitter and chunking evaluation."""

import pytest
from langchain_core.documents import Document

from src.ingestion.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocumentSplitter,
    evaluate_chunk_configurations,
)


def test_default_initialization():
    """Verify default chunk size and overlap values match specifications."""
    splitter = DocumentSplitter()
    assert splitter.chunk_size == DEFAULT_CHUNK_SIZE
    assert splitter.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    assert splitter.chunk_size == 800
    assert splitter.chunk_overlap == 150


def test_invalid_configurations():
    """Verify error handling for invalid chunk size and overlap parameters."""
    # Overlap equal to or greater than chunk size
    with pytest.raises(ValueError) as exc_info:
        DocumentSplitter(chunk_size=500, chunk_overlap=500)
    assert "must be less than" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        DocumentSplitter(chunk_size=500, chunk_overlap=600)
    assert "must be less than" in str(exc_info.value)

    # Negative overlap
    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=500, chunk_overlap=-10)

    # Zero or negative chunk size
    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=0, chunk_overlap=0)


def test_metadata_inheritance():
    """Verify chunks inherit all parent metadata fields and add chunk identifiers."""
    parent_meta = {
        "source": "contract.pdf",
        "filename": "contract.pdf",
        "page": 3,
        "total_pages": 10,
        "doc_id": "test-doc-uuid-1234",
        "doc_hash": "a1b2c3d4e5f6",
        "file_size_bytes": 45000,
        "upload_timestamp": "2026-11-08T12:00:00Z",
    }
    long_text = "This is a sentence about RAG architectures. " * 30
    parent_doc = Document(page_content=long_text, metadata=parent_meta)

    splitter = DocumentSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents([parent_doc])

    assert len(chunks) > 1

    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        # Inherited parent metadata
        assert meta["source"] == "contract.pdf"
        assert meta["filename"] == "contract.pdf"
        assert meta["page"] == 3
        assert meta["total_pages"] == 10
        assert meta["doc_id"] == "test-doc-uuid-1234"
        assert meta["doc_hash"] == "a1b2c3d4e5f6"
        assert meta["file_size_bytes"] == 45000
        # Injected chunk metadata
        assert meta["chunk_index"] == idx
        assert meta["chunk_id"] == f"test-doc-uuid-1234_c{idx}"
        assert meta["char_count"] == len(chunk.page_content)


def test_short_document_single_chunk():
    """Verify documents shorter than chunk_size result in a single chunk."""
    short_text = "A concise sentence that easily fits in one chunk."
    parent_doc = Document(
        page_content=short_text,
        metadata={"doc_id": "short-doc", "page": 1},
    )

    splitter = DocumentSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents([parent_doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"] == "short-doc_c0"


def test_chunk_overlap_behavior():
    """Verify overlap produces repeated boundary text across consecutive chunks."""
    sentences = [f"Sentence number {i} with distinct technical content." for i in range(25)]
    text = " ".join(sentences)
    parent_doc = Document(page_content=text, metadata={"doc_id": "overlap-doc"})

    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=60)
    chunks = splitter.split_documents([parent_doc])

    assert len(chunks) >= 2

    # Verify that the end of chunk 0 overlaps with the start of chunk 1
    chunk_0_words = chunks[0].page_content.split()
    chunk_1_words = chunks[1].page_content.split()

    # Find common tokens between end of chunk 0 and start of chunk 1
    tail_words = set(chunk_0_words[-5:])
    head_words = set(chunk_1_words[:8])
    overlap_words = tail_words.intersection(head_words)
    assert len(overlap_words) > 0


def test_empty_and_whitespace_documents():
    """Verify empty or whitespace-only documents are safely ignored."""
    splitter = DocumentSplitter()
    assert splitter.split_documents([]) == []

    docs = [
        Document(page_content="", metadata={"doc_id": "empty-1"}),
        Document(page_content="   \n\n  \t  ", metadata={"doc_id": "empty-2"}),
    ]
    chunks = splitter.split_documents(docs)
    assert len(chunks) == 0


def test_multipage_chunking_maintains_page_association():
    """Verify chunks maintain the exact page number of their origin page."""
    docs = [
        Document(
            page_content="Page 1 discussion on indexing algorithms. " * 15,
            metadata={"doc_id": "multi-page-doc", "page": 1},
        ),
        Document(
            page_content="Page 2 analysis of vector search complexity. " * 15,
            metadata={"doc_id": "multi-page-doc", "page": 2},
        ),
    ]

    splitter = DocumentSplitter(chunk_size=250, chunk_overlap=40)
    chunks = splitter.split_documents(docs)

    page_1_chunks = [c for c in chunks if c.metadata["page"] == 1]
    page_2_chunks = [c for c in chunks if c.metadata["page"] == 2]

    assert len(page_1_chunks) > 1
    assert len(page_2_chunks) > 1

    for c in page_1_chunks:
        assert "Page 1" in c.page_content
    for c in page_2_chunks:
        assert "Page 2" in c.page_content

    # Global chunk indices must be continuous
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_split_text_method():
    """Verify split_text convenience helper."""
    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    text = "Short text for direct splitting test."
    chunks = splitter.split_text(text, base_metadata={"source": "direct_input.txt"})

    assert len(chunks) == 1
    assert chunks[0].page_content == text
    assert chunks[0].metadata["source"] == "direct_input.txt"


def test_evaluate_chunk_configurations():
    """Verify evaluation function calculates comparative metrics across configurations."""
    paragraph = (
        "Retrieval-Augmented Generation (RAG) is an architecture that optimizes the output "
        "of a large language model by referencing an authoritative knowledge base outside of "
        "its training data sources before generating a response. Modern RAG systems utilize "
        "dense vector embeddings to map text into geometric space where semantic similarity "
        "corresponds to vector distance. "
    ) * 10

    docs = [
        Document(page_content=paragraph, metadata={"doc_id": "doc-eval", "page": 1}),
        Document(page_content=paragraph, metadata={"doc_id": "doc-eval", "page": 2}),
    ]

    results = evaluate_chunk_configurations(
        documents=docs,
        configurations=[
            (800, 150),
            (1000, 200),
            (1200, 200),
        ],
    )

    assert len(results) == 3

    config_names = [r.config_name for r in results]
    assert config_names == ["800/150", "1000/200", "1200/200"]

    # Larger chunk sizes should produce fewer chunks
    chunks_800 = results[0].total_chunks
    chunks_1000 = results[1].total_chunks
    chunks_1200 = results[2].total_chunks

    assert chunks_800 >= chunks_1000 >= chunks_1200
    assert all(r.avg_chunk_length > 0 for r in results)
    assert all(r.max_chunk_length <= r.chunk_size for r in results)

"""Unit tests for LangChain-based PDFDocumentLoader."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.core.exceptions import DuplicateDocumentError
from src.ingestion.loader import PDFDocumentLoader
from src.utils.hashing import DuplicateDetector


def test_valid_pdf_loading(sample_valid_pdf: Path):
    """Requirement 1: Valid PDF loading extracts text and returns LangChain Documents."""
    loader = PDFDocumentLoader()
    docs = loader.load(sample_valid_pdf)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Retrieval-Augmented Generation" in docs[0].page_content


def test_multipage_pdf_loading(sample_multipage_pdf: Path):
    """Requirement 2: Multi-page PDF loading extracts text page by page with 1-based indexing."""
    loader = PDFDocumentLoader()
    docs = loader.load(sample_multipage_pdf)

    assert len(docs) == 3

    # Verify content of each page
    assert "Chapter 1" in docs[0].page_content
    assert "Chapter 2" in docs[1].page_content
    assert "Chapter 3" in docs[2].page_content

    # Verify 1-based page numbering normalization
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert docs[2].metadata["page"] == 3


def test_metadata_preservation(sample_multipage_pdf: Path):
    """Requirement 3: Verify all metadata fields are preserved consistently across all pages."""
    loader = PDFDocumentLoader()
    docs = loader.load(sample_multipage_pdf)

    first_doc_id = docs[0].metadata["doc_id"]
    first_doc_hash = docs[0].metadata["doc_hash"]

    for idx, doc in enumerate(docs):
        meta = doc.metadata
        # Preserved fields check
        assert meta["source"] == "multipage_handbook.pdf"
        assert meta["filename"] == "multipage_handbook.pdf"
        assert meta["page"] == idx + 1
        assert meta["total_pages"] == 3
        assert meta["file_size_bytes"] > 0
        assert "upload_timestamp" in meta
        # Consistency across pages
        assert meta["doc_id"] == first_doc_id
        assert meta["doc_hash"] == first_doc_hash


def test_unicode_text_handling(pdf_factory):
    """Verify proper handling and normalization of Unicode text."""
    unicode_content = "RAG with café, naïve, and résumé symbols: über, façade."
    pdf_path = pdf_factory("unicode_test.pdf", [unicode_content])

    loader = PDFDocumentLoader()
    docs = loader.load(pdf_path)

    assert len(docs) == 1
    assert "café" in docs[0].page_content
    assert "résumé" in docs[0].page_content
    assert "über" in docs[0].page_content
    assert "façade" in docs[0].page_content


def test_multiple_pdf_loading(sample_valid_pdf: Path, sample_multipage_pdf: Path):
    """Requirement: Handle multiple PDFs batch ingestion."""
    loader = PDFDocumentLoader()
    batch_results = loader.load_multiple([sample_valid_pdf, sample_multipage_pdf])

    assert len(batch_results) == 2
    assert sample_valid_pdf.name in batch_results
    assert sample_multipage_pdf.name in batch_results

    assert len(batch_results[sample_valid_pdf.name]) == 1
    assert len(batch_results[sample_multipage_pdf.name]) == 3


def test_loader_duplicate_detection(sample_valid_pdf: Path):
    """Requirement: Ingestion rejects duplicate files on subsequent loads."""
    detector = DuplicateDetector()
    loader = PDFDocumentLoader(duplicate_detector=detector)

    # First load must succeed
    docs1 = loader.load(sample_valid_pdf)
    assert len(docs1) == 1

    # Second load of identical file must raise DuplicateDocumentError
    with pytest.raises(DuplicateDocumentError) as exc_info:
        loader.load(sample_valid_pdf, raise_on_duplicate=True)
    assert exc_info.value.filename == sample_valid_pdf.name

    # Second load with raise_on_duplicate=False should safely return empty list
    docs_skipped = loader.load(sample_valid_pdf, raise_on_duplicate=False)
    assert docs_skipped == []


def test_loader_summary_generation(sample_multipage_pdf: Path):
    """Verify IngestionSummary calculation from loaded documents."""
    loader = PDFDocumentLoader()
    docs = loader.load(sample_multipage_pdf)

    summary = loader.get_summary(docs)
    assert summary is not None
    assert summary.filename == "multipage_handbook.pdf"
    assert summary.total_pages == 3
    assert summary.total_characters > 50
    assert summary.file_size_bytes > 0

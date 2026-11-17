"""Unit tests for PDFPageLocator and EvidenceMapper."""

import pytest

from src.citations.locator import PDFPageLocator
from src.citations.mapper import EvidenceMapper
from src.core.models import RetrievalResult


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievalResult]:
    """Sample chunks for testing evidence mapping and duplicate filename resolution."""
    return [
        RetrievalResult(
            chunk_id="doc_alpha_c0",
            doc_id="doc_alpha",
            filename="specification.pdf",
            page=1,
            chunk_index=0,
            content="HNSW graphs construct multi-layer proximity graphs for fast vector search.",
            distance=0.12,
            score=0.88,
            metadata={"total_pages": 4},
        ),
        RetrievalResult(
            chunk_id="doc_alpha_c1",
            doc_id="doc_alpha",
            filename="specification.pdf",
            page=2,
            chunk_index=1,
            content="Recursive character splitting uses an 800 character window with 150 overlap.",
            distance=0.18,
            score=0.82,
            metadata={"total_pages": 4},
        ),
        RetrievalResult(
            chunk_id="doc_beta_c0",
            doc_id="doc_beta",  # Different document with identical filename!
            filename="specification.pdf",
            page=1,
            chunk_index=0,
            content="Second specification document discussing database engine configuration.",
            distance=0.25,
            score=0.75,
            metadata={"total_pages": 2},
        ),
        RetrievalResult(
            chunk_id="doc_gamma_c0",
            doc_id="doc_gamma",
            filename="financial_results.pdf",
            page=3,
            chunk_index=0,
            content="Operating margin expanded to 28.4% and cash reserves reached $8.9B.",
            distance=0.15,
            score=0.85,
            metadata={"total_pages": 5},
        ),
    ]


def test_locator_page_bounds():
    """Verify page number validation against total document pages."""
    assert PDFPageLocator.validate_page_bounds(1, 5) is True
    assert PDFPageLocator.validate_page_bounds(5, 5) is True
    assert PDFPageLocator.validate_page_bounds(0, 5) is False
    assert PDFPageLocator.validate_page_bounds(6, 5) is False


def test_locator_url_fragment():
    """Verify viewer anchor fragment generation."""
    assert PDFPageLocator.build_page_url_fragment(3) == "#page=3"
    assert PDFPageLocator.build_page_url_fragment(0) == "#page=1"


def test_locator_extract_snippet():
    """Verify excerpt extraction prioritizes search terms."""
    text = (
        "In traditional computing architectures, memory bandwidth is constrained. "
        "DocuMind utilizes recursive character splitting to maintain context. "
        "Overall throughput is high."
    )
    snippet = PDFPageLocator.extract_preview_snippet(
        text, query="recursive splitting", max_chars=80
    )
    assert "recursive character splitting" in snippet


def test_map_evidence_basic(sample_retrieved_chunks):
    """Verify evidence mapping produces structured cards and annotated answer."""
    mapper = EvidenceMapper()
    answer = (
        "DocuMind uses HNSW graphs [Page 1, specification.pdf] and chunks text with "
        "an 800 character window [Page 2, specification.pdf]."
    )

    ev_map = mapper.map_evidence(answer, sample_retrieved_chunks)

    assert ev_map.total_citations == 2
    assert ev_map.verified_count == 2
    assert len(ev_map.evidence_cards) == 2
    assert len(ev_map.unverified_references) == 0

    # Verify Card 1
    card1 = ev_map.evidence_cards[0]
    assert card1.citation_index == 1
    assert card1.doc_id == "doc_alpha"
    assert card1.filename == "specification.pdf"
    assert card1.page == 1
    assert card1.total_pages == 4
    assert card1.page_url_fragment == "#page=1"
    assert card1.is_verified is True

    # Verify Card 2
    card2 = ev_map.evidence_cards[1]
    assert card2.citation_index == 2
    assert card2.page == 2
    assert card2.chunk_id == "doc_alpha_c1"

    # Verify annotated text has normalized [1], [2] markers
    assert "[1]" in ev_map.annotated_answer
    assert "[2]" in ev_map.annotated_answer
    assert "[Page 1, specification.pdf]" not in ev_map.annotated_answer

    # Verify cited pages dictionary
    assert ev_map.cited_pages_by_document["specification.pdf"] == [1, 2]


def test_map_evidence_duplicate_filename_disambiguation(sample_retrieved_chunks):
    """Verify doc_id matching properly disambiguates identical filenames."""
    mapper = EvidenceMapper()
    # Explicitly reference doc_beta via doc_id pattern
    answer = "Database engine is configured [doc_beta, Page 1]."

    ev_map = mapper.map_evidence(answer, sample_retrieved_chunks)
    assert ev_map.verified_count == 1
    card = ev_map.evidence_cards[0]
    assert card.doc_id == "doc_beta"
    assert card.chunk_id == "doc_beta_c0"
    assert card.filename == "specification.pdf"


def test_map_evidence_cross_document_multi_citations(sample_retrieved_chunks):
    """Verify mapping across different documents and pages in a single answer."""
    mapper = EvidenceMapper()
    answer = (
        "Operating margin is 28.4% [Page 3, financial_results.pdf], while vector search "
        "uses HNSW graph [Page 1, specification.pdf]."
    )

    ev_map = mapper.map_evidence(answer, sample_retrieved_chunks)
    assert ev_map.verified_count == 2
    assert "financial_results.pdf" in ev_map.cited_pages_by_document
    assert "specification.pdf" in ev_map.cited_pages_by_document
    assert ev_map.cited_pages_by_document["financial_results.pdf"] == [3]
    assert ev_map.cited_pages_by_document["specification.pdf"] == [1]


def test_map_evidence_unverified_and_out_of_bounds(sample_retrieved_chunks):
    """Verify missing, invalid, or hallucinated citations are flagged."""
    mapper = EvidenceMapper()
    answer = (
        "Valid claim [Page 1, specification.pdf], but also fake claim [Page 99, fake_file.pdf] "
        "and unknown page [Page 44]."
    )

    ev_map = mapper.map_evidence(answer, sample_retrieved_chunks)
    assert ev_map.verified_count == 1
    assert len(ev_map.unverified_references) == 2
    assert "[Page 99, fake_file.pdf]" in ev_map.unverified_references
    assert "[Page 44]" in ev_map.unverified_references
    assert ev_map.total_citations == 3


def test_map_evidence_empty():
    """Verify empty input handling."""
    mapper = EvidenceMapper()
    ev_map = mapper.map_evidence("", [])
    assert ev_map.total_citations == 0
    assert len(ev_map.evidence_cards) == 0

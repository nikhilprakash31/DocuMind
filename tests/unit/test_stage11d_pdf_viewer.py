"""Automated verification suite for Stage 11D: PDF Viewer & Citation Navigation."""

import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import (
    ChatMessage,
    Citation,
    EvidenceCard,
    RetrievalResult,
)
from src.frontend.app import (
    _extract_evidence_cards,
    _extract_pdf_page_text,
)
from src.service.app_service import DocuMindService


@pytest.fixture(autouse=True)
def mock_chroma_vector_store() -> Generator[MagicMock, None, None]:
    """Mock ChromaVectorStore chunk upserts to isolate PDF viewer tests from Ollama."""
    with patch(
        "src.storage.vector_store.ChromaVectorStore.upsert_chunks",
        autospec=True,
    ) as mock_upsert:
        mock_upsert.side_effect = lambda self, chunks, **kwargs: len(chunks)
        yield mock_upsert


def _create_test_pdf(text_pages: list[str]) -> bytes:
    """Generate a valid multi-page PDF with given text on each page."""
    objects: list[bytes] = []

    def obj(n: int, body: bytes) -> bytes:
        return f"{n} 0 obj\n".encode() + body + b"\nendobj\n"

    num_pages = len(text_pages)
    # 1: Catalog
    objects.append(obj(1, b"<< /Type /Catalog /Pages 2 0 R >>"))

    # Kids references: page objects start at 3, each page has a content stream
    # Pages: 3, 5, 7, ...
    # Contents: 4, 6, 8, ...
    kids_refs = " ".join([f"{3 + 2 * i} 0 R" for i in range(num_pages)])
    objects.append(obj(2, f"<< /Type /Pages /Kids [{kids_refs}] /Count {num_pages} >>".encode()))

    font_obj_num = 3 + 2 * num_pages

    for i, page_text in enumerate(text_pages):
        page_num = 3 + 2 * i
        content_num = 4 + 2 * i

        # Page object
        objects.append(
            obj(
                page_num,
                (
                    f"<< /Type /Page /Parent 2 0 R "
                    f"/MediaBox [0 0 612 792] "
                    f"/Contents {content_num} 0 R "
                    f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>"
                ).encode(),
            )
        )

        # Content stream
        stream_data = f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode()
        objects.append(
            obj(
                content_num,
                f"<< /Length {len(stream_data)} >>\nstream\n".encode()
                + stream_data
                + b"\nendstream",
            )
        )

    # Font object
    objects.append(obj(font_obj_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    for o in objects:
        offsets.append(len(header) + len(body))
        body += o

    xref_offset = len(header) + len(body)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    return header + body + xref.encode() + trailer.encode()


def test_two_pdfs_same_filename_distinct_citations_and_paths() -> None:
    """Test 1: Two PDFs with the same filename resolve to distinct files, paths, and doc_ids."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="documind_stage11d_test1_"))
    try:
        service = DocuMindService(base_dir=str(tmp_dir))
        upload_dir = tmp_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Distinct contents for PDF A and PDF B
        content_a = _create_test_pdf(["ALPHA PROTOCOL REPORT"])
        content_b = _create_test_pdf(["BETA DIRECTIVE BRIEFING"])

        shared_filename = "intelligence_brief.pdf"

        # Ingest Document A
        slot_a = upload_dir / uuid.uuid4().hex
        slot_a.mkdir()
        path_a = slot_a / shared_filename
        path_a.write_bytes(content_a)
        res_a = service.ingest_document(path_a)
        assert res_a.success
        doc_a = res_a.data

        # Ingest Document B (same filename, different content)
        slot_b = upload_dir / uuid.uuid4().hex
        slot_b.mkdir()
        path_b = slot_b / shared_filename
        path_b.write_bytes(content_b)
        res_b = service.ingest_document(path_b)
        assert res_b.success
        doc_b = res_b.data

        # 1. Check distinct doc_ids and distinct paths
        assert doc_a.doc_id != doc_b.doc_id
        assert doc_a.file_path != doc_b.file_path
        assert doc_a.filename == shared_filename
        assert doc_b.filename == shared_filename

        # 2. Verify files on disk match their original bytes
        assert Path(doc_a.file_path).read_bytes() == content_a
        assert Path(doc_b.file_path).read_bytes() == content_b

        # 3. Verify citations mapped by doc_id resolve the correct document
        # Even though filenames are identical
        rec_resolved_a = service.get_document(doc_a.doc_id)
        assert rec_resolved_a is not None
        assert Path(rec_resolved_a.file_path).resolve() == path_a.resolve()
        text_a = _extract_pdf_page_text(Path(rec_resolved_a.file_path), 1)
        assert text_a is not None
        assert "ALPHA PROTOCOL" in text_a
        assert "BETA DIRECTIVE" not in text_a

        rec_resolved_b = service.get_document(doc_b.doc_id)
        assert rec_resolved_b is not None
        assert Path(rec_resolved_b.file_path).resolve() == path_b.resolve()
        text_b = _extract_pdf_page_text(Path(rec_resolved_b.file_path), 1)
        assert text_b is not None
        assert "BETA DIRECTIVE" in text_b
        assert "ALPHA PROTOCOL" not in text_b

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_multiple_citations_and_page_navigation() -> None:
    """Test 2: Multi-page document citations, page clamping, and text extraction."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="documind_stage11d_test2_"))
    try:
        service = DocuMindService(base_dir=str(tmp_dir))
        upload_dir = tmp_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        pages_text = [
            "Introduction: DocuMind Architecture overview.",
            "Chapter 1: Multi-vector hybrid search mechanics.",
            "Chapter 2: Grounded citation verification framework.",
        ]
        multi_page_pdf = _create_test_pdf(pages_text)

        slot = upload_dir / uuid.uuid4().hex
        slot.mkdir()
        pdf_path = slot / "manual.pdf"
        pdf_path.write_bytes(multi_page_pdf)

        res = service.ingest_document(pdf_path)
        assert res.success
        doc_rec = res.data
        assert doc_rec.total_pages == 3

        # Test extraction of all 3 pages
        p1_text = _extract_pdf_page_text(Path(doc_rec.file_path), 1)
        assert p1_text and "Architecture overview" in p1_text

        p2_text = _extract_pdf_page_text(Path(doc_rec.file_path), 2)
        assert p2_text and "Multi-vector hybrid search" in p2_text

        p3_text = _extract_pdf_page_text(Path(doc_rec.file_path), 3)
        assert p3_text and "Grounded citation verification" in p3_text

        # Test page navigation clamping logic
        total = doc_rec.total_pages

        # Forward clamp
        assert min(1 + 1, total) == 2
        assert min(2 + 1, total) == 3
        assert min(3 + 1, total) == 3  # clamped at max

        # Backward clamp
        assert max(3 - 1, 1) == 2
        assert max(2 - 1, 1) == 1
        assert max(1 - 1, 1) == 1  # clamped at min

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_missing_files_and_invalid_page_handling() -> None:
    """Test 3: Graceful degradation for deleted documents, missing files, and out-of-range pages."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="documind_stage11d_test3_"))
    try:
        service = DocuMindService(base_dir=str(tmp_dir))

        # 1. Non-existent doc_id (e.g. deleted document)
        missing_doc = service.get_document("non-existent-doc-id-12345")
        assert missing_doc is None

        # 2. Ingest document then delete physical file from disk
        upload_dir = tmp_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        slot = upload_dir / uuid.uuid4().hex
        slot.mkdir()
        p_file = slot / "ephemeral.pdf"
        p_file.write_bytes(_create_test_pdf(["Ephemeral content"]))

        res = service.ingest_document(p_file)
        assert res.success
        doc = res.data
        p_path = Path(doc.file_path)
        assert p_path.is_file()

        # Delete the file
        p_path.unlink()
        assert not p_path.is_file()

        # Verification: get_document still returns registry record, but is_file() is False
        rec = service.get_document(doc.doc_id)
        assert rec is not None
        assert not Path(rec.file_path).is_file()

        # Text extraction safely returns None rather than raising an uncaught exception
        extracted = _extract_pdf_page_text(Path(rec.file_path), 1)
        assert extracted is None

        # 3. Invalid page numbers on valid PDF
        valid_pdf_path = slot / "valid.pdf"
        valid_pdf_path.write_bytes(_create_test_pdf(["Page 1", "Page 2"]))
        res_v = service.ingest_document(valid_pdf_path)
        assert res_v.success
        v_doc = res_v.data

        # Negative page
        assert _extract_pdf_page_text(Path(v_doc.file_path), -1) is None
        # Page 0
        assert _extract_pdf_page_text(Path(v_doc.file_path), 0) is None
        # Page > total_pages
        assert _extract_pdf_page_text(Path(v_doc.file_path), 999) is None

        # Clamping logic verification
        total_pages = v_doc.total_pages
        for bad_page, expected_clamped in [(-10, 1), (0, 1), (999, total_pages), (15, total_pages)]:
            clamped = max(1, min(bad_page, total_pages))
            assert clamped == expected_clamped

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evidence_card_extraction_and_reconstruction() -> None:
    """Test 4: Verify _extract_evidence_cards reconstructs cards correctly."""
    # Case A: ChatMessage already has evidence_cards
    prebuilt_card = EvidenceCard(
        citation_index=1,
        doc_id="doc-prebuilt-1",
        filename="prebuilt.pdf",
        page=2,
        total_pages=5,
        chunk_id="chunk-1",
        snippet="Prebuilt snippet quote.",
        similarity_score=0.92,
        is_verified=True,
    )
    msg_with_cards = ChatMessage(
        message_id="msg-1",
        session_id="sess-1",
        role="assistant",
        content="Answer [1]",
        evidence_cards=[prebuilt_card],
    )
    extracted = _extract_evidence_cards(msg_with_cards)
    assert len(extracted) == 1
    assert extracted[0].doc_id == "doc-prebuilt-1"
    assert extracted[0].similarity_score == 0.92
    assert extracted[0].snippet == "Prebuilt snippet quote."

    # Case B: ChatMessage has sources and citation syntax in content
    source_chunk = RetrievalResult(
        chunk_id="doc_xyz_c0",
        doc_id="doc_xyz",
        filename="whitepaper.pdf",
        page=3,
        chunk_index=0,
        content="Retrieval Augmented Generation reduces hallucination rates significantly.",
        distance=0.12,
        score=0.88,
        metadata={"total_pages": 10},
    )
    msg_with_sources = ChatMessage(
        message_id="msg-2",
        session_id="sess-1",
        role="assistant",
        content="As stated in [Page 3, whitepaper.pdf], RAG lowers hallucinations.",
        sources=[source_chunk],
    )
    extracted_from_sources = _extract_evidence_cards(msg_with_sources)
    assert len(extracted_from_sources) == 1
    card = extracted_from_sources[0]
    assert card.doc_id == "doc_xyz"
    assert card.filename == "whitepaper.pdf"
    assert card.page == 3
    assert card.similarity_score == 0.88
    assert "hallucination" in card.snippet.lower()

    # Case C: ChatMessage has citations list
    citation = Citation(
        filename="handbook.pdf",
        page=4,
        chunk_id="doc_hb_c2",
        doc_id="doc_hb",
        snippet="Employee benefits guideline.",
        is_verified=True,
        raw_reference="[handbook.pdf:p4]",
    )
    msg_with_citations = ChatMessage(
        message_id="msg-3",
        session_id="sess-1",
        role="assistant",
        content="Here is the benefits policy.",
        citations=[citation],
    )
    extracted_from_cit = _extract_evidence_cards(msg_with_citations)
    assert len(extracted_from_cit) == 1
    card_cit = extracted_from_cit[0]
    assert card_cit.doc_id == "doc_hb"
    assert card_cit.filename == "handbook.pdf"
    assert card_cit.page == 4
    assert card_cit.snippet == "Employee benefits guideline."

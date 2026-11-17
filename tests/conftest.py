"""Pytest fixtures for PDF ingestion testing.

Provides programmatic PDF generation without relying on external or personal files.
"""

import io
from pathlib import Path

import pypdf
import pytest


def build_pdf_bytes(pages: list[str]) -> bytes:
    """Generate a standards-compliant PDF byte string containing the given page texts."""
    num_pages = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(num_pages))

    objs: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }

    for i, text in enumerate(pages):
        page_obj_id = 4 + 2 * i
        content_obj_id = 5 + 2 * i
        # Ensure stream is latin-1 safe for standard Type1 Helvetica font
        encoded_text = text.encode("latin-1", errors="replace").decode("latin-1")
        safe_text = encoded_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content = f"BT /F1 12 Tf 50 700 Td ({safe_text}) Tj ET"
        stream_bytes = stream_content.encode("latin-1")
        stream_len = len(stream_bytes)
        objs[page_obj_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj_id} 0 R >>"
        )
        objs[content_obj_id] = f"<< /Length {stream_len} >>\nstream\n{stream_content}\nendstream"

    out = b"%PDF-1.4\n"
    xref_offsets: dict[int, int] = {}
    total_len = len(out)
    for obj_id in sorted(objs.keys()):
        xref_offsets[obj_id] = total_len
        body = f"{obj_id} 0 obj\n{objs[obj_id]}\nendobj\n".encode("latin1")
        out += body
        total_len += len(body)

    startxref = total_len
    xref = f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    for obj_id in sorted(objs.keys()):
        xref += f"{xref_offsets[obj_id]:010d} 00000 n \n"
    out += xref.encode("latin1")
    trailer_bytes = (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{startxref}\n%%EOF\n"
    ).encode("latin1")
    out += trailer_bytes
    return out


@pytest.fixture
def pdf_factory(tmp_path: Path):
    """Factory fixture to create temporary PDF files with custom pages and filename."""

    def _create(filename: str, pages_text: list[str]) -> Path:
        content = build_pdf_bytes(pages_text)
        file_path = tmp_path / filename
        file_path.write_bytes(content)
        return file_path

    return _create


@pytest.fixture
def sample_valid_pdf(tmp_path: Path) -> Path:
    """Fixture providing a valid 1-page PDF file."""
    content = build_pdf_bytes(["Introduction to Retrieval-Augmented Generation."])
    file_path = tmp_path / "sample_doc.pdf"
    file_path.write_bytes(content)
    return file_path


@pytest.fixture
def sample_multipage_pdf(tmp_path: Path) -> Path:
    """Fixture providing a valid 3-page PDF file."""
    pages = [
        "Chapter 1: The Foundations of RAG Systems.",
        "Chapter 2: Vector Embeddings and Dense Retrieval.",
        "Chapter 3: Generation Guardrails and Evaluation.",
    ]
    content = build_pdf_bytes(pages)
    file_path = tmp_path / "multipage_handbook.pdf"
    file_path.write_bytes(content)
    return file_path


@pytest.fixture
def sample_blank_pdf(tmp_path: Path) -> Path:
    """Fixture providing a PDF with blank pages (no extractable text)."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    file_path = tmp_path / "scanned_invoice.pdf"
    file_path.write_bytes(buf.getvalue())
    return file_path


@pytest.fixture
def sample_encrypted_pdf(tmp_path: Path) -> Path:
    """Fixture providing an encrypted/password-protected PDF."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("topsecret123")
    buf = io.BytesIO()
    writer.write(buf)
    file_path = tmp_path / "confidential_locked.pdf"
    file_path.write_bytes(buf.getvalue())
    return file_path


@pytest.fixture
def sample_empty_pdf(tmp_path: Path) -> Path:
    """Fixture providing a 0-byte file with .pdf extension."""
    file_path = tmp_path / "empty_file.pdf"
    file_path.write_bytes(b"")
    return file_path


@pytest.fixture
def sample_corrupted_pdf(tmp_path: Path) -> Path:
    """Fixture providing a corrupted PDF file."""
    file_path = tmp_path / "corrupted_payload.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%%TRUNCATED_GARBAGE_NO_TRAILER")
    return file_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Fixture providing a non-PDF file."""
    file_path = tmp_path / "document.txt"
    file_path.write_text("Plain text content")
    return file_path

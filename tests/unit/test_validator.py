"""Unit tests for PDF document validation."""

from pathlib import Path

import pytest

from src.core.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    FileSizeLimitExceededError,
    InvalidFileExtensionError,
    InvalidPDFSignatureError,
    NoExtractableTextError,
    PasswordProtectedPDFError,
)
from src.ingestion.validator import PDFValidator


def test_valid_pdf_passes_validation(sample_valid_pdf: Path):
    """Verify that a well-formed PDF passes validation cleanly."""
    validator = PDFValidator()
    result = validator.validate_or_raise(sample_valid_pdf)
    assert result.is_valid is True
    assert result.page_count == 1
    assert result.has_text is True
    assert result.is_encrypted is False
    assert result.requires_ocr is False


def test_invalid_file_extension(sample_text_file: Path):
    """Verify that non-.pdf files are rejected with InvalidFileExtensionError."""
    validator = PDFValidator()
    with pytest.raises(InvalidFileExtensionError) as exc_info:
        validator.validate_or_raise(sample_text_file)
    assert exc_info.value.expected == ".pdf"


def test_empty_pdf(sample_empty_pdf: Path):
    """Verify that 0-byte PDF files are rejected with EmptyFileError."""
    validator = PDFValidator()
    with pytest.raises(EmptyFileError) as exc_info:
        validator.validate_or_raise(sample_empty_pdf)
    assert exc_info.value.filename == "empty_file.pdf"


def test_file_size_limit_exceeded(sample_valid_pdf: Path):
    """Verify that files exceeding the configured size limit are rejected."""
    # Set limit below the sample PDF size
    validator = PDFValidator(max_file_size_bytes=10)
    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        validator.validate_or_raise(sample_valid_pdf)
    assert exc_info.value.max_bytes == 10


def test_invalid_pdf_signature(tmp_path: Path):
    """Verify that a file with .pdf extension but missing %PDF- header is rejected."""
    fake_pdf = tmp_path / "fake_header.pdf"
    fake_pdf.write_bytes(b"GIF89a Fake Image Payload")
    validator = PDFValidator()
    with pytest.raises(InvalidPDFSignatureError):
        validator.validate_or_raise(fake_pdf)


def test_corrupted_pdf(sample_corrupted_pdf: Path):
    """Verify that a truncated or structurally malformed PDF raises CorruptedPDFError."""
    validator = PDFValidator()
    with pytest.raises(CorruptedPDFError):
        validator.validate_or_raise(sample_corrupted_pdf)


def test_password_protected_pdf(sample_encrypted_pdf: Path):
    """Verify that password-protected PDFs raise PasswordProtectedPDFError."""
    validator = PDFValidator()
    with pytest.raises(PasswordProtectedPDFError):
        validator.validate_or_raise(sample_encrypted_pdf)


def test_pdf_without_extractable_text(sample_blank_pdf: Path):
    """Verify that image-only or scanned PDFs raise NoExtractableTextError advising OCR."""
    validator = PDFValidator()
    with pytest.raises(NoExtractableTextError) as exc_info:
        validator.validate_or_raise(sample_blank_pdf)
    assert "OCR is required" in str(exc_info.value)


def test_validate_non_raising_method(sample_blank_pdf: Path, sample_valid_pdf: Path):
    """Verify that validate() returns a ValidationResult object without raising exceptions."""
    validator = PDFValidator()

    # Valid PDF
    res_valid = validator.validate(sample_valid_pdf)
    assert res_valid.is_valid is True
    assert res_valid.error_message is None

    # Blank PDF (needs OCR)
    res_blank = validator.validate(sample_blank_pdf)
    assert res_blank.is_valid is False
    assert res_blank.requires_ocr is True
    assert "OCR is required" in (res_blank.error_message or "")

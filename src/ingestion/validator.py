"""PDF validation module for structural integrity, encryption, and readability checks."""

from pathlib import Path

import pypdf
from pypdf.errors import EmptyFileError as PyPDFEmptyFileError
from pypdf.errors import FileNotDecryptedError, PdfReadError

from src.core.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    FileSizeLimitExceededError,
    InvalidFileExtensionError,
    InvalidPDFSignatureError,
    NoExtractableTextError,
    PasswordProtectedPDFError,
)
from src.core.models import ValidationResult

DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
PDF_MAGIC_BYTES = b"%PDF-"


class PDFValidator:
    """Validates PDF files prior to parsing and ingestion."""

    def __init__(self, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def validate_or_raise(self, file_path: str | Path) -> ValidationResult:
        """Validate a PDF file and raise a domain exception if any check fails."""
        path = Path(file_path)

        # 1. File existence
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found at '{path}'")

        filename = path.name

        # 2. File extension check (case-insensitive)
        if path.suffix.lower() != ".pdf":
            raise InvalidFileExtensionError(filename=filename, expected=".pdf")

        # 3. File size checks (0 bytes and max size limit)
        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyFileError(filename=filename)

        if file_size > self.max_file_size_bytes:
            raise FileSizeLimitExceededError(
                filename=filename,
                size_bytes=file_size,
                max_bytes=self.max_file_size_bytes,
            )

        # 4. PDF Magic Bytes Signature check (%PDF- within first 1024 bytes)
        with open(path, "rb") as f:
            header = f.read(1024)
            if PDF_MAGIC_BYTES not in header:
                raise InvalidPDFSignatureError(filename=filename)

        # 5. PDF Structure, Encryption, and Text Readability checks via pypdf
        try:
            reader = pypdf.PdfReader(str(path))

            # 5a. Password protection check
            if reader.is_encrypted:
                raise PasswordProtectedPDFError(filename=filename)

            # 5b. Page count and readability
            page_count = len(reader.pages)
            if page_count == 0:
                raise CorruptedPDFError(
                    filename=filename, details="Document contains 0 pages."
                )

            # 5c. Extractable text check across all pages
            extracted_chars = 0
            for page_idx, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    extracted_chars += len(text.strip())
                except Exception as e:
                    raise CorruptedPDFError(
                        filename=filename,
                        details=f"Failed to read page {page_idx + 1}: {e}",
                    ) from e

            # If document has pages but zero extractable text (e.g. scanned image PDF)
            if extracted_chars == 0:
                raise NoExtractableTextError(filename=filename)

            return ValidationResult(
                is_valid=True,
                filename=filename,
                file_size_bytes=file_size,
                page_count=page_count,
                has_text=True,
                is_encrypted=False,
                requires_ocr=False,
            )

        except (PasswordProtectedPDFError, NoExtractableTextError):
            raise
        except (PyPDFEmptyFileError, EmptyFileError):
            raise EmptyFileError(filename=filename)
        except FileNotDecryptedError:
            raise PasswordProtectedPDFError(filename=filename)
        except PdfReadError as e:
            raise CorruptedPDFError(filename=filename, details=str(e)) from e
        except Exception as e:
            if isinstance(
                e, (CorruptedPDFError, PasswordProtectedPDFError, NoExtractableTextError)
            ):
                raise
            raise CorruptedPDFError(filename=filename, details=str(e)) from e

    def validate(self, file_path: str | Path) -> ValidationResult:
        """Validate a PDF file and return a structured ValidationResult without raising."""
        path = Path(file_path)
        filename = path.name if path.exists() else str(file_path)
        size = path.stat().st_size if path.is_file() else 0

        try:
            return self.validate_or_raise(file_path)
        except NoExtractableTextError as e:
            return ValidationResult(
                is_valid=False,
                filename=filename,
                file_size_bytes=size,
                has_text=False,
                error_message=str(e),
                requires_ocr=True,
            )
        except PasswordProtectedPDFError as e:
            return ValidationResult(
                is_valid=False,
                filename=filename,
                file_size_bytes=size,
                is_encrypted=True,
                error_message=str(e),
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                filename=filename,
                file_size_bytes=size,
                error_message=str(e),
            )

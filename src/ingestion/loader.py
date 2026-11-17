"""PDF document loader with metadata normalization and deduplication."""

import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.core.exceptions import DuplicateDocumentError
from src.core.models import DocumentMetadata, IngestionSummary
from src.ingestion.validator import PDFValidator
from src.utils.hashing import DuplicateDetector, calculate_sha256, generate_doc_id


class PDFDocumentLoader:
    """Loads PDF documents, normalizes metadata to 1-based pages, and detects duplicates."""

    def __init__(
        self,
        validator: PDFValidator | None = None,
        duplicate_detector: DuplicateDetector | None = None,
    ) -> None:
        self.validator = validator or PDFValidator()
        self.duplicate_detector = duplicate_detector or DuplicateDetector()

    def load(
        self,
        file_path: str | Path,
        raise_on_duplicate: bool = True,
    ) -> list[Document]:
        """Load a single PDF file, validate it, extract pages, and normalize metadata.

        Args:
            file_path: Path to the target PDF file.
            raise_on_duplicate: If True, raises DuplicateDocumentError when file_hash is seen.
                                If False, returns an empty list for duplicates.

        Returns:
            List of LangChain Document objects with standardized 1-based page metadata.

        Raises:
            DocumentValidationError: If the PDF fails validation.
            DuplicateDocumentError: If the document is a duplicate and raise_on_duplicate is True.
        """
        path = Path(file_path).resolve()

        # Step 1: Structural and content validation
        self.validator.validate_or_raise(path)

        # Step 2: Compute SHA-256 fingerprint & generate deterministic doc_id
        file_hash = calculate_sha256(path)
        doc_id = generate_doc_id(file_hash)
        filename = path.name
        file_size = path.stat().st_size
        upload_time = datetime.now(UTC).isoformat()

        # Step 3: Duplicate detection check
        if self.duplicate_detector.is_duplicate(file_hash):
            existing = self.duplicate_detector.get_existing(file_hash)
            existing_id = existing["doc_id"] if existing else None
            if raise_on_duplicate:
                raise DuplicateDocumentError(
                    filename=filename,
                    file_hash=file_hash,
                    existing_doc_id=existing_id,
                )
            return []

        # Step 4: Extract text page by page using LangChain's PyPDFLoader
        loader = PyPDFLoader(str(path))
        raw_documents = loader.load()

        total_pages = len(raw_documents)
        normalized_documents: list[Document] = []

        # Step 5: Normalize page numbers (0-based -> 1-based) and attach unified metadata
        for idx, doc in enumerate(raw_documents):
            # Normalize Unicode characters (e.g. standardizing ligatures and whitespace)
            clean_text = unicodedata.normalize("NFKC", doc.page_content).strip()

            # PyPDFLoader yields 0-based page index in doc.metadata['page']
            raw_page = doc.metadata.get("page", idx)
            normalized_page = int(raw_page) + 1  # Standardize to 1-based indexing

            metadata = DocumentMetadata(
                source=filename,
                filename=filename,
                page=normalized_page,
                total_pages=total_pages,
                doc_id=doc_id,
                doc_hash=file_hash,
                file_size_bytes=file_size,
                upload_timestamp=upload_time,
            )

            normalized_documents.append(
                Document(
                    page_content=clean_text,
                    metadata=metadata.model_dump(),
                )
            )

        # Step 6: Register fingerprint in duplicate detector after successful extraction
        self.duplicate_detector.register(
            file_hash=file_hash,
            doc_id=doc_id,
            filename=filename,
        )

        return normalized_documents

    def load_multiple(
        self,
        file_paths: list[str | Path],
        raise_on_duplicate: bool = False,
    ) -> dict[str, list[Document]]:
        """Load multiple PDF files, returning a mapping of filename -> list of Documents.

        Args:
            file_paths: List of file paths to process.
            raise_on_duplicate: If False, duplicate files are recorded as empty lists.

        Returns:
            Dict mapping filename to extracted Document list.
        """
        results: dict[str, list[Document]] = {}
        for path in file_paths:
            p = Path(path)
            try:
                docs = self.load(p, raise_on_duplicate=raise_on_duplicate)
                results[p.name] = docs
            except DuplicateDocumentError:
                if raise_on_duplicate:
                    raise
                results[p.name] = []
        return results

    def get_summary(self, documents: list[Document]) -> IngestionSummary | None:
        """Generate an IngestionSummary from a list of loaded documents."""
        if not documents:
            return None

        first_meta = documents[0].metadata
        total_chars = sum(len(d.page_content) for d in documents)

        return IngestionSummary(
            filename=first_meta["filename"],
            doc_id=first_meta["doc_id"],
            doc_hash=first_meta["doc_hash"],
            total_pages=first_meta["total_pages"],
            total_characters=total_chars,
            file_size_bytes=first_meta["file_size_bytes"],
            upload_timestamp=first_meta["upload_timestamp"],
        )

"""Custom exceptions for document ingestion and validation."""


class IngestionError(Exception):
    """Base exception for all document ingestion and processing errors."""


class DocumentValidationError(IngestionError):
    """Base exception for document validation failures."""


class InvalidFileExtensionError(DocumentValidationError):
    """Raised when a file does not have a supported extension (e.g., .pdf)."""

    def __init__(self, filename: str, expected: str = ".pdf"):
        super().__init__(
            f"Invalid file extension for '{filename}'. Expected '{expected}'."
        )
        self.filename = filename
        self.expected = expected


class EmptyFileError(DocumentValidationError):
    """Raised when an uploaded file is empty (0 bytes)."""

    def __init__(self, filename: str):
        super().__init__(f"File '{filename}' is empty (0 bytes).")
        self.filename = filename


class FileSizeLimitExceededError(DocumentValidationError):
    """Raised when a file exceeds the configured maximum upload size limit."""

    def __init__(self, filename: str, size_bytes: int, max_bytes: int):
        size_mb = size_bytes / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        super().__init__(
            f"File '{filename}' size ({size_mb:.2f} MB) exceeds limit ({max_mb:.2f} MB)."
        )
        self.filename = filename
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class InvalidPDFSignatureError(DocumentValidationError):
    """Raised when a file does not have a valid PDF magic header (%PDF-)."""

    def __init__(self, filename: str):
        super().__init__(
            f"File '{filename}' is not a valid PDF. Missing '%PDF-' file signature."
        )
        self.filename = filename


class CorruptedPDFError(DocumentValidationError):
    """Raised when a PDF file structure is corrupted or unreadable."""

    def __init__(self, filename: str, details: str = ""):
        message = f"PDF file '{filename}' is corrupted or cannot be read."
        if details:
            message += f" Details: {details}"
        super().__init__(message)
        self.filename = filename
        self.details = details


class PasswordProtectedPDFError(DocumentValidationError):
    """Raised when a PDF is encrypted and requires a password to open."""

    def __init__(self, filename: str):
        super().__init__(
            f"PDF file '{filename}' is password-protected. Encrypted documents are not supported."
        )
        self.filename = filename


class NoExtractableTextError(DocumentValidationError):
    """Raised when a PDF contains pages but no extractable text (e.g. scanned images)."""

    def __init__(self, filename: str):
        super().__init__(
            f"PDF file '{filename}' contains no extractable text. "
            "Scanned document detected — OCR is required to extract text from images."
        )
        self.filename = filename


class DuplicateDocumentError(IngestionError):
    """Raised when an identical document (by SHA-256 hash) has already been processed."""

    def __init__(self, filename: str, file_hash: str, existing_doc_id: str | None = None):
        message = f"Document '{filename}' is a duplicate (SHA-256: {file_hash})."
        if existing_doc_id:
            message += f" Already registered with doc_id '{existing_doc_id}'."
        super().__init__(message)
        self.filename = filename
        self.file_hash = file_hash
        self.existing_doc_id = existing_doc_id


# --- Vector Store & Embedding Exceptions ---


class StorageError(Exception):
    """Base exception for all storage and vector store errors."""


class EmbeddingError(StorageError):
    """Base exception for embedding generation errors."""


class EmbeddingConnectionError(EmbeddingError):
    """Raised when connecting to the embedding model or server fails."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when an embedding vector has unexpected dimensionality."""

    def __init__(self, expected: int, received: int):
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, received {received}."
        )
        self.expected = expected
        self.received = received


class VectorStoreError(StorageError):
    """Raised when an operation on the vector store fails."""


class RegistryError(StorageError):
    """Raised when a document registry operation fails."""


# --- Generation Exceptions ---


class GenerationError(Exception):
    """Base exception for text generation errors."""


class GenerationConnectionError(GenerationError):
    """Raised when connecting to the LLM generation service fails."""


# --- Conversation Exceptions ---


class ConversationError(Exception):
    """Base exception for conversation and session management errors."""


class SessionNotFoundError(ConversationError):
    """Raised when a requested session_id does not exist."""

    def __init__(self, session_id: str):
        super().__init__(f"Conversation session '{session_id}' not found.")
        self.session_id = session_id


class QueryReformulationError(ConversationError):
    """Raised when follow-up query reformulation fails critically."""

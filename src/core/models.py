"""Data models and schemas for document ingestion and validation."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProcessingStatus(StrEnum):
    """Lifecycle status of a document in the registry."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Persistent record of a document tracked in the SQLite registry."""

    doc_id: str = Field(description="Deterministic UUIDv5 derived from SHA-256 hash")
    filename: str = Field(description="Original filename at upload time")
    file_hash: str = Field(description="SHA-256 hex digest of file contents")
    file_path: str = Field(description="Absolute path to the original PDF file on disk")
    file_size_bytes: int = Field(ge=0, description="File size in bytes")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages extracted")
    total_chunks: int = Field(default=0, ge=0, description="Number of chunks indexed")
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current lifecycle status of the document",
    )
    error_message: str | None = Field(
        default=None,
        description="Error detail if status is FAILED; None otherwise",
    )
    upload_timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of initial registration",
    )
    completed_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp when ingestion completed successfully",
    )


class DocumentMetadata(BaseModel):
    """Metadata schema attached to every extracted Document page."""

    source: str = Field(description="Normalized source identifier (filename)")
    filename: str = Field(description="Original filename of the document")
    page: int = Field(ge=1, description="1-based page number")
    total_pages: int = Field(ge=1, description="Total number of pages in the source document")
    doc_id: str = Field(description="Unique, deterministic UUID for the document")
    doc_hash: str = Field(description="SHA-256 fingerprint of the document contents")
    file_size_bytes: int = Field(ge=0, description="Size of the source file in bytes")
    upload_timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of ingestion",
    )


class ValidationResult(BaseModel):
    """Outcome of PDF file validation."""

    is_valid: bool = Field(description="Whether the PDF passed all validation checks")
    filename: str = Field(description="Name of the evaluated file")
    file_size_bytes: int = Field(description="File size in bytes")
    page_count: int = Field(default=0, description="Number of pages detected")
    has_text: bool = Field(default=False, description="Whether extractable text was found")
    is_encrypted: bool = Field(default=False, description="Whether the file is password-protected")
    error_message: str | None = Field(default=None, description="Validation failure reason")
    requires_ocr: bool = Field(default=False, description="True if scanned images")


class IngestionSummary(BaseModel):
    """Summary of a document ingestion run."""

    filename: str
    doc_id: str
    doc_hash: str
    total_pages: int
    total_characters: int
    file_size_bytes: int
    upload_timestamp: str


class ChunkMetadata(DocumentMetadata):
    """Metadata schema attached to each split chunk, inheriting document metadata."""

    chunk_index: int = Field(
        ge=0, description="0-based sequence index of the chunk in the document"
    )
    chunk_id: str = Field(
        description="Unique deterministic chunk identifier: {doc_id}_c{chunk_index}"
    )
    char_count: int = Field(ge=0, description="Character length of chunk text")


class ChunkingConfig(BaseModel):
    """Configuration parameters for recursive character text splitting."""

    chunk_size: int = Field(
        default=800, ge=50, le=8000, description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=150, ge=0, description="Overlap between consecutive chunks"
    )


class ChunkingEvaluationResult(BaseModel):
    """Comparative evaluation metrics for a chunking configuration."""

    config_name: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    total_characters: int


class RetrievalResult(BaseModel):
    """Structured result returned from vector retrieval."""

    chunk_id: str = Field(description="Unique deterministic identifier of the retrieved chunk")
    doc_id: str = Field(description="Unique identifier of the source document")
    filename: str = Field(description="Original filename of the source document")
    page: int = Field(ge=1, description="1-based page number where the chunk originates")
    chunk_index: int = Field(ge=0, description="0-based sequence index within the document")
    content: str = Field(description="Extracted text content of the chunk")
    distance: float = Field(
        description="Cosine distance from query vector (0.0 = identical, lower is closer)"
    )
    score: float = Field(
        description=(
            "Cosine similarity score ranging from -1.0 to +1.0 "
            "(1.0 - distance, higher is more similar)"
        )
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Complete metadata dictionary attached to chunk"
    )


class RetrievalQuery(BaseModel):
    """Parameters for executing a semantic retrieval query."""

    query: str = Field(min_length=1, description="Natural language search query")
    top_k: int = Field(default=4, ge=1, le=50, description="Number of results to retrieve")
    filter_criteria: dict[str, Any] | None = Field(
        default=None, description="Metadata filter dictionary for Chroma where clause"
    )


class Citation(BaseModel):
    """Page-level document citation referenced in an answer."""

    filename: str = Field(description="Name of the cited source file")
    page: int = Field(ge=1, description="1-based page number")
    chunk_id: str = Field(default="", description="Identifier of the supporting chunk if verified")
    doc_id: str = Field(default="", description="Source document identifier")
    snippet: str = Field(default="", description="Relevant text excerpt from the cited chunk")
    is_verified: bool = Field(
        default=False,
        description="True if citation was validated against an actual retrieved context chunk",
    )
    raw_reference: str = Field(
        default="", description="Original reference token found in answer text"
    )


class EvidenceCard(BaseModel):
    """Detailed evidence card mapping an answer citation to source PDF coordinates."""

    citation_index: int = Field(ge=1, description="1-based sequential citation marker index")
    doc_id: str = Field(description="Unique source document identifier")
    filename: str = Field(description="Original source PDF filename")
    page: int = Field(ge=1, description="1-based cited page number")
    total_pages: int = Field(default=1, ge=1, description="Total pages in the source document")
    chunk_id: str = Field(default="", description="Unique identifier of supporting text chunk")
    chunk_index: int = Field(default=0, ge=0, description="0-based chunk sequence index")
    snippet: str = Field(default="", description="Extracted key quote or supporting excerpt")
    similarity_score: float = Field(
        default=0.0, description="Cosine similarity score of supporting chunk"
    )
    distance: float = Field(default=0.0, description="Cosine distance of supporting chunk")
    page_url_fragment: str = Field(
        default="", description="URL anchor fragment for PDF viewer, e.g. #page=2"
    )
    is_verified: bool = Field(
        default=True, description="True if citation matched an actual retrieved passage"
    )
    raw_reference: str = Field(
        default="", description="Original reference token extracted from generated answer"
    )


class EvidenceMap(BaseModel):
    """Structured collection of verified evidence cards and annotated answer text."""

    evidence_cards: list[EvidenceCard] = Field(
        default_factory=list, description="List of verified evidence cards"
    )
    annotated_answer: str = Field(
        default="", description="Answer text with standardized [1], [2] citation markers"
    )
    cited_pages_by_document: dict[str, list[int]] = Field(
        default_factory=dict, description="Mapping of filename to sorted list of cited page numbers"
    )
    unverified_references: list[str] = Field(
        default_factory=list, description="Extracted citation tokens that could not be verified"
    )
    total_citations: int = Field(default=0, description="Total citation markers detected")
    verified_count: int = Field(default=0, description="Count of successfully verified citations")


class RAGResponse(BaseModel):
    """Complete grounded answer with citations and retrieval metadata."""

    answer: str = Field(description="Synthesized natural language answer")
    query: str = Field(description="Original user query")
    sources: list[RetrievalResult] = Field(
        default_factory=list, description="Retrieved chunks provided as context"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Structured citations extracted and verified from answer"
    )
    evidence_map: EvidenceMap | None = Field(
        default=None, description="Full evidence mapping package for evidence cards and PDF viewer"
    )
    latency_seconds: float = Field(ge=0.0, description="End-to-end generation latency in seconds")
    model_name: str = Field(description="Name of the generation LLM")
    is_grounded: bool = Field(
        default=True,
        description=(
            "Heuristic refusal check indicating the answer was synthesized from retrieved "
            "context passages without triggering a lack-of-context refusal or empty-retrieval "
            "guard. Does not provide formal sentence-level logical proof."
        ),
    )


class ChatMessage(BaseModel):
    """Individual message in a conversation session."""

    message_id: str = Field(description="Unique identifier for the message")
    session_id: str = Field(description="Conversation session identifier")
    role: str = Field(description="Message author role: 'user' or 'assistant'")
    content: str = Field(description="Original, unmodified message content")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of the message",
    )
    standalone_query: str | None = Field(
        default=None,
        description="Reformulated standalone retrieval query (for assistant responses)",
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Verified citations associated with response"
    )
    evidence_cards: list[EvidenceCard] = Field(
        default_factory=list, description="Structured evidence cards for UI presentation"
    )
    sources: list[RetrievalResult] = Field(
        default_factory=list, description="Retrieved chunks used to synthesize response"
    )
    latency_seconds: float | None = Field(
        default=None, description="Generation latency in seconds"
    )
    is_grounded: bool = Field(
        default=True, description="Grounding heuristic flag for assistant response"
    )


class ConversationSession(BaseModel):
    """Managed conversation session tracking history and associated documents."""

    session_id: str = Field(description="Unique conversation session identifier")
    title: str = Field(default="New Conversation", description="Human-readable session title")
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of creation",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of last message",
    )
    selected_doc_ids: list[str] = Field(
        default_factory=list,
        description="List of document IDs scope-filtering this conversation",
    )
    messages: list[ChatMessage] = Field(
        default_factory=list, description="Chronological sequence of messages"
    )


class SessionSummary(BaseModel):
    """Metadata summary of a conversation session for sidebar listings."""

    session_id: str = Field(description="Unique conversation session identifier")
    title: str = Field(description="Conversation title")
    message_count: int = Field(ge=0, description="Total count of messages in session")
    created_at: str = Field(description="Session creation timestamp")
    updated_at: str = Field(description="Session last updated timestamp")
    selected_doc_ids: list[str] = Field(
        default_factory=list, description="Document IDs associated with session"
    )


class ConversationalRAGResponse(BaseModel):
    """Result of a conversational RAG turn including reformulation details."""

    answer: str = Field(description="Synthesized grounded answer")
    user_query: str = Field(description="Original raw user input question")
    standalone_query: str = Field(
        description="Contextually rewritten query used for vector retrieval"
    )
    session_id: str = Field(description="Active conversation session ID")
    citations: list[Citation] = Field(
        default_factory=list, description="Verified citations supporting the answer"
    )
    evidence_map: EvidenceMap | None = Field(
        default=None, description="Full evidence mapping package with cards and annotated text"
    )
    sources: list[RetrievalResult] = Field(
        default_factory=list, description="Retrieved document chunks"
    )
    latency_seconds: float = Field(ge=0.0, description="Total response latency in seconds")
    reformulation_latency_seconds: float = Field(
        default=0.0, ge=0.0, description="Latency spent on query reformulation"
    )
    is_grounded: bool = Field(
        default=True, description="Grounding status flag for response"
    )

"""Application service layer and high-level coordinator for DocuMind.

Coordinates document ingestion, deletion, conversation sessions, document filtering,
query reformulation, vector retrieval, grounded RAG generation, and evidence mapping.
Serves as the single unified API for application interfaces (such as Streamlit),
encapsulating business logic and standardizing structured results and error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.conversation.chain import ConversationalRAGChain
from src.conversation.manager import ConversationManager
from src.conversation.reformulator import QueryReformulator
from src.core.exceptions import (
    ConversationError,
    DocumentValidationError,
    DuplicateDocumentError,
    EmbeddingConnectionError,
    GenerationConnectionError,
    GenerationError,
    IngestionError,
    RegistryError,
    SessionNotFoundError,
    StorageError,
    VectorStoreError,
)
from src.core.models import (
    ConversationalRAGResponse,
    ConversationSession,
    DocumentRecord,
    ProcessingStatus,
    SessionSummary,
)
from src.generation.chain import RAGGenerationChain
from src.ingestion.loader import PDFDocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.retrieval.retriever import VectorRetriever
from src.storage.embeddings import NomicEmbeddingAdapter
from src.storage.ingestion_pipeline import IngestionPipeline
from src.storage.registry import DocumentRegistry
from src.storage.vector_store import ChromaVectorStore


@dataclass
class ServiceResult:
    """Standardized outcome wrapper for UI consumption.

    Attributes:
        success: Whether the operation succeeded.
        data: The payload data (if successful), or None.
        error_message: User-friendly error message if failed; None if successful.
        error_code: Categorical error identifier for UI handling.
    """

    success: bool
    data: Any = None
    error_message: str | None = None
    error_code: str | None = None


class DocuMindService:
    """Unified application service facade for DocuMind.

    Encapsulates registry, ingestion pipeline, conversation manager,
    vector store, retriever, and conversational RAG chain.
    """

    def __init__(
        self,
        base_dir: str | Path = "./data",
        registry_path: str | Path | None = None,
        chroma_dir: str | Path | None = None,
        conversations_dir: str | Path | None = None,
        embedding_adapter: NomicEmbeddingAdapter | None = None,
        llm_model: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        base = Path(base_dir).resolve()
        reg_path = Path(registry_path).resolve() if registry_path else base / "registry.db"
        vec_dir = Path(chroma_dir).resolve() if chroma_dir else base / "vectorstore"
        conv_dir = (
            Path(conversations_dir).resolve() if conversations_dir else base / "conversations"
        )

        # 1. Storage & Registry
        self.registry = DocumentRegistry(db_path=reg_path)
        self.embedding_adapter = embedding_adapter or NomicEmbeddingAdapter(base_url=ollama_url)
        self.vector_store = ChromaVectorStore(
            persist_directory=vec_dir,
            embedding_adapter=self.embedding_adapter,
        )

        # 2. Ingestion Pipeline
        self.loader = PDFDocumentLoader()
        self.splitter = DocumentSplitter()
        self.ingestion_pipeline = IngestionPipeline(
            registry=self.registry,
            vector_store=self.vector_store,
            loader=self.loader,
            splitter=self.splitter,
        )

        # 3. Retrieval & Generation
        self.retriever = VectorRetriever(vector_store=self.vector_store)
        self.generator = RAGGenerationChain(
            retriever=self.retriever,
            model_name=llm_model,
            base_url=ollama_url,
            temperature=temperature,
        )

        # 4. Conversation & Query Management
        self.conversation_manager = ConversationManager(persist_dir=conv_dir)
        self.reformulator = QueryReformulator(
            model_name=llm_model,
            base_url=ollama_url,
        )
        self.conversational_chain = ConversationalRAGChain(
            conversation_manager=self.conversation_manager,
            query_reformulator=self.reformulator,
            retriever=self.retriever,
            rag_generator=self.generator,
        )

        logger.info(
            "DocuMindService initialized successfully (base_dir='{}', model='{}')",
            base,
            llm_model,
        )

    # ------------------------------------------------------------------
    # Document Management
    # ------------------------------------------------------------------

    def ingest_document(self, file_path: str | Path) -> ServiceResult:
        """Ingest a PDF file into the registry and vector store."""
        try:
            record = self.ingestion_pipeline.ingest(file_path)
            return ServiceResult(success=True, data=record)
        except DocumentValidationError as e:
            logger.warning("Document validation failed: {}", e)
            return ServiceResult(
                success=False,
                error_message=f"Validation failed: {e}",
                error_code="VALIDATION_ERROR",
            )
        except DuplicateDocumentError as e:
            logger.info("Document duplicate detected: {}", e)
            existing_rec = (
                self.ingestion_pipeline.get_record(e.existing_doc_id)
                if e.existing_doc_id
                else None
            )
            return ServiceResult(
                success=True,
                data=existing_rec,
            )
        except IngestionError as e:
            logger.error("Ingestion failed: {}", e)
            return ServiceResult(
                success=False,
                error_message=f"Ingestion failed: {e}",
                error_code="INGESTION_ERROR",
            )
        except (StorageError, RegistryError, VectorStoreError) as e:
            logger.error("Storage error during ingestion: {}", e)
            return ServiceResult(
                success=False,
                error_message="A storage error occurred while processing the document.",
                error_code="STORAGE_ERROR",
            )
        except Exception as e:
            logger.exception("Unexpected error during ingestion: {}", e)
            return ServiceResult(
                success=False,
                error_message="An unexpected internal error occurred during ingestion.",
                error_code="INTERNAL_ERROR",
            )

    def list_documents(self, status: ProcessingStatus | None = None) -> list[DocumentRecord]:
        """List all documents or filter by processing status."""
        if status:
            return self.registry.list_by_status(status)
        return self.ingestion_pipeline.list_documents()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Retrieve a document record by doc_id."""
        return self.ingestion_pipeline.get_record(doc_id)

    def delete_document(self, doc_id: str) -> ServiceResult:
        """Delete a document from vector store and registry, updating sessions if needed."""
        try:
            existed = self.ingestion_pipeline.delete(doc_id)
            if not existed:
                return ServiceResult(
                    success=False,
                    error_message=f"Document '{doc_id}' not found.",
                    error_code="DOCUMENT_NOT_FOUND",
                )

            # Clean up references to this doc_id from conversation sessions
            for session_summary in self.conversation_manager.list_sessions():
                session = self.conversation_manager.get_session(session_summary.session_id)
                if doc_id in session.selected_doc_ids:
                    new_docs = [d for d in session.selected_doc_ids if d != doc_id]
                    self.conversation_manager.update_session_documents(
                        session.session_id, new_docs
                    )

            return ServiceResult(success=True, data={"doc_id": doc_id, "deleted": True})
        except Exception as e:
            logger.exception("Failed to delete document '{}': {}", doc_id, e)
            return ServiceResult(
                success=False,
                error_message=f"Failed to delete document: {e}",
                error_code="DELETE_ERROR",
            )

    # ------------------------------------------------------------------
    # Conversation & Session Management
    # ------------------------------------------------------------------

    def create_session(
        self,
        title: str = "New Conversation",
        selected_doc_ids: list[str] | None = None,
    ) -> ServiceResult:
        """Create a new conversation session, optionally binding it to documents."""
        try:
            # Validate that selected doc_ids exist in registry
            valid_doc_ids: list[str] = []
            if selected_doc_ids:
                for did in selected_doc_ids:
                    rec = self.registry.get(did)
                    if rec and rec.status == ProcessingStatus.COMPLETED:
                        valid_doc_ids.append(did)
                    else:
                        logger.warning(
                            "Doc ID '{}' is invalid or not COMPLETED; skipped for session", did
                        )

            session = self.conversation_manager.create_session(
                title=title,
                selected_doc_ids=valid_doc_ids,
            )
            return ServiceResult(success=True, data=session)
        except Exception as e:
            logger.exception("Failed to create conversation session: {}", e)
            return ServiceResult(
                success=False,
                error_message="Failed to create session.",
                error_code="SESSION_CREATE_ERROR",
            )

    def list_sessions(self) -> list[SessionSummary]:
        """List summary of all conversation sessions."""
        return self.conversation_manager.list_sessions()

    def get_session(self, session_id: str) -> ServiceResult:
        """Retrieve full details and message history of a conversation session."""
        try:
            session = self.conversation_manager.get_session(session_id)
            return ServiceResult(success=True, data=session)
        except SessionNotFoundError:
            return ServiceResult(
                success=False,
                error_message=f"Session '{session_id}' not found.",
                error_code="SESSION_NOT_FOUND",
            )
        except Exception as e:
            logger.exception("Error getting session '{}': {}", session_id, e)
            return ServiceResult(
                success=False,
                error_message="Error retrieving session.",
                error_code="SESSION_GET_ERROR",
            )

    def set_active_session(self, session_id: str) -> ServiceResult:
        """Set the active conversation session."""
        try:
            self.conversation_manager.set_active_session(session_id)
            return ServiceResult(success=True, data={"active_session_id": session_id})
        except SessionNotFoundError:
            return ServiceResult(
                success=False,
                error_message=f"Session '{session_id}' not found.",
                error_code="SESSION_NOT_FOUND",
            )

    def get_active_session(self) -> ConversationSession | None:
        """Retrieve the currently active conversation session."""
        return self.conversation_manager.get_active_session()

    def update_session_documents(
        self, session_id: str, doc_ids: list[str]
    ) -> ServiceResult:
        """Update the document scope filter for an existing conversation session."""
        try:
            valid_doc_ids: list[str] = []
            for did in doc_ids:
                rec = self.registry.get(did)
                if rec and rec.status == ProcessingStatus.COMPLETED:
                    valid_doc_ids.append(did)
                else:
                    logger.warning("Doc ID '{}' is not COMPLETED; skipped for session", did)

            self.conversation_manager.update_session_documents(session_id, valid_doc_ids)
            updated = self.conversation_manager.get_session(session_id)
            return ServiceResult(success=True, data=updated)
        except SessionNotFoundError:
            return ServiceResult(
                success=False,
                error_message=f"Session '{session_id}' not found.",
                error_code="SESSION_NOT_FOUND",
            )
        except Exception as e:
            logger.exception("Failed to update documents for session '{}': {}", session_id, e)
            return ServiceResult(
                success=False,
                error_message="Failed to update session documents.",
                error_code="SESSION_UPDATE_ERROR",
            )

    def update_session_title(self, session_id: str, new_title: str) -> ServiceResult:
        """Update the human-readable title of a session."""
        try:
            self.conversation_manager.update_session_title(session_id, new_title)
            updated = self.conversation_manager.get_session(session_id)
            return ServiceResult(success=True, data=updated)
        except SessionNotFoundError:
            return ServiceResult(
                success=False,
                error_message=f"Session '{session_id}' not found.",
                error_code="SESSION_NOT_FOUND",
            )

    def clear_session_messages(self, session_id: str) -> ServiceResult:
        """Clear message history in a session while preserving document selections."""
        try:
            self.conversation_manager.clear_session_messages(session_id)
            updated = self.conversation_manager.get_session(session_id)
            return ServiceResult(success=True, data=updated)
        except SessionNotFoundError:
            return ServiceResult(
                success=False,
                error_message=f"Session '{session_id}' not found.",
                error_code="SESSION_NOT_FOUND",
            )

    def delete_session(self, session_id: str) -> ServiceResult:
        """Delete a conversation session."""
        existed = self.conversation_manager.delete_session(session_id)
        if not existed:
            return ServiceResult(
                success=False,
                error_message=f"Session '{session_id}' not found.",
                error_code="SESSION_NOT_FOUND",
            )
        return ServiceResult(success=True, data={"session_id": session_id, "deleted": True})

    # ------------------------------------------------------------------
    # Query & Conversational RAG
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> ServiceResult:
        """Execute a conversational question answering turn.

        Args:
            query: The user's question.
            session_id: Session ID to run in; defaults to active session if None.
            top_k: Retrieval chunk limit.

        Returns:
            ServiceResult containing ConversationalRAGResponse on success.
        """
        target_session_id = session_id or self.conversation_manager.active_session_id
        if not target_session_id:
            return ServiceResult(
                success=False,
                error_message="No active conversation session. Please create or select a session.",
                error_code="NO_ACTIVE_SESSION",
            )

        cleaned_query = query.strip()
        if not cleaned_query:
            return ServiceResult(
                success=False,
                error_message="Query cannot be empty.",
                error_code="EMPTY_QUERY",
            )

        try:
            response: ConversationalRAGResponse = self.conversational_chain.ask(
                session_id=target_session_id,
                query=cleaned_query,
                top_k=top_k,
            )
            return ServiceResult(success=True, data=response)
        except SessionNotFoundError:
            return ServiceResult(
                success=False,
                error_message=f"Session '{target_session_id}' does not exist.",
                error_code="SESSION_NOT_FOUND",
            )
        except (GenerationConnectionError, EmbeddingConnectionError, VectorStoreError) as e:
            # If VectorStoreError wraps an EmbeddingConnectionError, treat it
            # as an Ollama connection issue.
            cause_is_embedding = isinstance(e.__cause__, EmbeddingConnectionError)
            if isinstance(e, VectorStoreError) and hasattr(e, "__cause__") and cause_is_embedding:
                logger.error("Ollama embedding connection error wrapped in VectorStoreError: {}", e)
                return ServiceResult(
                    success=False,
                    error_message=(
                        "Could not connect to Ollama service. "
                        "Please ensure Ollama is running at http://localhost:11434."
                    ),
                    error_code="OLLAMA_CONNECTION_ERROR",
                )
            # Otherwise handle as before for direct connection errors
            logger.error("Ollama connection error: {}", e)
            return ServiceResult(
                success=False,
                error_message=(
                    "Could not connect to Ollama service. "
                    "Please ensure Ollama is running at http://localhost:11434."
                ),
                error_code="OLLAMA_CONNECTION_ERROR",
            )
            logger.error("Ollama connection error: {}", e)
            return ServiceResult(
                success=False,
                error_message=(
                    "Could not connect to Ollama service. "
                    "Please ensure Ollama is running at http://localhost:11434."
                ),
                error_code="OLLAMA_CONNECTION_ERROR",
            )
        except GenerationError as e:
            logger.error("LLM generation error: {}", e)
            return ServiceResult(
                success=False,
                error_message=f"Generation error: {e}",
                error_code="GENERATION_ERROR",
            )
        except ConversationError as e:
            logger.error("Conversation workflow error: {}", e)
            return ServiceResult(
                success=False,
                error_message=f"Conversation error: {e}",
                error_code="CONVERSATION_ERROR",
            )
        except Exception as e:
            logger.exception("Unexpected error processing question: {}", e)
            return ServiceResult(
                success=False,
                error_message="An internal error occurred while generating the answer.",
                error_code="INTERNAL_ERROR",
            )

    # ------------------------------------------------------------------
    # Diagnostics & Health
    # ------------------------------------------------------------------

    def get_system_status(self) -> dict[str, Any]:
        """Return diagnostic metrics across registry, vector store, and sessions."""
        return {
            "total_documents": self.registry.count(),
            "completed_documents": len(self.registry.list_by_status(ProcessingStatus.COMPLETED)),
            "failed_documents": len(self.registry.list_by_status(ProcessingStatus.FAILED)),
            "vector_store_chunks": self.vector_store.count(),
            "total_sessions": len(self.conversation_manager.list_sessions()),
            "active_session_id": self.conversation_manager.active_session_id,
        }

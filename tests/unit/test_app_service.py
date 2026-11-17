"""Integration tests for DocuMindService application service facade.

Tests end-to-end workflows including:
- Document ingestion, listing, and deletion.
- Multi-document and multi-session isolation.
- Conversational querying with document scope filtering.
- Dynamic document selection and session updates.
- Consistent ServiceResult responses and user-safe error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import ConversationalRAGResponse, ProcessingStatus
from src.service.app_service import DocuMindService, ServiceResult
from src.storage.embeddings import EXPECTED_EMBEDDING_DIM, NomicEmbeddingAdapter


@pytest.fixture
def mock_embedding_adapter() -> MagicMock:
    """Deterministic embedding adapter."""
    adapter = MagicMock(spec=NomicEmbeddingAdapter)
    adapter.expected_dim = EXPECTED_EMBEDDING_DIM
    adapter.model = "nomic-embed-text"

    def _embed_docs(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * EXPECTED_EMBEDDING_DIM
            if "alpha" in text.lower():
                vec[0] = 1.0
            elif "beta" in text.lower():
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            vectors.append(vec)
        return vectors

    def _embed_query(query: str) -> list[float]:
        vec = [0.0] * EXPECTED_EMBEDDING_DIM
        if "alpha" in query.lower():
            vec[0] = 1.0
        elif "beta" in query.lower():
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        return vec

    adapter.embed_documents.side_effect = _embed_docs
    adapter.embed_query.side_effect = _embed_query
    return adapter


@pytest.fixture
def service(tmp_path: Path, mock_embedding_adapter: MagicMock) -> DocuMindService:
    """Create isolated DocuMindService instance."""
    return DocuMindService(
        base_dir=tmp_path / "app_data",
        embedding_adapter=mock_embedding_adapter,
        llm_model="llama3.2:3b",
    )


# ---------------------------------------------------------------------------
# Document Management Workflows
# ---------------------------------------------------------------------------


def test_service_document_lifecycle(service: DocuMindService, pdf_factory) -> None:
    """Test full document lifecycle: ingest, list, get, and delete."""
    pdf = pdf_factory("architecture.pdf", ["Alpha architecture specs.", "Chunking specs."])

    # 1. Ingest
    res: ServiceResult = service.ingest_document(pdf)
    assert res.success is True
    assert res.error_message is None
    record = res.data
    assert record.filename == "architecture.pdf"
    assert record.status == ProcessingStatus.COMPLETED
    assert record.total_pages == 2

    # 2. List
    docs = service.list_documents()
    assert len(docs) == 1
    assert docs[0].doc_id == record.doc_id

    # 3. Get
    fetched = service.get_document(record.doc_id)
    assert fetched is not None
    assert fetched.doc_id == record.doc_id

    # 4. System status
    status = service.get_system_status()
    assert status["total_documents"] == 1
    assert status["completed_documents"] == 1
    assert status["vector_store_chunks"] > 0

    # 5. Delete
    del_res = service.delete_document(record.doc_id)
    assert del_res.success is True
    assert service.get_document(record.doc_id) is None
    assert len(service.list_documents()) == 0


def test_service_ingest_validation_failure(service: DocuMindService, tmp_path: Path) -> None:
    """Validation failure returns structured error without exposing internal traceback."""
    invalid_file = tmp_path / "not_a_pdf.txt"
    invalid_file.write_text("Hello plain text")

    res = service.ingest_document(invalid_file)
    assert res.success is False
    assert res.error_code == "VALIDATION_ERROR"
    assert "Invalid file extension" in (res.error_message or "")


def test_service_delete_nonexistent_document(service: DocuMindService) -> None:
    """Deleting unknown doc_id returns error result."""
    res = service.delete_document("missing-doc-id")
    assert res.success is False
    assert res.error_code == "DOCUMENT_NOT_FOUND"


# ---------------------------------------------------------------------------
# Session & Conversation Workflows
# ---------------------------------------------------------------------------


def test_service_session_lifecycle(service: DocuMindService, pdf_factory) -> None:
    """Test creating sessions, binding documents, updating titles, and deletion."""
    pdf = pdf_factory("doc1.pdf", ["Alpha specifications."])
    doc_res = service.ingest_document(pdf)
    doc_id = doc_res.data.doc_id

    # Create session with document
    create_res = service.create_session(
        title="Specs Discussion",
        selected_doc_ids=[doc_id],
    )
    assert create_res.success is True
    session = create_res.data
    assert session.title == "Specs Discussion"
    assert session.selected_doc_ids == [doc_id]

    # Verify active session
    active = service.get_active_session()
    assert active is not None
    assert active.session_id == session.session_id

    # Update title
    update_res = service.update_session_title(session.session_id, "Updated Specs")
    assert update_res.success is True
    assert update_res.data.title == "Updated Specs"

    # Update documents
    update_docs_res = service.update_session_documents(session.session_id, [])
    assert update_docs_res.success is True
    assert update_docs_res.data.selected_doc_ids == []

    # Delete session
    del_res = service.delete_session(session.session_id)
    assert del_res.success is True
    assert len(service.list_sessions()) == 0


def test_service_deletion_cascades_to_session_filters(
    service: DocuMindService, pdf_factory
) -> None:
    """Deleting a document automatically purges it from any session's selected_doc_ids."""
    pdf1 = pdf_factory("keep.pdf", ["Beta content."])
    pdf2 = pdf_factory("remove.pdf", ["Alpha content to remove."])

    rec1 = service.ingest_document(pdf1).data
    rec2 = service.ingest_document(pdf2).data

    sess = service.create_session(
        title="Multi-Doc Session",
        selected_doc_ids=[rec1.doc_id, rec2.doc_id],
    ).data
    assert len(sess.selected_doc_ids) == 2

    # Delete rec2
    service.delete_document(rec2.doc_id)

    # Session filter updated without deleting the session or conversation history
    updated_sess = service.get_session(sess.session_id).data
    assert rec2.doc_id not in updated_sess.selected_doc_ids
    assert updated_sess.selected_doc_ids == [rec1.doc_id]


# ---------------------------------------------------------------------------
# Conversational Querying Workflows
# ---------------------------------------------------------------------------


def test_service_ask_workflow_with_scoping(service: DocuMindService, pdf_factory) -> None:
    """End-to-end question answering turn with document scoping and citations."""
    pdf_alpha = pdf_factory("alpha.pdf", ["Alpha architecture uses ChromaDB vectors. [Page 1]"])
    pdf_beta = pdf_factory("beta.pdf", ["Beta architecture uses SQLite registry. [Page 1]"])

    rec_alpha = service.ingest_document(pdf_alpha).data
    service.ingest_document(pdf_beta)

    # Session 1: scoped strictly to Alpha
    sess1 = service.create_session(title="Alpha Thread", selected_doc_ids=[rec_alpha.doc_id]).data

    # Mock generator and reformulator methods to test service orchestration
    from src.core.models import RAGResponse

    mock_rag_response = RAGResponse(
        answer="Alpha uses ChromaDB vectors. [Page 1, alpha.pdf]",
        query="What does Alpha use?",
        sources=[
            service.retriever.retrieve(
                query="What does Alpha use?",
                filter_criteria={"doc_id": rec_alpha.doc_id},
            )[0]
        ],
        citations=[],
        latency_seconds=0.1,
        model_name="llama3.2:3b",
        is_grounded=True,
    )

    with patch.object(
        service.conversational_chain.generator,
        "generate_from_context",
        return_value=mock_rag_response,
    ), patch.object(
        service.conversational_chain.reformulator,
        "reformulate",
        return_value=("What does Alpha architecture use?", 0.05),
    ):
        ask_res: ServiceResult = service.ask(
            query="What does Alpha use?",
            session_id=sess1.session_id,
        )

        assert ask_res.success is True
        rag_resp: ConversationalRAGResponse = ask_res.data
        assert "ChromaDB" in rag_resp.answer
        assert len(rag_resp.sources) > 0
        # All retrieved sources must belong strictly to rec_alpha
        for source in rag_resp.sources:
            assert source.doc_id == rec_alpha.doc_id

        # Verify message persisted in session
        updated_sess = service.get_session(sess1.session_id).data
        assert len(updated_sess.messages) == 2  # user + assistant


def test_service_ask_error_handling(service: DocuMindService) -> None:
    """Ask error handling for missing active session and empty query."""
    # No active session
    res = service.ask("Hello without session")
    assert res.success is False
    assert res.error_code == "NO_ACTIVE_SESSION"

    # Create session, but pass empty query
    sess = service.create_session("Test Session").data
    res_empty = service.ask("   ", session_id=sess.session_id)
    assert res_empty.success is False
    assert res_empty.error_code == "EMPTY_QUERY"


def test_service_clear_session_messages(service: DocuMindService, pdf_factory) -> None:
    """Clearing messages resets history while keeping document selections intact."""
    pdf = pdf_factory("doc.pdf", ["Some facts."])
    rec = service.ingest_document(pdf).data

    sess = service.create_session("Chat", selected_doc_ids=[rec.doc_id]).data

    # Add dummy message via conversation manager
    service.conversation_manager.add_message(sess.session_id, role="user", content="Hi")
    assert len(service.get_session(sess.session_id).data.messages) == 1

    # Clear messages
    res = service.clear_session_messages(sess.session_id)
    assert res.success is True
    assert len(res.data.messages) == 0
    assert res.data.selected_doc_ids == [rec.doc_id]

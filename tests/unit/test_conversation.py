"""Unit tests for conversation management, query reformulation, and conversational RAG."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.conversation.chain import ConversationalRAGChain
from src.conversation.manager import ConversationManager
from src.conversation.reformulator import QueryReformulator
from src.core.exceptions import SessionNotFoundError
from src.core.models import (
    ChatMessage,
    Citation,
    RAGResponse,
    RetrievalResult,
)
from src.generation.chain import RAGGenerationChain
from src.retrieval.retriever import VectorRetriever


@pytest.fixture
def temp_conv_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary directory for conversation files."""
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir


def test_session_lifecycle(temp_conv_dir):
    """Verify session creation, retrieval, listing, active tracking, and deletion."""
    mgr = ConversationManager(persist_dir=temp_conv_dir)
    assert len(mgr.list_sessions()) == 0

    # 1. Create sessions
    s1 = mgr.create_session(title="Financial Q3 Discussion", selected_doc_ids=["doc_fin_1"])
    s2 = mgr.create_session(title="Architecture Specs")

    assert s1.session_id != s2.session_id
    assert mgr.active_session_id == s2.session_id
    assert len(mgr.list_sessions()) == 2

    # 2. Get session
    retrieved_s1 = mgr.get_session(s1.session_id)
    assert retrieved_s1.title == "Financial Q3 Discussion"
    assert retrieved_s1.selected_doc_ids == ["doc_fin_1"]

    # 3. Switch active session
    mgr.set_active_session(s1.session_id)
    assert mgr.active_session_id == s1.session_id
    assert mgr.get_active_session().session_id == s1.session_id

    # 4. Update title and docs
    mgr.update_session_title(s1.session_id, "Q3 Earnings Deep Dive")
    mgr.update_session_documents(s1.session_id, ["doc_fin_1", "doc_fin_2"])
    updated_s1 = mgr.get_session(s1.session_id)
    assert updated_s1.title == "Q3 Earnings Deep Dive"
    assert updated_s1.selected_doc_ids == ["doc_fin_1", "doc_fin_2"]

    # 5. Delete session
    deleted = mgr.delete_session(s1.session_id)
    assert deleted is True
    assert len(mgr.list_sessions()) == 1
    # Active session should adjust to s2
    assert mgr.active_session_id == s2.session_id

    # Non-existent session
    with pytest.raises(SessionNotFoundError):
        mgr.get_session("non_existent_id")


def test_session_persistence(temp_conv_dir):
    """Verify sessions and message histories persist across manager re-instantiation."""
    # Instance 1: Create session and add turns
    mgr1 = ConversationManager(persist_dir=temp_conv_dir)
    s = mgr1.create_session(title="Persistent Thread")
    mgr1.add_message(s.session_id, role="user", content="Hello, what is DocuMind?")
    mgr1.add_message(s.session_id, role="assistant", content="DocuMind is a local RAG assistant.")

    # Instance 2: Reload from same directory
    mgr2 = ConversationManager(persist_dir=temp_conv_dir)
    assert len(mgr2.list_sessions()) == 1

    loaded = mgr2.get_session(s.session_id)
    assert loaded.title == "Persistent Thread"
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "Hello, what is DocuMind?"
    assert loaded.messages[1].role == "assistant"
    assert loaded.messages[1].content == "DocuMind is a local RAG assistant."


def test_conversation_isolation(temp_conv_dir):
    """Verify messages added to one session never leak into another session."""
    mgr = ConversationManager(persist_dir=temp_conv_dir)
    sess_a = mgr.create_session(title="Session A")
    sess_b = mgr.create_session(title="Session B")

    mgr.add_message(sess_a.session_id, role="user", content="Confidential info in A")
    mgr.add_message(sess_a.session_id, role="assistant", content="Answer in A")

    session_b_fresh = mgr.get_session(sess_b.session_id)
    assert len(session_b_fresh.messages) == 0

    session_a_fresh = mgr.get_session(sess_a.session_id)
    assert len(session_a_fresh.messages) == 2
    assert "Confidential info in A" in session_a_fresh.messages[0].content


def test_clear_session_messages(temp_conv_dir):
    """Verify clearing messages empties history while retaining session configuration."""
    mgr = ConversationManager(persist_dir=temp_conv_dir)
    sess = mgr.create_session(title="Clearing Test", selected_doc_ids=["doc123"])
    mgr.add_message(sess.session_id, role="user", content="Test question")
    mgr.add_message(sess.session_id, role="assistant", content="Test response")

    assert len(mgr.get_session(sess.session_id).messages) == 2

    mgr.clear_session_messages(sess.session_id)
    cleared = mgr.get_session(sess.session_id)
    assert len(cleared.messages) == 0
    assert cleared.selected_doc_ids == ["doc123"]
    assert cleared.title == "Clearing Test"


def test_query_reformulator_empty_history():
    """Verify empty history immediately returns raw query without invoking LLM."""
    reformulator = QueryReformulator()
    standalone, latency = reformulator.reformulate("What is chunk size?", chat_history=[])
    assert standalone == "What is chunk size?"
    assert latency == 0.0


def test_query_reformulator_no_assistant_turns():
    """Verify history without assistant answers returns raw query with zero latency."""
    reformulator = QueryReformulator()
    history = [
        ChatMessage(
            message_id="1",
            session_id="s1",
            role="user",
            content="First user question",
        )
    ]
    standalone, latency = reformulator.reformulate("Follow-up question", chat_history=history)
    assert standalone == "Follow-up question"
    assert latency == 0.0


@patch("src.conversation.reformulator.ChatOllama")
def test_query_reformulator_with_history(mock_chat_cls):
    """Verify follow-up with pronouns is contextualized by LLM."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(
        content="What is the default text chunk size in DocuMind?"
    )

    reformulator = QueryReformulator()
    history = [
        ChatMessage(
            message_id="1",
            session_id="s1",
            role="user",
            content="What are the ingestion settings for DocuMind?",
        ),
        ChatMessage(
            message_id="2",
            session_id="s1",
            role="assistant",
            content="DocuMind uses recursive chunking with 800 character window and 150 overlap.",
        ),
    ]

    standalone, latency = reformulator.reformulate("What is its size?", chat_history=history)
    assert standalone == "What is the default text chunk size in DocuMind?"
    assert latency >= 0.0
    mock_llm.invoke.assert_called_once()


@patch("src.conversation.reformulator.ChatOllama")
def test_query_reformulator_error_fallback(mock_chat_cls):
    """Verify connection error safely falls back to original query without crashing."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.side_effect = ConnectionError("Ollama offline")

    reformulator = QueryReformulator()
    history = [
        ChatMessage(message_id="1", session_id="s1", role="user", content="Q1"),
        ChatMessage(message_id="2", session_id="s1", role="assistant", content="A1"),
    ]

    standalone, latency = reformulator.reformulate("Follow up query", chat_history=history)
    assert standalone == "Follow up query"


def test_conversational_rag_chain_ask(temp_conv_dir):
    """Verify end-to-end conversational turn saves turns and passes reformulated query."""
    mgr = ConversationManager(persist_dir=temp_conv_dir)
    sess = mgr.create_session(title="Multi-turn Test", selected_doc_ids=["doc_handbook"])

    mock_reformulator = MagicMock(spec=QueryReformulator)
    mock_reformulator.reformulate.return_value = ("Standalone query text", 0.15)

    mock_retriever = MagicMock(spec=VectorRetriever)
    mock_chunk = RetrievalResult(
        chunk_id="c0",
        doc_id="doc_handbook",
        filename="handbook.pdf",
        page=1,
        chunk_index=0,
        content="Content text on architecture.",
        distance=0.1,
        score=0.9,
    )
    mock_retriever.retrieve.return_value = [mock_chunk]

    mock_generator = MagicMock(spec=RAGGenerationChain)
    mock_rag_response = RAGResponse(
        answer="Grounded answer based on page 1 [Page 1, handbook.pdf].",
        query="Original user query",
        sources=[mock_chunk],
        citations=[
            Citation(
                filename="handbook.pdf",
                page=1,
                chunk_id="c0",
                doc_id="doc_handbook",
                snippet="Content text",
                is_verified=True,
            )
        ],
        latency_seconds=1.2,
        model_name="llama3.2:3b",
        is_grounded=True,
    )
    mock_generator.generate_from_context.return_value = mock_rag_response

    chain = ConversationalRAGChain(
        conversation_manager=mgr,
        query_reformulator=mock_reformulator,
        retriever=mock_retriever,
        rag_generator=mock_generator,
    )

    # Execute conversational turn
    res = chain.ask(session_id=sess.session_id, query="Original user query", top_k=2)

    # 1. Verify response contents
    assert res.user_query == "Original user query"
    assert res.standalone_query == "Standalone query text"
    assert res.session_id == sess.session_id
    assert res.is_grounded is True
    assert len(res.citations) == 1
    assert res.citations[0].is_verified is True

    # 2. Verify retriever was called with standalone query and document filter
    mock_retriever.retrieve.assert_called_once_with(
        query="Standalone query text",
        top_k=2,
        filter_criteria={"doc_id": "doc_handbook"},
    )

    # 3. Verify session history recorded both turns with original user query unchanged
    session_after = mgr.get_session(sess.session_id)
    assert len(session_after.messages) == 2
    assert session_after.messages[0].role == "user"
    assert session_after.messages[0].content == "Original user query"
    assert session_after.messages[1].role == "assistant"
    assert session_after.messages[1].content == res.answer
    assert session_after.messages[1].standalone_query == "Standalone query text"

"""Session management and persistent conversation history."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.core.exceptions import SessionNotFoundError
from src.core.models import (
    ChatMessage,
    Citation,
    ConversationSession,
    RetrievalResult,
    SessionSummary,
)

DEFAULT_CONVERSATIONS_DIR = "./data/conversations"


class ConversationManager:
    """Manages chat sessions, message histories, and active conversation tracking."""

    def __init__(self, persist_dir: str | Path = DEFAULT_CONVERSATIONS_DIR) -> None:
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, ConversationSession] = {}
        self._active_session_id: str | None = None

        self._load_all_sessions()

    def _load_all_sessions(self) -> None:
        """Load existing session JSON files from the persist directory."""
        session_files = list(self.persist_dir.glob("*.json"))
        loaded_count = 0

        for f in session_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                session = ConversationSession.model_validate(data)
                self._sessions[session.session_id] = session
                loaded_count += 1
            except Exception as e:
                logger.error("Failed to load session file '{}': {}", f.name, e)

        if loaded_count > 0:
            # Set most recently updated session as active
            sorted_sessions = sorted(
                self._sessions.values(), key=lambda s: s.updated_at, reverse=True
            )
            self._active_session_id = sorted_sessions[0].session_id
            logger.info(
                "Loaded {} existing sessions from '{}' (active: {})",
                loaded_count,
                self.persist_dir,
                self._active_session_id,
            )

    def _save_session(self, session: ConversationSession) -> None:
        """Persist a single session to disk atomically as formatted JSON."""
        target_path = self.persist_dir / f"{session.session_id}.json"
        tmp_path = self.persist_dir / f"{session.session_id}.json.tmp"
        try:
            tmp_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            tmp_path.replace(target_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            logger.error("Failed to save session '{}' to disk: {}", session.session_id, e)

    def create_session(
        self,
        title: str = "New Conversation",
        selected_doc_ids: list[str] | None = None,
    ) -> ConversationSession:
        """Create and register a new conversation session."""
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        session = ConversationSession(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            selected_doc_ids=selected_doc_ids or [],
            messages=[],
        )

        self._sessions[session_id] = session
        self._active_session_id = session_id
        self._save_session(session)

        logger.info("Created session '{}' (title: '{}')", session_id, title)
        return session

    def get_session(self, session_id: str) -> ConversationSession:
        """Retrieve a session by its unique session_id."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> list[SessionSummary]:
        """List summary metadata for all sessions, sorted by last updated descending."""
        summaries = [
            SessionSummary(
                session_id=s.session_id,
                title=s.title,
                message_count=len(s.messages),
                created_at=s.created_at,
                updated_at=s.updated_at,
                selected_doc_ids=s.selected_doc_ids,
            )
            for s in self._sessions.values()
        ]
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    @property
    def active_session_id(self) -> str | None:
        """Currently selected active session identifier."""
        return self._active_session_id

    def set_active_session(self, session_id: str) -> None:
        """Set the active session ID."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        self._active_session_id = session_id
        logger.debug("Active session switched to '{}'", session_id)

    def get_active_session(self) -> ConversationSession | None:
        """Retrieve the currently active session object, or None if none exist."""
        if not self._active_session_id or self._active_session_id not in self._sessions:
            return None
        return self._sessions[self._active_session_id]

    def update_session_title(self, session_id: str, new_title: str) -> None:
        """Update session title."""
        session = self.get_session(session_id)
        session.title = new_title.strip() or "Untitled Conversation"
        session.updated_at = datetime.now(UTC).isoformat()
        self._save_session(session)

    def update_session_documents(self, session_id: str, doc_ids: list[str]) -> None:
        """Update the set of selected documents associated with this conversation."""
        session = self.get_session(session_id)
        session.selected_doc_ids = doc_ids
        session.updated_at = datetime.now(UTC).isoformat()
        self._save_session(session)
        logger.info("Updated documents for session '{}': {}", session_id, doc_ids)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        standalone_query: str | None = None,
        citations: list[Citation] | None = None,
        sources: list[RetrievalResult] | None = None,
        latency_seconds: float | None = None,
        is_grounded: bool = True,
    ) -> ChatMessage:
        """Add a new message turn to a session and persist it."""
        session = self.get_session(session_id)
        msg_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        message = ChatMessage(
            message_id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=now,
            standalone_query=standalone_query,
            citations=citations or [],
            sources=sources or [],
            latency_seconds=latency_seconds,
            is_grounded=is_grounded,
        )

        session.messages.append(message)
        session.updated_at = now

        # Auto-generate title from first user query if still default
        if len(session.messages) == 1 and role == "user" and session.title == "New Conversation":
            session.title = content[:40].strip() + ("..." if len(content) > 40 else "")

        self._save_session(session)
        return message

    def clear_session_messages(self, session_id: str) -> None:
        """Clear all messages in a session while preserving session settings."""
        session = self.get_session(session_id)
        session.messages.clear()
        session.updated_at = datetime.now(UTC).isoformat()
        self._save_session(session)
        logger.info("Cleared messages in session '{}'", session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session and remove its file from disk."""
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]

        file_path = self.persist_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()

        if self._active_session_id == session_id:
            # Switch to another available session or None
            remaining = list(self._sessions.keys())
            self._active_session_id = remaining[0] if remaining else None

        logger.info("Deleted session '{}'", session_id)
        return True

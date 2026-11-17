"""Unit tests for DocumentRegistry (SQLite-backed document lifecycle tracking).

All tests use isolated temporary SQLite databases via pytest's ``tmp_path``
fixture — no shared state, no production files touched.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.exceptions import RegistryError
from src.core.models import DocumentRecord, ProcessingStatus
from src.storage.registry import DocumentRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    doc_id: str = "doc-001",
    filename: str = "report.pdf",
    file_hash: str = "aabbcc" * 10 + "1234",
    file_path: str = "/tmp/report.pdf",
    file_size_bytes: int = 4096,
    status: ProcessingStatus = ProcessingStatus.PENDING,
    total_pages: int = 0,
    total_chunks: int = 0,
    error_message: str | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        file_hash=file_hash,
        file_path=file_path,
        file_size_bytes=file_size_bytes,
        total_pages=total_pages,
        total_chunks=total_chunks,
        status=status,
        error_message=error_message,
        upload_timestamp=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_registry_creates_db_file(tmp_path: Path) -> None:
    """Registry creates the SQLite file and parent directories on first use."""
    db_path = tmp_path / "sub" / "dir" / "registry.db"
    assert not db_path.exists()

    reg = DocumentRegistry(db_path=db_path)
    assert db_path.exists()
    assert reg.count() == 0


def test_registry_idempotent_init(tmp_path: Path) -> None:
    """Re-initialising with the same path does not corrupt existing data."""
    db_path = tmp_path / "registry.db"
    reg1 = DocumentRegistry(db_path=db_path)
    reg1.register(_make_record(doc_id="doc-a", file_hash="hash_a" * 10 + "12"))

    reg2 = DocumentRegistry(db_path=db_path)
    assert reg2.count() == 1
    assert reg2.get("doc-a") is not None


# ---------------------------------------------------------------------------
# CRUD — basic operations
# ---------------------------------------------------------------------------


def test_register_and_get(tmp_path: Path) -> None:
    """Registered record can be retrieved by doc_id."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    rec = _make_record(doc_id="doc-001", filename="a.pdf", file_hash="hash001" * 9 + "h1")
    reg.register(rec)

    fetched = reg.get("doc-001")
    assert fetched is not None
    assert fetched.doc_id == "doc-001"
    assert fetched.filename == "a.pdf"
    assert fetched.status == ProcessingStatus.PENDING


def test_get_nonexistent_returns_none(tmp_path: Path) -> None:
    """Getting a doc_id that was never registered returns None."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    assert reg.get("nonexistent-id") is None


def test_get_by_hash(tmp_path: Path) -> None:
    """Record can be retrieved by SHA-256 hash."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    h = "feeddeadbeef" * 5 + "12345678"
    reg.register(_make_record(doc_id="doc-002", file_hash=h))

    fetched = reg.get_by_hash(h)
    assert fetched is not None
    assert fetched.doc_id == "doc-002"


def test_get_by_hash_nonexistent_returns_none(tmp_path: Path) -> None:
    reg = DocumentRegistry(tmp_path / "reg.db")
    assert reg.get_by_hash("0" * 64) is None


def test_delete_existing_record(tmp_path: Path) -> None:
    """Deleting an existing record returns True and removes it."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-del", file_hash="del" * 20 + "1234"))
    assert reg.count() == 1

    result = reg.delete("doc-del")
    assert result is True
    assert reg.count() == 0
    assert reg.get("doc-del") is None


def test_delete_nonexistent_returns_false(tmp_path: Path) -> None:
    """Deleting a doc_id that does not exist returns False without error."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    assert reg.delete("ghost-id") is False


def test_list_all_ordering(tmp_path: Path) -> None:
    """list_all() returns records ordered by upload_timestamp descending."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    for i in range(3):
        reg.register(
            _make_record(
                doc_id=f"doc-{i}",
                file_hash=f"hash{i:02d}" * 10 + "12",
                file_size_bytes=1000 + i,
            )
        )
    records = reg.list_all()
    assert len(records) == 3
    # Check descending timestamp order (timestamps are ISO-8601 strings sortable lexically)
    timestamps = [r.upload_timestamp for r in records]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_by_status(tmp_path: Path) -> None:
    """list_by_status() returns only documents matching that status."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-p", file_hash="pending" * 9 + "12"))
    reg.register(_make_record(doc_id="doc-c", file_hash="complet" * 9 + "12"))
    reg.update_status("doc-c", ProcessingStatus.COMPLETED)

    pending = reg.list_by_status(ProcessingStatus.PENDING)
    completed = reg.list_by_status(ProcessingStatus.COMPLETED)

    assert len(pending) == 1 and pending[0].doc_id == "doc-p"
    assert len(completed) == 1 and completed[0].doc_id == "doc-c"


def test_count(tmp_path: Path) -> None:
    reg = DocumentRegistry(tmp_path / "reg.db")
    assert reg.count() == 0
    reg.register(_make_record(doc_id="d1", file_hash="a" * 64))
    reg.register(_make_record(doc_id="d2", file_hash="b" * 64))
    assert reg.count() == 2


# ---------------------------------------------------------------------------
# Status lifecycle transitions
# ---------------------------------------------------------------------------


def test_status_transitions(tmp_path: Path) -> None:
    """Document progresses through PENDING → PROCESSING → COMPLETED."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-flow", file_hash="fl" * 32))

    reg.mark_processing("doc-flow")
    assert reg.get("doc-flow").status == ProcessingStatus.PROCESSING

    reg.mark_completed("doc-flow", total_pages=5, total_chunks=12)
    record = reg.get("doc-flow")
    assert record.status == ProcessingStatus.COMPLETED
    assert record.total_pages == 5
    assert record.total_chunks == 12
    assert record.completed_timestamp is not None
    assert record.error_message is None


def test_mark_failed_stores_error(tmp_path: Path) -> None:
    """mark_failed() stores error message and sets FAILED status."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-fail", file_hash="ff" * 32))
    reg.mark_processing("doc-fail")
    reg.mark_failed("doc-fail", "Embedding service unavailable")

    record = reg.get("doc-fail")
    assert record.status == ProcessingStatus.FAILED
    assert "Embedding service unavailable" in record.error_message


def test_reset_for_retry(tmp_path: Path) -> None:
    """reset_for_retry() transitions FAILED → PENDING and clears error."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-retry", file_hash="rr" * 32))
    reg.mark_processing("doc-retry")
    reg.mark_failed("doc-retry", "Transient network error")

    reg.reset_for_retry("doc-retry")
    record = reg.get("doc-retry")
    assert record.status == ProcessingStatus.PENDING
    assert record.error_message is None
    assert record.completed_timestamp is None


def test_reset_for_retry_on_non_failed_raises(tmp_path: Path) -> None:
    """reset_for_retry() raises RegistryError if the document is not FAILED."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-ok", file_hash="ok" * 32))
    # Still PENDING — cannot reset

    with pytest.raises(RegistryError, match="not found or not in FAILED status"):
        reg.reset_for_retry("doc-ok")


def test_update_status_nonexistent_raises(tmp_path: Path) -> None:
    """update_status() raises RegistryError for unknown doc_id."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    with pytest.raises(RegistryError, match="not found"):
        reg.update_status("ghost-id", ProcessingStatus.PROCESSING)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_is_duplicate_false_for_unknown_hash(tmp_path: Path) -> None:
    reg = DocumentRegistry(tmp_path / "reg.db")
    assert reg.is_duplicate("0" * 64) is False


def test_is_duplicate_true_after_registration(tmp_path: Path) -> None:
    reg = DocumentRegistry(tmp_path / "reg.db")
    h = "deadbeef" * 8
    reg.register(_make_record(doc_id="doc-dup", file_hash=h))
    assert reg.is_duplicate(h) is True


def test_duplicate_registration_raises(tmp_path: Path) -> None:
    """Registering a record with the same doc_id OR file_hash raises RegistryError."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    h = "cafecafe" * 8
    reg.register(_make_record(doc_id="doc-orig", file_hash=h))

    # Same doc_id — primary key collision
    with pytest.raises(RegistryError):
        reg.register(_make_record(doc_id="doc-orig", file_hash="a" * 64))

    # Same file_hash — unique index collision (different doc_id)
    with pytest.raises(RegistryError):
        reg.register(_make_record(doc_id="doc-other", file_hash=h))


def test_same_filename_different_hash_allowed(tmp_path: Path) -> None:
    """Two documents with the same filename but different content are distinct records."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(
        _make_record(doc_id="doc-v1", filename="report.pdf", file_hash="hash_v1" * 9 + "12")
    )
    reg.register(
        _make_record(doc_id="doc-v2", filename="report.pdf", file_hash="hash_v2" * 9 + "12")
    )
    assert reg.count() == 2
    assert reg.get("doc-v1").filename == reg.get("doc-v2").filename == "report.pdf"
    assert reg.get("doc-v1").doc_id != reg.get("doc-v2").doc_id


def test_same_content_different_filename_is_duplicate(tmp_path: Path) -> None:
    """Same hash registers once; second attempt is detected as duplicate."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    h = "shared_hash_" * 5 + "12"
    reg.register(_make_record(doc_id="doc-a", filename="alpha.pdf", file_hash=h))

    # Same hash → duplicate, even though filename differs
    assert reg.is_duplicate(h) is True
    fetched = reg.get_by_hash(h)
    assert fetched.filename == "alpha.pdf"  # original filename preserved


# ---------------------------------------------------------------------------
# Persistence across instantiation
# ---------------------------------------------------------------------------


def test_persistence_across_restart(tmp_path: Path) -> None:
    """Data survives closing and re-opening the registry."""
    db_path = tmp_path / "registry.db"

    reg1 = DocumentRegistry(db_path)
    reg1.register(_make_record(doc_id="doc-persist", file_hash="pp" * 32))
    reg1.mark_processing("doc-persist")
    reg1.mark_completed("doc-persist", total_pages=3, total_chunks=9)
    del reg1  # simulate restart

    reg2 = DocumentRegistry(db_path)
    record = reg2.get("doc-persist")
    assert record is not None
    assert record.status == ProcessingStatus.COMPLETED
    assert record.total_pages == 3
    assert record.total_chunks == 9
    assert record.completed_timestamp is not None


def test_failed_state_persists_across_restart(tmp_path: Path) -> None:
    """FAILED status and error message survive registry re-open."""
    db_path = tmp_path / "registry.db"

    reg1 = DocumentRegistry(db_path)
    reg1.register(_make_record(doc_id="doc-fail", file_hash="fe" * 32))
    reg1.mark_failed("doc-fail", "ChromaDB connection refused")
    del reg1

    reg2 = DocumentRegistry(db_path)
    record = reg2.get("doc-fail")
    assert record.status == ProcessingStatus.FAILED
    assert "ChromaDB connection refused" in record.error_message


# ---------------------------------------------------------------------------
# Error message truncation
# ---------------------------------------------------------------------------


def test_long_error_message_is_truncated(tmp_path: Path) -> None:
    """Error messages longer than 2000 chars are truncated to prevent DB bloat."""
    reg = DocumentRegistry(tmp_path / "reg.db")
    reg.register(_make_record(doc_id="doc-long", file_hash="lo" * 32))
    long_msg = "x" * 5000
    reg.mark_failed("doc-long", long_msg)

    record = reg.get("doc-long")
    assert len(record.error_message) == 2000

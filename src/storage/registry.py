"""SQLite-backed document registry for persistent document lifecycle tracking.

Provides durable storage of document metadata and processing status across
application restarts. The registry is the single source of truth for which
documents have been ingested, their current status, and their content hashes.

Consistency model
-----------------
The registry (SQLite) and the vector store (ChromaDB) are two independent
storage systems. This module does NOT guarantee atomic transactions across
both. The recovery strategy for partial failures is:

  - PENDING / PROCESSING rows without a corresponding ChromaDB entry:
    safe to retry via ``IngestionPipeline.ingest()``.
  - FAILED rows: always safe to retry; the pipeline resets status to
    PENDING before reattempting.
  - COMPLETED rows whose ChromaDB chunks were deleted externally:
    call ``DocumentRegistry.mark_failed()`` then retry, or use
    ``IngestionPipeline.delete()`` for a clean removal from both stores.

SQLite WAL mode is enabled so concurrent reads do not block writes.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.core.exceptions import RegistryError
from src.core.models import DocumentRecord, ProcessingStatus

DEFAULT_REGISTRY_PATH = "./data/registry.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    file_hash           TEXT NOT NULL,
    file_path           TEXT NOT NULL,
    file_size_bytes     INTEGER NOT NULL DEFAULT 0,
    total_pages         INTEGER NOT NULL DEFAULT 0,
    total_chunks        INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'pending',
    error_message       TEXT,
    upload_timestamp    TEXT NOT NULL,
    completed_timestamp TEXT
);
"""

_CREATE_HASH_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash
    ON documents (file_hash);
"""

_CREATE_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);
"""


class DocumentRegistry:
    """SQLite-backed persistent registry for document metadata and lifecycle status.

    All write operations use ``BEGIN IMMEDIATE`` transactions to serialise
    concurrent writers while still allowing concurrent readers (WAL mode).
    """

    def __init__(self, db_path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("Initialized DocumentRegistry at '{}'", self.db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with WAL mode and row_factory set."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that provides a connection with an active transaction.

        Commits on success, rolls back on any exception, always closes the
        connection.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create schema if it does not already exist."""
        try:
            with self._transaction() as conn:
                conn.execute(_CREATE_TABLE_SQL)
                conn.execute(_CREATE_HASH_INDEX_SQL)
                conn.execute(_CREATE_STATUS_INDEX_SQL)
        except Exception as e:
            raise RegistryError(f"Failed to initialize registry schema: {e}") from e

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            filename=row["filename"],
            file_hash=row["file_hash"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"],
            total_pages=row["total_pages"],
            total_chunks=row["total_chunks"],
            status=ProcessingStatus(row["status"]),
            error_message=row["error_message"],
            upload_timestamp=row["upload_timestamp"],
            completed_timestamp=row["completed_timestamp"],
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register(self, record: DocumentRecord) -> None:
        """Insert a new document record.

        Raises:
            RegistryError: If a record with the same ``doc_id`` or ``file_hash``
                already exists, or on any other database error.
        """
        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO documents
                        (doc_id, filename, file_hash, file_path, file_size_bytes,
                         total_pages, total_chunks, status, error_message,
                         upload_timestamp, completed_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.doc_id,
                        record.filename,
                        record.file_hash,
                        record.file_path,
                        record.file_size_bytes,
                        record.total_pages,
                        record.total_chunks,
                        record.status.value,
                        record.error_message,
                        record.upload_timestamp,
                        record.completed_timestamp,
                    ),
                )
            logger.debug("Registered doc_id='{}' filename='{}'", record.doc_id, record.filename)
        except sqlite3.IntegrityError as e:
            raise RegistryError(
                f"Document already exists (doc_id='{record.doc_id}', "
                f"file_hash='{record.file_hash}'): {e}"
            ) from e
        except Exception as e:
            raise RegistryError(f"Failed to register document '{record.doc_id}': {e}") from e

    def update_status(
        self,
        doc_id: str,
        status: ProcessingStatus,
        *,
        error_message: str | None = None,
        total_pages: int | None = None,
        total_chunks: int | None = None,
        completed_timestamp: str | None = None,
    ) -> None:
        """Update lifecycle status and optional metadata fields for a document.

        Only columns provided as non-None keyword arguments are updated.
        """
        updates: list[str] = ["status = ?"]
        values: list[object] = [status.value]

        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)
        if total_pages is not None:
            updates.append("total_pages = ?")
            values.append(total_pages)
        if total_chunks is not None:
            updates.append("total_chunks = ?")
            values.append(total_chunks)
        if completed_timestamp is not None:
            updates.append("completed_timestamp = ?")
            values.append(completed_timestamp)

        values.append(doc_id)
        sql = f"UPDATE documents SET {', '.join(updates)} WHERE doc_id = ?"  # noqa: S608

        try:
            with self._transaction() as conn:
                cursor = conn.execute(sql, values)
                if cursor.rowcount == 0:
                    raise RegistryError(f"Document not found for status update: '{doc_id}'")
            logger.debug("Updated doc_id='{}' status={}", doc_id, status.value)
        except RegistryError:
            raise
        except Exception as e:
            raise RegistryError(f"Failed to update status for '{doc_id}': {e}") from e

    def mark_processing(self, doc_id: str) -> None:
        """Transition a document to PROCESSING status."""
        self.update_status(doc_id, ProcessingStatus.PROCESSING, error_message=None)

    def mark_completed(self, doc_id: str, total_pages: int, total_chunks: int) -> None:
        """Transition a document to COMPLETED status and record final metrics."""
        self.update_status(
            doc_id,
            ProcessingStatus.COMPLETED,
            error_message=None,
            total_pages=total_pages,
            total_chunks=total_chunks,
            completed_timestamp=datetime.now(UTC).isoformat(),
        )

    def mark_failed(self, doc_id: str, error_message: str) -> None:
        """Transition a document to FAILED status with an error description."""
        self.update_status(
            doc_id,
            ProcessingStatus.FAILED,
            error_message=error_message[:2000],  # guard against very long tracebacks
        )

    def reset_for_retry(self, doc_id: str) -> None:
        """Reset a FAILED document back to PENDING so it can be retried.

        Clears the stored error message and resets timestamps so a fresh
        ingestion attempt starts with a clean slate.
        """
        try:
            with self._transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE documents
                    SET status = ?, error_message = NULL, completed_timestamp = NULL
                    WHERE doc_id = ? AND status = ?
                    """,
                    (ProcessingStatus.PENDING.value, doc_id, ProcessingStatus.FAILED.value),
                )
                if cursor.rowcount == 0:
                    raise RegistryError(
                        f"Cannot reset doc_id='{doc_id}': not found or not in FAILED status."
                    )
            logger.info("Reset doc_id='{}' to PENDING for retry", doc_id)
        except RegistryError:
            raise
        except Exception as e:
            raise RegistryError(f"Failed to reset '{doc_id}' for retry: {e}") from e

    def delete(self, doc_id: str) -> bool:
        """Remove a document record from the registry.

        Returns:
            True if the record was deleted, False if it did not exist.
        """
        try:
            with self._transaction() as conn:
                cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Deleted registry record for doc_id='{}'", doc_id)
            return deleted
        except Exception as e:
            raise RegistryError(f"Failed to delete document '{doc_id}': {e}") from e

    # ------------------------------------------------------------------
    # Read operations (no transaction needed — WAL allows concurrent reads)
    # ------------------------------------------------------------------

    def get(self, doc_id: str) -> DocumentRecord | None:
        """Retrieve a document record by its ``doc_id``, or None if not found."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
            finally:
                conn.close()
            return self._row_to_record(row) if row else None
        except Exception as e:
            raise RegistryError(f"Failed to retrieve document '{doc_id}': {e}") from e

    def get_by_hash(self, file_hash: str) -> DocumentRecord | None:
        """Retrieve a document record by its SHA-256 ``file_hash``, or None if not found."""
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
                ).fetchone()
            finally:
                conn.close()
            return self._row_to_record(row) if row else None
        except Exception as e:
            raise RegistryError(f"Failed to retrieve by hash '{file_hash}': {e}") from e

    def list_all(self) -> list[DocumentRecord]:
        """Return all document records ordered by upload timestamp descending."""
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY upload_timestamp DESC"
                ).fetchall()
            finally:
                conn.close()
            return [self._row_to_record(r) for r in rows]
        except Exception as e:
            raise RegistryError(f"Failed to list documents: {e}") from e

    def list_by_status(self, status: ProcessingStatus) -> list[DocumentRecord]:
        """Return all document records with the given processing status."""
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE status = ? ORDER BY upload_timestamp DESC",
                    (status.value,),
                ).fetchall()
            finally:
                conn.close()
            return [self._row_to_record(r) for r in rows]
        except Exception as e:
            raise RegistryError(f"Failed to list by status '{status.value}': {e}") from e

    def is_duplicate(self, file_hash: str) -> bool:
        """Return True if any document with this hash is already registered."""
        return self.get_by_hash(file_hash) is not None

    def count(self) -> int:
        """Return total number of documents in the registry."""
        try:
            conn = self._connect()
            try:
                result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            finally:
                conn.close()
            return result[0] if result else 0
        except Exception as e:
            raise RegistryError(f"Failed to count documents: {e}") from e

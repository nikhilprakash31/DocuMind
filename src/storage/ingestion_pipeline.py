"""Orchestrated document ingestion pipeline with registry-backed lifecycle tracking.

This module provides a single high-level entry point (``IngestionPipeline``) that
coordinates all ingestion steps:

    register (PENDING) → transition (PROCESSING)
                        └→ validate → load → split → embed → upsert → complete (COMPLETED)
                        └→ on error (FAILED)

Lifecycle guarantees
--------------------
Each document is tracked in SQLite from the very beginning of the pipeline:

    PENDING → PROCESSING → COMPLETED
                        └→ FAILED

1. Step 1 (Hash & Pre-check): Computes deterministic SHA-256 and doc_id.
2. Step 2 (Register / Reset PENDING): If new, an initial PENDING record is immediately
   persisted to SQLite before any heavy processing. If retrying a FAILED or interrupted
   PROCESSING document, status is reset to PENDING. If already COMPLETED, returns existing.
3. Step 3 (Transition to PROCESSING): Status updated in SQLite so crashes in subsequent
   steps are detectable.
4. Step 4 (Processing): Validates, loads, chunks, embeds, and upserts into ChromaDB.
   Any exception transitions SQLite status to FAILED and stores the error message.
5. Step 5 (Transition to COMPLETED): On success, records total pages, total chunks,
   completion timestamp, and transitions to COMPLETED.

Partial-failure recovery
-------------------------
Because SQLite and ChromaDB are independent stores, cross-store atomicity is impossible.
The recovery strategy is:
  - If embedding, chunking, or ChromaDB upsert fails: document is marked FAILED in SQLite.
    ChromaDB chunks (if partially written) are overwritten idempotently on retry
    because chunk IDs are deterministic ({doc_id}_c{chunk_index}).
  - If a crash/kill occurs while PROCESSING: on restart, the record remains in PROCESSING.
    Calling ingest() or reconcile_interrupted() resets it to PENDING and retries cleanly.
  - If deletion partially fails (e.g., ChromaDB succeeded but SQLite failed):
    calling delete() again safely retries (ChromaDB deletion is idempotent) and cleans SQLite.
  - reconcile_orphans() provides proactive detection and cleanup for orphaned records
    or chunks in either store.

Duplicate handling
------------------
Duplicates are detected by SHA-256 file hash via the registry.

- **Same hash, any filename**: returns the existing ``DocumentRecord`` without
  re-embedding. The existing doc_id is stable.
- **Same filename, different hash**: treated as a distinct document with its
  own doc_id. Both records coexist in the registry and vector store.
"""

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.core.exceptions import (
    IngestionError,
    RegistryError,
    VectorStoreError,
)
from src.core.models import DocumentRecord, IngestionSummary, ProcessingStatus
from src.ingestion.loader import PDFDocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.storage.registry import DocumentRegistry
from src.storage.vector_store import ChromaVectorStore
from src.utils.hashing import calculate_sha256, generate_doc_id


class IngestionPipeline:
    """Orchestrates the full document ingestion lifecycle with registry persistence.

    Args:
        registry:     ``DocumentRegistry`` instance for lifecycle tracking.
        vector_store: ``ChromaVectorStore`` instance for chunk storage.
        loader:       ``PDFDocumentLoader`` for validation and text extraction.
        splitter:     ``DocumentSplitter`` for text chunking.
    """

    def __init__(
        self,
        registry: DocumentRegistry,
        vector_store: ChromaVectorStore,
        loader: PDFDocumentLoader | None = None,
        splitter: DocumentSplitter | None = None,
    ) -> None:
        self.registry = registry
        self.vector_store = vector_store
        self.loader = loader or PDFDocumentLoader()
        self.splitter = splitter or DocumentSplitter()

    def _unregister_from_loader(self, file_hash: str) -> None:
        """Clear hash from loader's in-memory detector if present."""
        if hasattr(self.loader, "duplicate_detector") and hasattr(
            self.loader.duplicate_detector, "unregister"
        ):
            self.loader.duplicate_detector.unregister(file_hash)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str | Path) -> DocumentRecord:
        """Ingest a PDF file through the full pipeline.

        Steps:
            1. Compute SHA-256 hash of the file (pre-load, cheap).
            2. Check registry for an existing record with the same hash.
               - COMPLETED → return existing record (duplicate; no re-embedding).
               - PENDING / FAILED → reset to PENDING and retry.
               - PROCESSING → indicates a prior interrupted run; reset and retry.
               - Not found → create new PENDING record.
            3. Transition to PROCESSING.
            4. Validate, load, split, embed, upsert.
            5. Mark COMPLETED on success or FAILED on error.

        Returns:
            The final ``DocumentRecord`` for the ingested document.

        Raises:
            DuplicateDocumentError: Never — duplicates return the existing record.
            IngestionError:         On unrecoverable pipeline failure.
            RegistryError:          On registry I/O failure.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise IngestionError(f"File not found: '{path}'")

        # --- Step 1: Hash first (fast, before anything else) ---
        try:
            file_hash = calculate_sha256(path)
            doc_id = generate_doc_id(file_hash)
        except Exception as e:
            raise IngestionError(f"Failed to hash file '{path.name}': {e}") from e

        # --- Step 2: Duplicate / recovery check ---
        existing = self.registry.get_by_hash(file_hash)
        if existing is not None:
            if existing.status == ProcessingStatus.COMPLETED:
                logger.info(
                    "Duplicate detected: '{}' matches existing doc_id='{}' (COMPLETED). "
                    "Returning existing record.",
                    path.name,
                    existing.doc_id,
                )
                return existing

            # Interrupted or failed run — reset and retry
            logger.info(
                "Found existing record doc_id='{}' with status='{}'. Resetting for retry.",
                existing.doc_id,
                existing.status.value,
            )
            if existing.status == ProcessingStatus.FAILED:
                self.registry.reset_for_retry(existing.doc_id)
            else:
                # PENDING or PROCESSING — overwrite to PENDING cleanly
                self.registry.update_status(existing.doc_id, ProcessingStatus.PENDING)

        else:
            # Brand-new document — create registry record
            stat = path.stat()
            record = DocumentRecord(
                doc_id=doc_id,
                filename=path.name,
                file_hash=file_hash,
                file_path=str(path),
                file_size_bytes=stat.st_size,
                status=ProcessingStatus.PENDING,
                upload_timestamp=datetime.now(UTC).isoformat(),
            )
            self.registry.register(record)

        # --- Step 3: Transition to PROCESSING ---
        self.registry.mark_processing(doc_id)

        # --- Step 4: Validate → load → split → embed → upsert ---
        try:
            # Ensure in-memory loader detector does not block valid retries
            self._unregister_from_loader(file_hash)

            # Validation and text extraction
            # Pass raise_on_duplicate=False because dedup is handled by the registry
            pages = self.loader.load(path, raise_on_duplicate=False)
            if not pages:
                raise IngestionError(
                    f"Loader returned no pages for '{path.name}'. "
                    "File may be blank or extraction failed."
                )

            total_pages = len(pages)

            # Text splitting
            chunks = self.splitter.split_documents(pages)
            total_chunks = len(chunks)

            if total_chunks == 0:
                raise IngestionError(
                    f"Splitter produced zero chunks for '{path.name}'. "
                    "File may contain only whitespace."
                )

            # Embed and upsert (ChromaDB upsert is idempotent on chunk_id)
            self.vector_store.upsert_chunks(chunks)

        except (IngestionError, VectorStoreError) as e:
            self._unregister_from_loader(file_hash)
            self.registry.mark_failed(doc_id, str(e))
            raise
        except Exception as e:
            self._unregister_from_loader(file_hash)
            err_msg = f"Pipeline error for '{path.name}': {e}"
            self.registry.mark_failed(doc_id, err_msg)
            raise IngestionError(err_msg) from e

        # --- Step 5: Mark COMPLETED ---
        self.registry.mark_completed(doc_id, total_pages=total_pages, total_chunks=total_chunks)

        final = self.registry.get(doc_id)
        if final is None:
            raise RegistryError(f"Registry record vanished after completion for '{doc_id}'.")

        logger.info(
            "Ingested '{}' → doc_id='{}' ({} pages, {} chunks)",
            path.name,
            doc_id,
            total_pages,
            total_chunks,
        )
        return final

    def delete(self, doc_id: str) -> bool:
        """Delete a document from both the registry and vector store.

        Deletion order:
            1. Remove chunks from ChromaDB.
            2. Remove record from registry.

        This order ensures that a crash between steps leaves at most an orphan
        registry record (safe — the record can be manually deleted or will be
        treated as a stale COMPLETED entry). The alternative (registry first)
        would leave orphan chunks with no registry record and no way to clean
        them up by doc_id.

        Only chunks belonging to ``doc_id`` are removed. Unrelated documents
        are never touched.

        Args:
            doc_id: The document identifier to delete.

        Returns:
            True if the document existed in the registry and was deleted.
            False if no registry record was found (vector store is still cleaned
            up defensively).

        Raises:
            VectorStoreError: If ChromaDB deletion fails.
            RegistryError:    If registry deletion fails after vector store succeeded.
        """
        # Step 1: Remove from vector store (idempotent — returns 0 if not present)
        try:
            record = self.registry.get(doc_id)
            if record:
                self._unregister_from_loader(record.file_hash)

            chunks_deleted = self.vector_store.delete_document(doc_id)
            logger.info(
                "Removed {} chunk(s) from vector store for doc_id='{}'", chunks_deleted, doc_id
            )
        except VectorStoreError as e:
            logger.error("Vector store deletion failed for doc_id='{}': {}", doc_id, e)
            raise

        # Step 2: Remove from registry
        try:
            existed = self.registry.delete(doc_id)
        except RegistryError as e:
            logger.error(
                "Registry deletion failed for doc_id='{}' AFTER vector store removal: {}. "
                "Orphan registry record may remain.",
                doc_id,
                e,
            )
            raise

        return existed

    def get_record(self, doc_id: str) -> DocumentRecord | None:
        """Retrieve the current registry record for a document, or None."""
        return self.registry.get(doc_id)

    def list_documents(self) -> list[DocumentRecord]:
        """Return all registry records ordered by upload timestamp descending."""
        return self.registry.list_all()

    def list_failed(self) -> list[DocumentRecord]:
        """Return all documents currently in FAILED status."""
        return self.registry.list_by_status(ProcessingStatus.FAILED)

    def get_ingestion_summary(self, doc_id: str) -> IngestionSummary | None:
        """Return an ``IngestionSummary`` for a completed document, or None."""
        record = self.registry.get(doc_id)
        if record is None or record.status != ProcessingStatus.COMPLETED:
            return None
        return IngestionSummary(
            filename=record.filename,
            doc_id=record.doc_id,
            doc_hash=record.file_hash,
            total_pages=record.total_pages,
            total_characters=0,  # not stored at pipeline level; available via chunk metadata
            file_size_bytes=record.file_size_bytes,
            upload_timestamp=record.upload_timestamp,
        )

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    def retry_failed(self, doc_id: str) -> DocumentRecord:
        """Retry ingestion for a document that is currently in FAILED status.

        Raises:
            IngestionError: If the document is not in FAILED status.
        """
        record = self.registry.get(doc_id)
        if record is None:
            raise IngestionError(f"Document not found in registry: '{doc_id}'")
        if record.status != ProcessingStatus.FAILED:
            raise IngestionError(
                f"Cannot retry doc_id='{doc_id}': current status is '{record.status.value}', "
                "expected 'failed'."
            )
        logger.info("Retrying failed ingestion for doc_id='{}'", doc_id)
        return self.ingest(record.file_path)

    def reconcile_interrupted(self) -> list[DocumentRecord]:
        """Find and recover any documents left in PENDING or PROCESSING state from a crash.

        Marks them as FAILED so that they can be inspected, or retried via retry_failed().
        Returns the list of transitioned records.
        """
        recovered: list[DocumentRecord] = []
        for status in (ProcessingStatus.PENDING, ProcessingStatus.PROCESSING):
            for rec in self.registry.list_by_status(status):
                msg = f"Process interrupted while in {status.value} state (recovered on restart)."
                self.registry.mark_failed(rec.doc_id, msg)
                updated = self.registry.get(rec.doc_id)
                if updated:
                    recovered.append(updated)
                logger.warning("Reconciled interrupted doc_id='{}' (was {})", rec.doc_id, status)
        return recovered

    def purge_orphaned_chunks(self) -> int:
        """Remove vector store chunks whose parent doc_id is missing from the SQLite registry.

        This handles the rare recovery scenario where SQLite records were deleted or lost
        out-of-band while vector chunks remained in ChromaDB.

        Returns:
            The number of orphaned chunks deleted.
        """
        total_purged = 0
        try:
            # Query all metadata in Chroma collection
            col = self.vector_store.collection
            all_records = col.get(include=["metadatas"])
            all_ids = all_records.get("ids", [])
            all_metas = all_records.get("metadatas", [])

            # Collect unique doc_ids from ChromaDB
            chroma_doc_ids: set[str] = set()
            chunk_to_doc: dict[str, str] = {}
            for chunk_id, meta in zip(all_ids, all_metas, strict=False):
                if meta and "doc_id" in meta:
                    did = str(meta["doc_id"])
                    chroma_doc_ids.add(did)
                    chunk_to_doc[chunk_id] = did

            # Identify which doc_ids are absent from SQLite registry or FAILED
            valid_completed_doc_ids = {
                r.doc_id for r in self.registry.list_by_status(ProcessingStatus.COMPLETED)
            }
            orphaned_doc_ids = chroma_doc_ids - valid_completed_doc_ids

            for orphan_id in orphaned_doc_ids:
                deleted = self.vector_store.delete_document(orphan_id)
                total_purged += deleted
                logger.info("Purged {} orphaned chunk(s) for doc_id='{}'", deleted, orphan_id)

        except Exception as e:
            logger.error("Failed to purge orphaned chunks from vector store: {}", e)
            raise VectorStoreError(f"Purging orphaned chunks failed: {e}") from e

        return total_purged

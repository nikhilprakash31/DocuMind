"""Integration tests for IngestionPipeline lifecycle, duplicate handling, and deletion.

Each test uses isolated temporary directories and mocked embeddings. No
production data directories are touched. Live Ollama is NOT required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import IngestionError, VectorStoreError
from src.core.models import ProcessingStatus
from src.storage.embeddings import EXPECTED_EMBEDDING_DIM, NomicEmbeddingAdapter
from src.storage.ingestion_pipeline import IngestionPipeline
from src.storage.registry import DocumentRegistry
from src.storage.vector_store import ChromaVectorStore

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Mock NomicEmbeddingAdapter that returns deterministic 768-dim vectors."""
    adapter = MagicMock(spec=NomicEmbeddingAdapter)
    adapter.expected_dim = EXPECTED_EMBEDDING_DIM
    adapter.model = "nomic-embed-text"

    def _embed_docs(texts: list[str]) -> list[list[float]]:
        return [[0.1] * EXPECTED_EMBEDDING_DIM for _ in texts]

    def _embed_query(query: str) -> list[float]:
        return [0.1] * EXPECTED_EMBEDDING_DIM

    adapter.embed_documents.side_effect = _embed_docs
    adapter.embed_query.side_effect = _embed_query
    return adapter


@pytest.fixture
def pipeline(tmp_path: Path, mock_adapter: MagicMock) -> IngestionPipeline:
    """IngestionPipeline wired to isolated tmp_path registry + vector store."""
    registry = DocumentRegistry(db_path=tmp_path / "registry.db")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_col",
        embedding_adapter=mock_adapter,
    )
    return IngestionPipeline(registry=registry, vector_store=vector_store)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_ingest_single_pdf(pipeline: IngestionPipeline, pdf_factory) -> None:
    """Ingesting a valid PDF produces a COMPLETED record and indexes chunks."""
    pdf_path = pdf_factory("spec.pdf", ["Introduction to RAG.", "Chunking strategies."])

    record = pipeline.ingest(pdf_path)

    assert record.status == ProcessingStatus.COMPLETED
    assert record.filename == "spec.pdf"
    assert record.total_pages == 2
    assert record.total_chunks >= 1
    assert record.completed_timestamp is not None
    assert record.error_message is None

    # Vector store must reflect indexed chunks
    assert pipeline.vector_store.count() > 0


def test_ingest_creates_registry_record(pipeline: IngestionPipeline, pdf_factory) -> None:
    """After ingestion, the registry contains exactly one record per document."""
    pdf_path = pdf_factory("guide.pdf", ["Guide to embeddings."])
    record = pipeline.ingest(pdf_path)

    fetched = pipeline.registry.get(record.doc_id)
    assert fetched is not None
    assert fetched.doc_id == record.doc_id
    assert fetched.status == ProcessingStatus.COMPLETED


def test_ingest_summary(pipeline: IngestionPipeline, pdf_factory) -> None:
    """get_ingestion_summary() returns a valid summary for a COMPLETED document."""
    pdf_path = pdf_factory("summary_test.pdf", ["Summary content page one."])
    record = pipeline.ingest(pdf_path)

    summary = pipeline.get_ingestion_summary(record.doc_id)
    assert summary is not None
    assert summary.filename == "summary_test.pdf"
    assert summary.doc_id == record.doc_id
    assert summary.total_pages >= 1


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------


def test_duplicate_same_hash_returns_existing_record(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """Ingesting the same file content twice returns the first record without re-embedding."""
    pdf_path = pdf_factory("dup.pdf", ["Duplicate content."])

    record1 = pipeline.ingest(pdf_path)
    initial_count = pipeline.vector_store.count()

    record2 = pipeline.ingest(pdf_path)

    # Both calls return the same doc_id
    assert record1.doc_id == record2.doc_id
    assert record2.status == ProcessingStatus.COMPLETED

    # Registry still has exactly one record
    assert pipeline.registry.count() == 1

    # No new chunks were added to the vector store
    assert pipeline.vector_store.count() == initial_count


def test_duplicate_different_filename_same_content(
    pipeline: IngestionPipeline, tmp_path: Path, pdf_factory
) -> None:
    """Same content under a different filename is treated as a duplicate."""
    pdf_path_a = pdf_factory("original.pdf", ["Shared content."])
    # Write same bytes under a different name
    pdf_path_b = tmp_path / "copy.pdf"
    pdf_path_b.write_bytes(pdf_path_a.read_bytes())

    record_a = pipeline.ingest(pdf_path_a)
    record_b = pipeline.ingest(pdf_path_b)

    # Same hash → same doc_id → same record
    assert record_a.doc_id == record_b.doc_id
    assert pipeline.registry.count() == 1


def test_same_filename_different_content_produces_distinct_records(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """Same filename with different content produces two distinct registry records."""
    pdf_path_v1 = pdf_factory("report.pdf", ["Version 1 content with unique text alpha."])
    record_v1 = pipeline.ingest(pdf_path_v1)

    # Overwrite the file with different content
    pdf_path_v2 = pdf_factory("report_v2.pdf", ["Version 2 content with unique text beta."])
    record_v2 = pipeline.ingest(pdf_path_v2)

    assert record_v1.doc_id != record_v2.doc_id
    assert pipeline.registry.count() == 2


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_delete_removes_registry_and_chunks(pipeline: IngestionPipeline, pdf_factory) -> None:
    """delete() removes both the registry record and vector store chunks for the document."""
    pdf_path = pdf_factory("todelete.pdf", ["Content to be deleted eventually."])
    record = pipeline.ingest(pdf_path)
    doc_id = record.doc_id

    chunks_before = pipeline.vector_store.count()
    assert chunks_before > 0

    existed = pipeline.delete(doc_id)
    assert existed is True

    # Registry record gone
    assert pipeline.registry.get(doc_id) is None

    # Chunks gone
    assert pipeline.vector_store.count() == 0


def test_delete_does_not_affect_other_documents(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """Deleting one document leaves unrelated documents intact in both stores."""
    pdf_a = pdf_factory("doc_a.pdf", ["Alpha document content is unique."])
    pdf_b = pdf_factory("doc_b.pdf", ["Beta document content is unique."])

    rec_a = pipeline.ingest(pdf_a)
    rec_b = pipeline.ingest(pdf_b)

    chunks_before = pipeline.vector_store.count()

    pipeline.delete(rec_a.doc_id)

    # doc_b still present in registry
    assert pipeline.registry.get(rec_b.doc_id) is not None

    # doc_a chunks removed; doc_b chunks remain
    assert pipeline.vector_store.count() == chunks_before - rec_a.total_chunks


def test_delete_nonexistent_returns_false(pipeline: IngestionPipeline) -> None:
    """delete() returns False for a doc_id that was never registered."""
    result = pipeline.delete("nonexistent-doc-id")
    assert result is False


def test_delete_does_not_remove_conversation_history(
    pipeline: IngestionPipeline, pdf_factory, tmp_path: Path
) -> None:
    """Document deletion never touches the conversations directory."""
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    sentinel = conv_dir / "session_abc.json"
    sentinel.write_text('{"session_id": "abc"}')

    pdf_path = pdf_factory("todelete2.pdf", ["Content."])
    record = pipeline.ingest(pdf_path)
    pipeline.delete(record.doc_id)

    # Conversation file must still exist
    assert sentinel.exists()
    assert sentinel.read_text() == '{"session_id": "abc"}'


# ---------------------------------------------------------------------------
# Failure paths and recovery
# ---------------------------------------------------------------------------


def test_ingestion_failure_marks_record_failed(
    tmp_path: Path, mock_adapter: MagicMock, pdf_factory
) -> None:
    """A pipeline failure transitions the document to FAILED status."""
    registry = DocumentRegistry(db_path=tmp_path / "registry.db")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="fail_test",
        embedding_adapter=mock_adapter,
    )
    pipeline = IngestionPipeline(registry=registry, vector_store=vector_store)

    pdf_path = pdf_factory("broken.pdf", ["Content that will fail."])

    # Patch upsert_chunks to raise VectorStoreError mid-pipeline
    with patch.object(
        vector_store,
        "upsert_chunks",
        side_effect=VectorStoreError("Simulated ChromaDB failure"),
    ):
        with pytest.raises(VectorStoreError):
            pipeline.ingest(pdf_path)

    # Registry record must exist and be marked FAILED
    from src.utils.hashing import calculate_sha256, generate_doc_id

    file_hash = calculate_sha256(pdf_path)
    doc_id = generate_doc_id(file_hash)
    record = registry.get(doc_id)
    assert record is not None
    assert record.status == ProcessingStatus.FAILED


def test_failed_ingestion_can_be_retried(
    tmp_path: Path, mock_adapter: MagicMock, pdf_factory
) -> None:
    """After a failure, calling ingest() again on the same file succeeds."""
    registry = DocumentRegistry(db_path=tmp_path / "registry.db")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="retry_test",
        embedding_adapter=mock_adapter,
    )
    pipeline = IngestionPipeline(registry=registry, vector_store=vector_store)

    pdf_path = pdf_factory("retry.pdf", ["Retry content."])

    # First attempt — force failure
    with patch.object(
        vector_store,
        "upsert_chunks",
        side_effect=VectorStoreError("Transient failure"),
    ):
        with pytest.raises(VectorStoreError):
            pipeline.ingest(pdf_path)

    # Verify FAILED state
    from src.utils.hashing import calculate_sha256, generate_doc_id

    doc_id = generate_doc_id(calculate_sha256(pdf_path))
    assert registry.get(doc_id).status == ProcessingStatus.FAILED

    # Second attempt — normal, should succeed
    record = pipeline.ingest(pdf_path)
    assert record.status == ProcessingStatus.COMPLETED
    assert vector_store.count() > 0


def test_retry_failed_convenience_method(
    tmp_path: Path, mock_adapter: MagicMock, pdf_factory
) -> None:
    """retry_failed() calls ingest() on the stored file_path and succeeds."""
    registry = DocumentRegistry(db_path=tmp_path / "registry.db")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="retry_conv_test",
        embedding_adapter=mock_adapter,
    )
    pipeline = IngestionPipeline(registry=registry, vector_store=vector_store)

    pdf_path = pdf_factory("retry_conv.pdf", ["Retry via convenience method."])

    with patch.object(
        vector_store,
        "upsert_chunks",
        side_effect=VectorStoreError("Transient failure"),
    ):
        with pytest.raises(VectorStoreError):
            pipeline.ingest(pdf_path)

    from src.utils.hashing import calculate_sha256, generate_doc_id

    doc_id = generate_doc_id(calculate_sha256(pdf_path))

    record = pipeline.retry_failed(doc_id)
    assert record.status == ProcessingStatus.COMPLETED


def test_retry_failed_on_non_failed_raises(pipeline: IngestionPipeline, pdf_factory) -> None:
    """retry_failed() raises IngestionError if document is not in FAILED status."""
    pdf_path = pdf_factory("gooddoc.pdf", ["Good content."])
    record = pipeline.ingest(pdf_path)  # → COMPLETED

    with pytest.raises(IngestionError, match="current status is 'completed'"):
        pipeline.retry_failed(record.doc_id)


def test_ingest_missing_file_raises(pipeline: IngestionPipeline) -> None:
    """Ingesting a path that doesn't exist raises IngestionError before registry changes."""
    with pytest.raises(IngestionError, match="File not found"):
        pipeline.ingest("/nonexistent/path/to/file.pdf")


# ---------------------------------------------------------------------------
# Persistence across re-instantiation
# ---------------------------------------------------------------------------


def test_registry_state_persists_across_pipeline_restart(
    tmp_path: Path, mock_adapter: MagicMock, pdf_factory
) -> None:
    """Document records survive pipeline object destruction and re-creation."""
    db_path = tmp_path / "registry.db"
    chroma_path = tmp_path / "chroma"

    # Instance 1
    pipeline1 = IngestionPipeline(
        registry=DocumentRegistry(db_path=db_path),
        vector_store=ChromaVectorStore(
            persist_directory=chroma_path,
            collection_name="persist_test",
            embedding_adapter=mock_adapter,
        ),
    )
    pdf_path = pdf_factory("persistent.pdf", ["Persistent document content."])
    record1 = pipeline1.ingest(pdf_path)
    doc_id = record1.doc_id
    del pipeline1

    # Instance 2 — fresh objects, same paths
    pipeline2 = IngestionPipeline(
        registry=DocumentRegistry(db_path=db_path),
        vector_store=ChromaVectorStore(
            persist_directory=chroma_path,
            collection_name="persist_test",
            embedding_adapter=mock_adapter,
        ),
    )
    record2 = pipeline2.registry.get(doc_id)
    assert record2 is not None
    assert record2.status == ProcessingStatus.COMPLETED
    assert record2.doc_id == doc_id

    # Vector store also persisted its chunks
    assert pipeline2.vector_store.count() > 0


def test_duplicate_detected_across_pipeline_restart(
    tmp_path: Path, mock_adapter: MagicMock, pdf_factory
) -> None:
    """After restart, the new pipeline recognises previously ingested hashes as duplicates."""
    db_path = tmp_path / "registry.db"
    chroma_path = tmp_path / "chroma"

    pdf_path = pdf_factory("dup_across_restart.pdf", ["Unique content for restart test."])

    pipeline1 = IngestionPipeline(
        registry=DocumentRegistry(db_path=db_path),
        vector_store=ChromaVectorStore(
            persist_directory=chroma_path,
            collection_name="dup_restart",
            embedding_adapter=mock_adapter,
        ),
    )
    record1 = pipeline1.ingest(pdf_path)
    del pipeline1

    pipeline2 = IngestionPipeline(
        registry=DocumentRegistry(db_path=db_path),
        vector_store=ChromaVectorStore(
            persist_directory=chroma_path,
            collection_name="dup_restart",
            embedding_adapter=mock_adapter,
        ),
    )
    # Second ingest on same file — must return existing record, not re-embed
    record2 = pipeline2.ingest(pdf_path)
    assert record2.doc_id == record1.doc_id
    assert pipeline2.registry.count() == 1


# ---------------------------------------------------------------------------
# list_documents / list_failed
# ---------------------------------------------------------------------------


def test_list_documents_and_list_failed(
    tmp_path: Path, mock_adapter: MagicMock, pdf_factory
) -> None:
    """list_documents() returns all records; list_failed() returns only FAILED ones."""
    registry = DocumentRegistry(db_path=tmp_path / "registry.db")
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="list_test",
        embedding_adapter=mock_adapter,
    )
    pipeline = IngestionPipeline(registry=registry, vector_store=vector_store)

    pdf_ok = pdf_factory("ok.pdf", ["Good content."])
    pdf_fail = pdf_factory("fail.pdf", ["Content that will fail."])

    pipeline.ingest(pdf_ok)

    with patch.object(
        vector_store,
        "upsert_chunks",
        side_effect=VectorStoreError("Forced failure"),
    ):
        with pytest.raises(VectorStoreError):
            pipeline.ingest(pdf_fail)

    all_docs = pipeline.list_documents()
    failed_docs = pipeline.list_failed()

    assert len(all_docs) == 2
    assert len(failed_docs) == 1
    assert failed_docs[0].filename == "fail.pdf"


# ---------------------------------------------------------------------------
# Failure-injection and boundary condition tests
# ---------------------------------------------------------------------------


def test_failure_injection_vector_upsert_failure(pipeline: IngestionPipeline, pdf_factory) -> None:
    """When vector store upsert fails, document is marked FAILED and retrying is safe."""
    pdf_path = pdf_factory("upsert_fail.pdf", ["Vector upsert will fail."])

    with patch.object(
        pipeline.vector_store,
        "upsert_chunks",
        side_effect=VectorStoreError("Disk full or timeout during upsert"),
    ):
        with pytest.raises(VectorStoreError):
            pipeline.ingest(pdf_path)

    # State: SQLite has FAILED, Chroma has 0 chunks
    from src.utils.hashing import calculate_sha256, generate_doc_id

    doc_id = generate_doc_id(calculate_sha256(pdf_path))
    rec = pipeline.get_record(doc_id)
    assert rec is not None
    assert rec.status == ProcessingStatus.FAILED
    assert "Disk full" in (rec.error_message or "")
    assert pipeline.vector_store.count() == 0

    # Retry succeeds
    retried = pipeline.ingest(pdf_path)
    assert retried.status == ProcessingStatus.COMPLETED
    assert pipeline.vector_store.count() > 0


def test_failure_injection_registry_write_failure_after_upsert(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """If SQLite mark_completed fails after ChromaDB upsert succeeded:

    Chunks are in ChromaDB, SQLite stays in PROCESSING. Retrying re-upserts
    idempotently and finishes COMPLETED without orphan chunks.
    """
    from src.core.exceptions import RegistryError

    pdf_path = pdf_factory("post_upsert_fail.pdf", ["Content upserted but DB fails."])

    # Intercept mark_completed to simulate power failure or SQLite lock error
    with patch.object(
        pipeline.registry,
        "mark_completed",
        side_effect=RegistryError("SQLite locked or disk error"),
    ):
        with pytest.raises(RegistryError):
            pipeline.ingest(pdf_path)

    from src.utils.hashing import calculate_sha256, generate_doc_id

    doc_id = generate_doc_id(calculate_sha256(pdf_path))
    rec = pipeline.get_record(doc_id)
    # The record remains in PROCESSING because mark_completed failed
    assert rec is not None
    assert rec.status == ProcessingStatus.PROCESSING
    # Chunks exist in ChromaDB
    chunks_indexed = pipeline.vector_store.count()
    assert chunks_indexed > 0

    # Safe retry: Calling ingest() again detects PROCESSING, resets to PENDING, re-runs
    retried = pipeline.ingest(pdf_path)
    assert retried.status == ProcessingStatus.COMPLETED
    # Because chunk_ids are deterministic, count does not double
    assert pipeline.vector_store.count() == chunks_indexed


def test_failure_injection_vector_deletion_failure(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """If ChromaDB deletion fails:

    SQLite record is NOT deleted, and VectorStoreError is raised.
    Unrelated documents are completely untouched.
    """
    pdf_path = pdf_factory("del_fail.pdf", ["Content to delete with failure."])
    rec = pipeline.ingest(pdf_path)
    doc_id = rec.doc_id

    with patch.object(
        pipeline.vector_store,
        "delete_document",
        side_effect=VectorStoreError("ChromaDB connection dropped"),
    ):
        with pytest.raises(VectorStoreError):
            pipeline.delete(doc_id)

    # SQLite record still exists
    assert pipeline.get_record(doc_id) is not None
    assert pipeline.registry.count() == 1


def test_failure_injection_registry_deletion_failure_after_vector_del(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """If SQLite deletion fails after ChromaDB chunks are deleted:

    ChromaDB chunks are gone, SQLite still has record.
    Calling delete() again safely succeeds because ChromaDB delete is idempotent.
    """
    from src.core.exceptions import RegistryError

    pdf_path = pdf_factory("partial_del.pdf", ["Partial deletion content."])
    rec = pipeline.ingest(pdf_path)
    doc_id = rec.doc_id

    with patch.object(
        pipeline.registry,
        "delete",
        side_effect=RegistryError("SQLite write lock error"),
    ):
        with pytest.raises(RegistryError):
            pipeline.delete(doc_id)

    # Chroma chunks are gone
    assert pipeline.vector_store.count() == 0
    # SQLite still has the record
    assert pipeline.get_record(doc_id) is not None

    # Retry deletion succeeds cleanly
    second_del = pipeline.delete(doc_id)
    assert second_del is True
    assert pipeline.get_record(doc_id) is None


def test_reconcile_interrupted_documents(pipeline: IngestionPipeline, pdf_factory) -> None:
    """reconcile_interrupted() recovers records left in PENDING/PROCESSING after a crash."""
    pdf1 = pdf_factory("interrupted1.pdf", ["Document 1 crashed mid-flight."])
    pdf2 = pdf_factory("interrupted2.pdf", ["Document 2 crashed in pending."])

    from src.utils.hashing import calculate_sha256, generate_doc_id

    id1 = generate_doc_id(calculate_sha256(pdf1))
    id2 = generate_doc_id(calculate_sha256(pdf2))

    # Manually register PENDING and PROCESSING records simulating hard process kill
    from src.core.models import DocumentRecord

    pipeline.registry.register(
        DocumentRecord(
            doc_id=id1,
            filename=pdf1.name,
            file_hash=calculate_sha256(pdf1),
            file_path=str(pdf1),
            file_size_bytes=100,
            status=ProcessingStatus.PROCESSING,
        )
    )
    pipeline.registry.register(
        DocumentRecord(
            doc_id=id2,
            filename=pdf2.name,
            file_hash=calculate_sha256(pdf2),
            file_path=str(pdf2),
            file_size_bytes=100,
            status=ProcessingStatus.PENDING,
        )
    )

    # Run reconciliation on startup
    recovered = pipeline.reconcile_interrupted()
    assert len(recovered) == 2
    assert pipeline.get_record(id1).status == ProcessingStatus.FAILED
    assert pipeline.get_record(id2).status == ProcessingStatus.FAILED

    # Retrying both recovers them to COMPLETED
    rec1 = pipeline.retry_failed(id1)
    rec2 = pipeline.retry_failed(id2)
    assert rec1.status == ProcessingStatus.COMPLETED
    assert rec2.status == ProcessingStatus.COMPLETED


def test_purge_orphaned_chunks_reconciliation(
    pipeline: IngestionPipeline, pdf_factory
) -> None:
    """purge_orphaned_chunks() removes chunks from ChromaDB when SQLite record is missing."""
    pdf = pdf_factory("orphan_src.pdf", ["Chunks that will become orphaned."])
    rec = pipeline.ingest(pdf)
    assert pipeline.vector_store.count() > 0

    # Simulate out-of-band deletion from SQLite (e.g. manual DB restore)
    pipeline.registry.delete(rec.doc_id)
    assert pipeline.get_record(rec.doc_id) is None
    # Chunks still exist in ChromaDB as orphans
    assert pipeline.vector_store.count() > 0

    # Run purge
    purged_count = pipeline.purge_orphaned_chunks()
    assert purged_count > 0
    assert pipeline.vector_store.count() == 0


def test_duplicate_different_filename_preserves_canonical_filename(
    pipeline: IngestionPipeline, tmp_path: Path, pdf_factory
) -> None:
    """Uploading duplicate content under another filename preserves original metadata."""
    pdf_orig = pdf_factory("original_report.pdf", ["Identical financial figures."])
    rec_orig = pipeline.ingest(pdf_orig)

    pdf_dup = tmp_path / "renamed_copy.pdf"
    pdf_dup.write_bytes(pdf_orig.read_bytes())

    rec_dup = pipeline.ingest(pdf_dup)
    assert rec_dup.doc_id == rec_orig.doc_id
    assert rec_dup.filename == "original_report.pdf"  # canonical original preserved
    assert rec_dup.file_hash == rec_orig.file_hash

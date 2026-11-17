"""Unit tests for SHA-256 hashing and duplicate detection utilities."""

import io
from pathlib import Path

from src.utils.hashing import (
    DuplicateDetector,
    calculate_sha256,
    generate_doc_id,
)


def test_sha256_generation_from_bytes():
    """Verify SHA-256 calculation for in-memory bytes."""
    import hashlib

    data = b"Hello RAG World"
    expected = hashlib.sha256(data).hexdigest()
    result = calculate_sha256(data)
    assert result == expected


def test_sha256_generation_from_file(tmp_path: Path):
    """Verify SHA-256 calculation for a physical file."""
    test_file = tmp_path / "test.dat"
    test_file.write_bytes(b"Consistent file content for hashing")
    hash1 = calculate_sha256(test_file)
    hash2 = calculate_sha256(test_file)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest length


def test_sha256_generation_from_stream():
    """Verify SHA-256 calculation from a file-like stream with position reset."""
    buf = io.BytesIO(b"Stream content")
    hash_val = calculate_sha256(buf)
    assert len(hash_val) == 64
    # Stream position should be restored
    assert buf.tell() == 0


def test_deterministic_doc_id():
    """Verify that identical hashes produce identical UUIDv5 doc_ids."""
    file_hash = "8c668b5bbca23053ef5137eb1ca776bb5ff64ec72566ecb765ec177b96e5797f"
    id1 = generate_doc_id(file_hash)
    id2 = generate_doc_id(file_hash)
    assert id1 == id2
    assert len(id1) == 36  # Standard UUID string length


def test_duplicate_file_detection():
    """Verify duplicate detector identifies registered hashes."""
    detector = DuplicateDetector()
    file_hash = "abcdef1234567890" * 4
    doc_id = "test-doc-id-1"

    assert not detector.is_duplicate(file_hash)
    detector.register(file_hash=file_hash, doc_id=doc_id, filename="report.pdf")
    assert detector.is_duplicate(file_hash)
    assert detector.registered_count == 1

    existing = detector.get_existing(file_hash)
    assert existing is not None
    assert existing["doc_id"] == doc_id
    assert existing["filename"] == "report.pdf"


def test_different_filenames_with_identical_contents(tmp_path: Path):
    """Requirement: Detect duplicate uploads even if the filename is different."""
    content = b"%PDF-1.4 Identical Content For Both Files"
    file_a = tmp_path / "annual_report_v1.pdf"
    file_b = tmp_path / "annual_report_copy.pdf"

    file_a.write_bytes(content)
    file_b.write_bytes(content)

    hash_a = calculate_sha256(file_a)
    hash_b = calculate_sha256(file_b)

    assert hash_a == hash_b

    detector = DuplicateDetector()
    detector.register(hash_a, "doc-1", file_a.name)

    # file_b must be detected as duplicate despite different filename
    assert detector.is_duplicate(hash_b)


def test_different_contents_with_identical_filenames(tmp_path: Path):
    """Requirement: Identify distinct documents even if they share the same filename."""
    dir_a = tmp_path / "folder_a"
    dir_b = tmp_path / "folder_b"
    dir_a.mkdir()
    dir_b.mkdir()

    file_a = dir_a / "document.pdf"
    file_b = dir_b / "document.pdf"

    file_a.write_bytes(b"%PDF-1.4 Version Alpha")
    file_b.write_bytes(b"%PDF-1.4 Version Beta")

    hash_a = calculate_sha256(file_a)
    hash_b = calculate_sha256(file_b)

    assert hash_a != hash_b

    detector = DuplicateDetector()
    detector.register(hash_a, "doc-a", file_a.name)

    # Same filename, but different content hash -> not a duplicate
    assert not detector.is_duplicate(hash_b)

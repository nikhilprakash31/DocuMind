"""Cryptographic hashing and duplicate detection utilities."""

import hashlib
import uuid
from pathlib import Path
from typing import BinaryIO

DEFAULT_CHUNK_SIZE = 65536  # 64 KB read buffer to avoid memory spikes on large files


def calculate_sha256(
    file_input: str | Path | BinaryIO | bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Calculate SHA-256 hexadecimal digest for a file, stream, or bytes.

    Reads in fixed-size chunks to ensure constant memory footprint regardless of file size.
    """
    hasher = hashlib.sha256()

    if isinstance(file_input, bytes):
        hasher.update(file_input)
        return hasher.hexdigest()

    if hasattr(file_input, "read"):
        # File-like object (e.g. io.BytesIO or open file)
        pos = file_input.tell() if hasattr(file_input, "tell") else None
        try:
            while chunk := file_input.read(chunk_size):
                hasher.update(chunk)
        finally:
            if pos is not None and hasattr(file_input, "seek"):
                file_input.seek(pos)  # Reset stream position for subsequent readers
        return hasher.hexdigest()

    path = Path(file_input)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: '{path}'")

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def generate_doc_id(file_hash: str) -> str:
    """Generate a deterministic UUIDv5 identifier based on SHA-256 content hash."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, file_hash))


class DuplicateDetector:
    """In-memory registry to track ingested document fingerprints and detect duplicates."""

    def __init__(self) -> None:
        # Maps SHA-256 file_hash -> dict of metadata {'doc_id': str, 'filename': str}
        self._registry: dict[str, dict[str, str]] = {}

    def is_duplicate(self, file_hash: str) -> bool:
        """Check if a file with the given SHA-256 hash has already been registered."""
        return file_hash in self._registry

    def register(self, file_hash: str, doc_id: str, filename: str) -> None:
        """Register a document fingerprint. Overwrites existing record if present."""
        self._registry[file_hash] = {
            "doc_id": doc_id,
            "filename": filename,
        }

    def get_existing(self, file_hash: str) -> dict[str, str] | None:
        """Retrieve existing registration info for a file hash, or None if not seen."""
        return self._registry.get(file_hash)

    def unregister(self, file_hash: str) -> bool:
        """Remove a document hash from the registry. Returns True if removed, False if absent."""
        return self._registry.pop(file_hash, None) is not None

    def clear(self) -> None:
        """Clear all registered document fingerprints."""
        self._registry.clear()

    @property
    def registered_count(self) -> int:
        """Number of unique document hashes currently registered."""
        return len(self._registry)

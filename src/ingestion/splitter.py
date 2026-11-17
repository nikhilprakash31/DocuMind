"""Configurable text chunking using RecursiveCharacterTextSplitter."""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.core.models import ChunkingEvaluationResult, ChunkMetadata

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


class DocumentSplitter:
    """Splits documents into semantically coherent chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            strip_whitespace=True,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of extracted documents into chunks with inherited metadata.

        Assigns a sequential chunk_index and deterministic chunk_id per document.
        """
        if not documents:
            return []

        raw_chunks = self._splitter.split_documents(documents)
        enriched_chunks: list[Document] = []
        doc_chunk_counters: dict[str, int] = defaultdict(int)

        for chunk in raw_chunks:
            content = chunk.page_content.strip()
            if not content:
                continue

            metadata_dict = dict(chunk.metadata)
            doc_id = metadata_dict.get("doc_id", "doc_default")
            chunk_idx = doc_chunk_counters[doc_id]
            doc_chunk_counters[doc_id] += 1

            chunk_id = f"{doc_id}_c{chunk_idx}"
            char_count = len(content)

            chunk_meta = ChunkMetadata(
                source=metadata_dict.get("source", "unknown"),
                filename=metadata_dict.get("filename", "unknown"),
                page=int(metadata_dict.get("page", 1)),
                total_pages=int(metadata_dict.get("total_pages", 1)),
                doc_id=doc_id,
                doc_hash=metadata_dict.get("doc_hash", ""),
                file_size_bytes=int(metadata_dict.get("file_size_bytes", 0)),
                upload_timestamp=metadata_dict.get("upload_timestamp", ""),
                chunk_index=chunk_idx,
                chunk_id=chunk_id,
                char_count=char_count,
            )

            enriched_chunks.append(
                Document(
                    page_content=content,
                    metadata=chunk_meta.model_dump(),
                )
            )

        logger.debug(
            "Split {} input document pages into {} text chunks (chunk_size={}, chunk_overlap={})",
            len(documents),
            len(enriched_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        return enriched_chunks

    def split_text(
        self,
        text: str,
        base_metadata: dict | None = None,
    ) -> list[Document]:
        """Split a raw text string into chunks with optional base metadata."""
        if not text.strip():
            return []

        doc = Document(page_content=text, metadata=base_metadata or {})
        return self.split_documents([doc])


def evaluate_chunk_configurations(
    documents: list[Document],
    configurations: list[tuple[int, int]] | None = None,
) -> list[ChunkingEvaluationResult]:
    """Evaluate multiple chunking configurations on a given document corpus."""
    target_configs = configurations or [
        (800, 150),
        (1000, 200),
        (1200, 200),
    ]

    results: list[ChunkingEvaluationResult] = []

    for size, overlap in target_configs:
        splitter = DocumentSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_documents(documents)

        if not chunks:
            results.append(
                ChunkingEvaluationResult(
                    config_name=f"{size}/{overlap}",
                    chunk_size=size,
                    chunk_overlap=overlap,
                    total_chunks=0,
                    avg_chunk_length=0.0,
                    min_chunk_length=0,
                    max_chunk_length=0,
                    total_characters=0,
                )
            )
            continue

        lengths = [len(c.page_content) for c in chunks]
        total_chars = sum(lengths)
        avg_len = total_chars / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        result = ChunkingEvaluationResult(
            config_name=f"{size}/{overlap}",
            chunk_size=size,
            chunk_overlap=overlap,
            total_chunks=len(chunks),
            avg_chunk_length=round(avg_len, 2),
            min_chunk_length=min_len,
            max_chunk_length=max_len,
            total_characters=total_chars,
        )
        results.append(result)

    return results

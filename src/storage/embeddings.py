"""Nomic embedding adapter enforcing asymmetric prefixing and dimensionality validation."""

import re
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from src.core.exceptions import (
    EmbeddingConnectionError,
    EmbeddingDimensionError,
    EmbeddingError,
)

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
EXPECTED_EMBEDDING_DIM = 768
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Regex to detect if text already starts with either prefix
DOC_PREFIX_PATTERN = re.compile(r"^\s*search_document:\s*", re.IGNORECASE)
QUERY_PREFIX_PATTERN = re.compile(r"^\s*search_query:\s*", re.IGNORECASE)


class NomicEmbeddingAdapter(Embeddings):
    """Adapter for nomic-embed-text enforcing task-specific prefixes and dimension checks."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.expected_dim = expected_dim
        self._embedder = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
            **kwargs,
        )

    @staticmethod
    def format_document_text(text: str) -> str:
        """Format text for document indexing, avoiding double prefixing."""
        cleaned = text.strip()
        if DOC_PREFIX_PATTERN.match(cleaned):
            return cleaned
        # If inadvertently prefixed with query prefix, replace it
        if QUERY_PREFIX_PATTERN.match(cleaned):
            cleaned = QUERY_PREFIX_PATTERN.sub("", cleaned).strip()
        return f"{DOC_PREFIX}{cleaned}"

    @staticmethod
    def format_query_text(text: str) -> str:
        """Format text for search query, avoiding double prefixing."""
        cleaned = text.strip()
        if QUERY_PREFIX_PATTERN.match(cleaned):
            return cleaned
        # If inadvertently prefixed with doc prefix, replace it
        if DOC_PREFIX_PATTERN.match(cleaned):
            cleaned = DOC_PREFIX_PATTERN.sub("", cleaned).strip()
        return f"{QUERY_PREFIX}{cleaned}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks with 'search_document: ' prefix and validate dimensions."""
        if not texts:
            return []

        formatted_texts = [self.format_document_text(t) for t in texts]

        try:
            vectors = self._embedder.embed_documents(formatted_texts)
        except Exception as e:
            logger.error("Embedding generation failed for {} documents: {}", len(texts), e)
            msg = str(e).lower()
            if "connection refused" in msg or "connect" in msg or "timeout" in msg:
                raise EmbeddingConnectionError(
                    f"Failed to connect to Ollama embedding service at '{self.base_url}': {e}"
                ) from e
            raise EmbeddingError(f"Ollama embedding generation failed: {e}") from e

        for idx, vec in enumerate(vectors):
            dim = len(vec)
            if dim != self.expected_dim:
                raise EmbeddingDimensionError(expected=self.expected_dim, received=dim)

        logger.debug(
            "Successfully embedded {} documents (dim={})",
            len(vectors),
            self.expected_dim,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with 'search_query: ' prefix and validate dimensions."""
        if not text.strip():
            raise ValueError("Query text cannot be empty.")

        formatted_query = self.format_query_text(text)

        try:
            vector = self._embedder.embed_query(formatted_query)
        except Exception as e:
            logger.error("Embedding query failed for '{}': {}", text[:50], e)
            msg = str(e).lower()
            if "connection refused" in msg or "connect" in msg or "timeout" in msg:
                raise EmbeddingConnectionError(
                    f"Failed to connect to Ollama embedding service at '{self.base_url}': {e}"
                ) from e
            raise EmbeddingError(f"Ollama query embedding failed: {e}") from e

        dim = len(vector)
        if dim != self.expected_dim:
            raise EmbeddingDimensionError(expected=self.expected_dim, received=dim)

        logger.debug("Successfully embedded query (dim={})", self.expected_dim)
        return vector

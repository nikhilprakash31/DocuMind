"""Baseline semantic retriever for querying vector store with metadata filtering."""

from typing import Any

from loguru import logger

from src.core.models import RetrievalResult
from src.storage.vector_store import ChromaVectorStore

DEFAULT_TOP_K = 4


class VectorRetriever:
    """Semantic retriever using dense vector similarity over persistent ChromaDB."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top_k most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Optional number of results to retrieve (defaults to self.default_top_k).
            filter_criteria: Optional Chroma metadata filter dictionary (e.g. {'doc_id': '...'}).

        Returns:
            List of RetrievalResult sorted by relevance (lowest distance / highest score first).
        """
        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            raise ValueError(f"top_k must be greater than zero, got {k}")

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty or whitespace.")

        logger.debug(
            "Executing retrieval for query='{}' with top_k={} and filter={}",
            cleaned_query[:50],
            k,
            filter_criteria,
        )

        results = self.vector_store.query_similar(
            query=cleaned_query,
            top_k=k,
            filter_criteria=filter_criteria,
        )

        # Sort by distance ascending (lower distance = closer semantic match)
        results.sort(key=lambda r: r.distance)

        logger.info(
            "Retrieved {} chunks for query '{}' (best score: {})",
            len(results),
            cleaned_query[:40],
            results[0].score if results else "N/A",
        )
        return results

    def retrieve_for_document(
        self,
        query: str,
        doc_id: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Convenience method to retrieve chunks constrained to a specific document."""
        if not doc_id:
            raise ValueError("doc_id cannot be empty.")
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_criteria={"doc_id": doc_id},
        )

    def retrieve_for_filename(
        self,
        query: str,
        filename: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Convenience method to retrieve chunks constrained to a specific source filename."""
        if not filename:
            raise ValueError("filename cannot be empty.")
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_criteria={"filename": filename},
        )

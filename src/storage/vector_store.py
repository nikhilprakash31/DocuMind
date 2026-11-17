"""Persistent ChromaDB vector store wrapper for document chunks."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from loguru import logger

from src.core.exceptions import VectorStoreError
from src.core.models import RetrievalResult
from src.storage.embeddings import NomicEmbeddingAdapter

DEFAULT_COLLECTION_NAME = "documind_chunks"
DEFAULT_PERSIST_DIRECTORY = "./data/vectorstore"
DEFAULT_BATCH_SIZE = 64
HNSW_SPACE = "cosine"


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage and cosine similarity retrieval."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_adapter: NomicEmbeddingAdapter | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_adapter = embedding_adapter or NomicEmbeddingAdapter()

        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": HNSW_SPACE},
            )
            logger.info(
                "Initialized ChromaDB collection '{}' at '{}' (metric: {})",
                self.collection_name,
                self.persist_directory,
                HNSW_SPACE,
            )
        except Exception as e:
            logger.error("Failed to initialize ChromaDB persistent client: {}", e)
            raise VectorStoreError(f"ChromaDB initialization failed: {e}") from e

    @property
    def collection(self) -> Any:
        """Direct handle to underlying Chroma collection."""
        return self._collection

    def count(self) -> int:
        """Return total number of chunks indexed in the collection."""
        try:
            return self._collection.count()
        except Exception as e:
            logger.error("Failed to retrieve collection count: {}", e)
            raise VectorStoreError(f"Could not retrieve collection count: {e}") from e

    def upsert_chunks(
        self,
        chunks: list[Document],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> int:
        """Upsert a list of document chunks into ChromaDB in batches.

        Preserves all chunk metadata and calculates embeddings with the Nomic adapter.
        Existing records with matching chunk_ids are overwritten.
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty chunks list.")
            return 0

        total_upserted = 0
        total_chunks = len(chunks)

        for start_idx in range(0, total_chunks, batch_size):
            batch = chunks[start_idx : start_idx + batch_size]
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []

            for chunk in batch:
                meta = dict(chunk.metadata)
                chunk_id = meta.get("chunk_id")
                if not chunk_id:
                    # Fallback deterministic chunk ID if missing
                    doc_id = meta.get("doc_id", "doc")
                    chunk_idx = meta.get("chunk_index", 0)
                    chunk_id = f"{doc_id}_c{chunk_idx}"
                    meta["chunk_id"] = chunk_id

                # Ensure all metadata values conform to Chroma primitive types
                sanitized_meta: dict[str, Any] = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    elif v is None:
                        sanitized_meta[k] = ""
                    else:
                        sanitized_meta[k] = str(v)

                ids.append(chunk_id)
                documents.append(chunk.page_content)
                metadatas.append(sanitized_meta)

            # Generate embeddings for the batch
            try:
                embeddings = self.embedding_adapter.embed_documents(documents)
            except Exception as e:
                end_idx = start_idx + len(batch)
                logger.error(
                    "Failed to generate embeddings for batch {}-{}: {}", start_idx, end_idx, e
                )
                raise VectorStoreError(f"Batch embedding generation failed: {e}") from e

            # Upsert into Chroma
            try:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                total_upserted += len(batch)
                logger.debug(
                    "Upserted batch {}/{} ({} chunks) into collection '{}'",
                    start_idx // batch_size + 1,
                    (total_chunks + batch_size - 1) // batch_size,
                    len(batch),
                    self.collection_name,
                )
            except Exception as e:
                end_idx = start_idx + len(batch)
                logger.error("Chroma upsert failed for batch {}-{}: {}", start_idx, end_idx, e)
                raise VectorStoreError(f"Chroma upsert operation failed: {e}") from e

        logger.info(
            "Successfully upserted {} chunks into ChromaDB (collection count: {})",
            total_upserted,
            self.count(),
        )
        return total_upserted

    def delete_document(self, doc_id: str) -> int:
        """Safely delete all chunks belonging to a specific document ID.

        Returns the number of chunks deleted.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty.")

        try:
            # Query existing chunk count for this document
            existing = self._collection.get(where={"doc_id": doc_id})
            existing_ids = existing.get("ids", [])
            count = len(existing_ids)

            if count > 0:
                self._collection.delete(where={"doc_id": doc_id})
                logger.info("Deleted {} chunks for document '{}'", count, doc_id)
            else:
                logger.debug("No chunks found to delete for document '{}'", doc_id)

            return count
        except Exception as e:
            logger.error("Failed to delete chunks for document '{}': {}", doc_id, e)
            raise VectorStoreError(f"Failed to delete document '{doc_id}': {e}") from e

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete specific chunks by their IDs."""
        if not chunk_ids:
            return

        try:
            self._collection.delete(ids=chunk_ids)
            logger.info("Deleted {} chunks by ID", len(chunk_ids))
        except Exception as e:
            logger.error("Failed to delete chunks {}: {}", chunk_ids, e)
            raise VectorStoreError(f"Failed to delete chunks: {e}") from e

    def query_similar(
        self,
        query: str,
        top_k: int = 4,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Query the vector store for top_k most similar chunks using cosine similarity.

        In Chroma with cosine distance, distance d is in [0, 2]. Lower distance indicates
        a closer match. Cosine similarity ranges from -1.0 to +1.0 and is computed as
        (1.0 - distance).
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than zero, got {top_k}")

        total_available = self.count()
        if total_available == 0:
            logger.warning("query_similar called on empty collection '{}'", self.collection_name)
            return []

        actual_n = min(top_k, total_available)

        # Generate query vector
        try:
            query_vector = self.embedding_adapter.embed_query(query)
        except Exception as e:
            logger.error("Failed to embed query for retrieval: {}", e)
            raise VectorStoreError(f"Query embedding generation failed: {e}") from e

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=actual_n,
                where=filter_criteria if filter_criteria else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query execution failed: {}", e)
            raise VectorStoreError(f"ChromaDB query execution failed: {e}") from e

        retrieval_results: list[RetrievalResult] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        for chunk_id, content, meta, dist in zip(
            ids_list, docs_list, metas_list, distances_list, strict=False
        ):
            meta_dict = dict(meta) if meta else {}
            cosine_dist = float(dist)
            # In cosine space, similarity = 1 - distance
            # Clamped to range [-1.0, 1.0]
            similarity = round(1.0 - cosine_dist, 4)

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=str(meta_dict.get("doc_id", "")),
                    filename=str(meta_dict.get("filename", "")),
                    page=int(meta_dict.get("page", 1)),
                    chunk_index=int(meta_dict.get("chunk_index", 0)),
                    content=content,
                    distance=round(cosine_dist, 4),
                    score=similarity,
                    metadata=meta_dict,
                )
            )

        logger.debug(
            "Query '{}' returned {} results (top_k={})",
            query[:40],
            len(retrieval_results),
            top_k,
        )
        return retrieval_results

    def get_collection_stats(self) -> dict[str, Any]:
        """Return diagnostic metrics about the collection and persistence directory."""
        return {
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "total_chunks": self.count(),
            "hnsw_space": HNSW_SPACE,
            "embedding_dim": self.embedding_adapter.expected_dim,
            "embedding_model": self.embedding_adapter.model,
        }

    def clear_collection(self) -> None:
        """Clear all records from the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": HNSW_SPACE},
            )
            logger.info("Cleared collection '{}'", self.collection_name)
        except Exception as e:
            logger.error("Failed to clear collection '{}': {}", self.collection_name, e)
            raise VectorStoreError(f"Failed to clear collection: {e}") from e

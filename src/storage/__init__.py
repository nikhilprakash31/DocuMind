"""Vector store, document registry, and embedding adapter components."""

from src.storage.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DOC_PREFIX,
    EXPECTED_EMBEDDING_DIM,
    QUERY_PREFIX,
    NomicEmbeddingAdapter,
)
from src.storage.ingestion_pipeline import IngestionPipeline
from src.storage.registry import (
    DEFAULT_REGISTRY_PATH,
    DocumentRegistry,
)
from src.storage.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    ChromaVectorStore,
)

__all__ = [
    "ChromaVectorStore",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_PERSIST_DIRECTORY",
    "DEFAULT_REGISTRY_PATH",
    "DOC_PREFIX",
    "DocumentRegistry",
    "EXPECTED_EMBEDDING_DIM",
    "IngestionPipeline",
    "NomicEmbeddingAdapter",
    "QUERY_PREFIX",
]

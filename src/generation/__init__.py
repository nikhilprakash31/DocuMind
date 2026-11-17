"""RAG generation pipeline components."""

from src.generation.chain import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    RAGGenerationChain,
)
from src.generation.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    FALLBACK_EMPTY_RETRIEVAL,
    FALLBACK_INSUFFICIENT_CONTEXT,
    extract_citations,
    format_context_passages,
)

__all__ = [
    "DEFAULT_GENERATION_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "FALLBACK_EMPTY_RETRIEVAL",
    "FALLBACK_INSUFFICIENT_CONTEXT",
    "RAGGenerationChain",
    "extract_citations",
    "format_context_passages",
]

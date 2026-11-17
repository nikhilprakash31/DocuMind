"""RAG generation chain integrating VectorRetriever with local Ollama LLM."""

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger

from src.citations.mapper import EvidenceMapper
from src.core.exceptions import GenerationConnectionError, GenerationError
from src.core.models import RAGResponse, RetrievalResult
from src.generation.prompts import (
    DEFAULT_RAG_SYSTEM_PROMPT,
    FALLBACK_EMPTY_RETRIEVAL,
    FALLBACK_INSUFFICIENT_CONTEXT,
    extract_citations,
    format_context_passages,
)
from src.retrieval.retriever import VectorRetriever

DEFAULT_GENERATION_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512


class RAGGenerationChain:
    """Orchestrates grounded question answering over retrieved document passages."""

    def __init__(
        self,
        retriever: VectorRetriever,
        model_name: str = DEFAULT_GENERATION_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.retriever = retriever
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        self._evidence_mapper = EvidenceMapper()

        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            **kwargs,
        )
        logger.info(
            "Initialized RAGGenerationChain with model='{}' (temperature={}, max_tokens={})",
            self.model_name,
            self.temperature,
            self.max_tokens,
        )

    def generate_from_context(
        self,
        query: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> RAGResponse:
        """Synthesize a grounded answer using pre-retrieved document passages."""
        start_time = time.perf_counter()
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        # Guard: Handle empty context
        if not retrieved_chunks:
            logger.info("Empty retrieval context provided for query '{}'", cleaned_query[:40])
            elapsed = time.perf_counter() - start_time
            return RAGResponse(
                answer=FALLBACK_EMPTY_RETRIEVAL,
                query=cleaned_query,
                sources=[],
                citations=[],
                latency_seconds=round(max(elapsed, 0.001), 4),
                model_name=self.model_name,
                is_grounded=False,
            )

        # Build prompt
        context_block = format_context_passages(retrieved_chunks)
        user_message_content = (
            f"<context>\n"
            f"{context_block}\n"
            f"</context>\n\n"
            f"Question: {cleaned_query}\n\n"
            f"Answer:"
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message_content),
        ]

        logger.debug(
            "Invoking LLM for query='{}' with {} context chunks",
            cleaned_query[:40],
            len(retrieved_chunks),
        )

        try:
            llm_response = self._llm.invoke(messages)
            answer_text = str(llm_response.content).strip()
        except Exception as e:
            logger.error("LLM generation failed for query '{}': {}", cleaned_query[:40], e)
            msg = str(e).lower()
            if "connection refused" in msg or "connect" in msg or "timeout" in msg:
                raise GenerationConnectionError(
                    f"Failed to connect to Ollama LLM service at '{self.base_url}': {e}"
                ) from e
            raise GenerationError(f"RAG text generation failed: {e}") from e

        elapsed = time.perf_counter() - start_time

        # Determine grounding status
        is_grounded = True
        lower_ans = answer_text.lower()
        if (
            "could not find information" in lower_ans
            or "not mentioned in the provided" in lower_ans
            or "does not contain" in lower_ans
            or lower_ans.startswith(FALLBACK_INSUFFICIENT_CONTEXT.lower())
        ):
            is_grounded = False

        citations = extract_citations(answer_text, retrieved_chunks)
        evidence_map = self._evidence_mapper.map_evidence(
            answer=answer_text,
            retrieved_chunks=retrieved_chunks,
            query=cleaned_query,
        )

        logger.info(
            "Synthesized answer in {:.2f}s (grounded={}, citations={}, cards={})",
            elapsed,
            is_grounded,
            len(citations),
            len(evidence_map.evidence_cards),
        )

        return RAGResponse(
            answer=answer_text,
            query=cleaned_query,
            sources=retrieved_chunks,
            citations=citations,
            evidence_map=evidence_map,
            latency_seconds=round(max(elapsed, 0.001), 4),
            model_name=self.model_name,
            is_grounded=is_grounded,
        )

    def generate(
        self,
        query: str,
        top_k: int | None = None,
        filter_criteria: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Execute full RAG pipeline: retrieve relevant chunks, then synthesize answer."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        # Step 1: Retrieve context chunks
        chunks = self.retriever.retrieve(
            query=cleaned_query,
            top_k=top_k,
            filter_criteria=filter_criteria,
        )

        # Step 2: Synthesize grounded response
        return self.generate_from_context(
            query=cleaned_query,
            retrieved_chunks=chunks,
        )

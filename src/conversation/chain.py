"""Conversational RAG orchestrator integrating sessions, reformulation, and generation."""

import time
from typing import Any

from loguru import logger

from src.conversation.manager import ConversationManager
from src.conversation.reformulator import QueryReformulator
from src.core.models import ConversationalRAGResponse
from src.generation.chain import RAGGenerationChain
from src.retrieval.retriever import VectorRetriever


class ConversationalRAGChain:
    """Orchestrates multi-turn grounded conversations over selected document corpora."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        query_reformulator: QueryReformulator,
        retriever: VectorRetriever,
        rag_generator: RAGGenerationChain,
    ) -> None:
        self.manager = conversation_manager
        self.reformulator = query_reformulator
        self.retriever = retriever
        self.generator = rag_generator

    def ask(
        self,
        session_id: str,
        query: str,
        top_k: int | None = None,
    ) -> ConversationalRAGResponse:
        """Process a conversational turn for a given session.

        1. Fetches session and conversation history.
        2. Contextualizes follow-up query into a standalone retrieval query.
        3. Appends raw user query to history (original text unchanged).
        4. Applies document scope filters based on session.selected_doc_ids.
        5. Executes vector retrieval using the standalone query.
        6. Generates grounded answer with citation verification.
        7. Appends assistant response to history and returns structured response.
        """
        turn_start = time.perf_counter()
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        # Step 1: Validate session exists
        session = self.manager.get_session(session_id)
        history = list(session.messages)

        # Step 2: Contextualize follow-up query if history exists
        standalone_query, reform_latency = self.reformulator.reformulate(
            query=cleaned_query,
            chat_history=history,
        )

        # Step 3: Record original user question in session history
        self.manager.add_message(
            session_id=session_id,
            role="user",
            content=cleaned_query,
        )

        # Step 4: Build metadata filter from session.selected_doc_ids
        filter_criteria: dict[str, Any] | None = None
        if session.selected_doc_ids:
            if len(session.selected_doc_ids) == 1:
                filter_criteria = {"doc_id": session.selected_doc_ids[0]}
            else:
                filter_criteria = {"doc_id": {"$in": session.selected_doc_ids}}

        # Step 5: Retrieve relevant chunks using the standalone query
        chunks = self.retriever.retrieve(
            query=standalone_query,
            top_k=top_k,
            filter_criteria=filter_criteria,
        )

        # Step 6: Generate grounded answer
        rag_res = self.generator.generate_from_context(
            query=cleaned_query,
            retrieved_chunks=chunks,
        )

        total_turn_latency = time.perf_counter() - turn_start

        # Step 7: Record assistant response in session history
        self.manager.add_message(
            session_id=session_id,
            role="assistant",
            content=rag_res.answer,
            standalone_query=standalone_query,
            citations=rag_res.citations,
            sources=rag_res.sources,
            latency_seconds=round(total_turn_latency, 4),
            is_grounded=rag_res.is_grounded,
        )

        logger.info(
            "Conversational turn completed for session '{}' in {:.2f}s (reform: {:.2f}s, hits: {})",
            session_id,
            total_turn_latency,
            reform_latency,
            len(chunks),
        )

        return ConversationalRAGResponse(
            answer=rag_res.answer,
            user_query=cleaned_query,
            standalone_query=standalone_query,
            session_id=session_id,
            citations=rag_res.citations,
            evidence_map=rag_res.evidence_map,
            sources=rag_res.sources,
            latency_seconds=round(total_turn_latency, 4),
            reformulation_latency_seconds=reform_latency,
            is_grounded=rag_res.is_grounded,
        )

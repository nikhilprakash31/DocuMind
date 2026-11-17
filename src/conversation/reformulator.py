"""Query reformulation and contextualization for multi-turn RAG conversations."""

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger

from src.core.models import ChatMessage

REFORMULATOR_SYSTEM_PROMPT = (
    "You are a search query reformulation expert for a document Q&A assistant.\n\n"
    "Task: Given a conversation history between a user and an assistant, and a follow-up "
    "question from the user, rephrase the follow-up question into a single, complete, "
    "standalone search query that incorporates necessary context (such as document names, "
    "metrics, and entities referenced by pronouns like 'it', 'they', 'that').\n\n"
    "Strict Rules:\n"
    "1. Do NOT answer the question under any circumstances.\n"
    "2. Do NOT include any prefixes, commentary, or quotes.\n"
    "3. If the follow-up question is already standalone, return it exactly as is.\n"
    "4. Output ONLY the standalone question."
)

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class QueryReformulator:
    """Reformulates conversational follow-up questions into standalone vector retrieval queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,
            num_predict=128,
            **kwargs,
        )

    @staticmethod
    def format_history_for_prompt(history: list[ChatMessage], max_turns: int = 4) -> str:
        """Format the recent conversation turns into a compact dialogue script."""
        if not history:
            return ""

        # Take the most recent max_turns messages
        recent = history[-max_turns:]
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            # Keep assistant responses compact to conserve prompt tokens
            content = msg.content.strip().replace("\n", " ")
            if len(content) > 150:
                content = content[:150] + "..."
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    def reformulate(
        self,
        query: str,
        chat_history: list[ChatMessage],
    ) -> tuple[str, float]:
        """Contextualize a follow-up query based on prior conversation history.

        Returns:
            Tuple of (standalone_query, latency_seconds).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return "", 0.0

        # Short-circuit: If no history exists, the query is already initial/standalone
        if not chat_history:
            return cleaned_query, 0.0

        # Short-circuit: If history contains no assistant responses yet, keep query
        has_assistant_turns = any(m.role == "assistant" for m in chat_history)
        if not has_assistant_turns:
            return cleaned_query, 0.0

        start_time = time.perf_counter()
        history_str = self.format_history_for_prompt(chat_history)

        prompt_content = (
            f"Conversation History:\n"
            f"{history_str}\n\n"
            f"Follow-up Question: {cleaned_query}\n\n"
            f"Standalone Question:"
        )

        messages = [
            SystemMessage(content=REFORMULATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        try:
            response = self._llm.invoke(messages)
            standalone = str(response.content).strip().strip('"').strip("'")
            elapsed = time.perf_counter() - start_time

            # Guard against empty or conversational replies from the rewriter
            if not standalone or len(standalone) < 3:
                standalone = cleaned_query

            logger.debug(
                "Reformulated follow-up '{}' -> '{}' ({:.2f}s)",
                cleaned_query[:40],
                standalone[:50],
                elapsed,
            )
            return standalone, round(elapsed, 4)

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                "Query reformulation failed for '{}': {}. Falling back to raw query.",
                cleaned_query[:40],
                e,
            )
            # Safe degradation: fall back to original user query
            return cleaned_query, round(elapsed, 4)

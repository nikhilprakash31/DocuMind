"""Prompts and context formatting utilities for grounded RAG generation."""

import re

from src.core.models import Citation, RetrievalResult

DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are DocuMind, an expert document assistant providing factual, grounded answers.\n\n"
    "Strict Rules:\n"
    "1. Answer the user's question using ONLY the facts provided in the <context> block below.\n"
    "2. For every factual claim, include a page citation in brackets, e.g. [Page X, filename].\n"
    "3. If the provided context does not contain sufficient facts to answer, respond with:\n"
    '   "Based on the provided documents, I could not find information to answer this question."\n'
    "4. Do NOT speculate, extrapolate, or introduce facts from outside the provided context.\n"
    "5. Keep answers direct, professional, and well-structured."
)

FALLBACK_INSUFFICIENT_CONTEXT = (
    "Based on the provided documents, I could not find information to answer this question."
)

FALLBACK_EMPTY_RETRIEVAL = (
    "No relevant document passages were found in the knowledge base to answer your question."
)


def format_context_passages(chunks: list[RetrievalResult]) -> str:
    """Format a list of retrieved chunks into an unambiguous context block for the LLM."""
    if not chunks:
        return ""

    formatted_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        header = f"[Passage {idx} | Source: {chunk.filename} | Page: {chunk.page}]"
        block = f"{header}\n{chunk.content.strip()}"
        formatted_blocks.append(block)

    return "\n\n---\n\n".join(formatted_blocks)


def extract_citations(
    answer: str,
    sources: list[RetrievalResult],
) -> list[Citation]:
    """Extract and verify structured citations referenced in the generated answer.

    Distinguishes citation extraction from verification:
    - Extracts bracketed citation patterns (e.g. [Page X, filename] or [Page X]).
    - Verifies whether the referenced page and filename match an actual retrieved chunk.
    - Sets is_verified=True for confirmed matches and preserves source doc_id and chunk_id.
    - Sets is_verified=False for references that do not correspond to any retrieved passage.
    """
    if not sources or FALLBACK_INSUFFICIENT_CONTEXT in answer:
        return []

    # Map (filename, page) and (page) to representative RetrievalResult
    source_map: dict[tuple[str, int], RetrievalResult] = {}
    page_to_sources: dict[int, list[RetrievalResult]] = {}
    for s in sources:
        key = (s.filename.lower(), s.page)
        if key not in source_map:
            source_map[key] = s
        page_to_sources.setdefault(s.page, []).append(s)

    # Regex to detect page citations like "[Page 2]" or "[Page 2, handbook.pdf]"
    page_pattern = re.compile(
        r"\[(?:Page\s*(\d+)(?:,\s*([^\]]+))?|([^\]]+),\s*Page\s*(\d+))\]",
        re.IGNORECASE,
    )

    found_citations: list[Citation] = []
    seen_refs: set[tuple[str, int]] = set()

    for match in page_pattern.finditer(answer):
        raw_token = match.group(0)
        p1, f1, f2, p2 = match.groups()
        page_num = int(p1) if p1 else int(p2)
        filename_hint = (f1 or f2 or "").strip()

        matched_source: RetrievalResult | None = None

        if filename_hint:
            # Match by page and filename substring
            for (src_file, src_page), src_res in source_map.items():
                if src_page == page_num and filename_hint.lower() in src_file:
                    matched_source = src_res
                    break
        else:
            # Match by page number alone if unique or default to first matching chunk
            candidates = page_to_sources.get(page_num, [])
            if candidates:
                matched_source = candidates[0]

        ref_key = (matched_source.filename if matched_source else filename_hint, page_num)
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)

        if matched_source is not None:
            snippet = (
                matched_source.content[:150] + "..."
                if len(matched_source.content) > 150
                else matched_source.content
            )
            found_citations.append(
                Citation(
                    filename=matched_source.filename,
                    page=matched_source.page,
                    chunk_id=matched_source.chunk_id,
                    doc_id=matched_source.doc_id,
                    snippet=snippet,
                    is_verified=True,
                    raw_reference=raw_token,
                )
            )
        else:
            # Unmatched / fabricated citation
            found_citations.append(
                Citation(
                    filename=filename_hint or "unknown",
                    page=page_num,
                    chunk_id="",
                    doc_id="",
                    snippet="[Unverified citation: page reference not found in retrieved context]",
                    is_verified=False,
                    raw_reference=raw_token,
                )
            )

    return found_citations

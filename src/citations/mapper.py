"""Citation mapping and evidence card extraction engine."""

import re

from loguru import logger

from src.citations.locator import PDFPageLocator
from src.core.models import EvidenceCard, EvidenceMap, RetrievalResult


class EvidenceMapper:
    """Extracts, verifies, and maps citations to structured evidence cards."""

    # Matches patterns like [Page 1, handbook.pdf], [Page 2], [handbook.pdf:p2], [doc_id:xxx, p1]
    CITATION_PATTERN = re.compile(
        r"\[(?:Page\s*(\d+)(?:,\s*([^\]]+))?|([^\]]+?)(?::p|\s*,\s*Page\s*)(\d+))\]",
        re.IGNORECASE,
    )

    def __init__(self, locator: PDFPageLocator | None = None) -> None:
        self.locator = locator or PDFPageLocator()

    def map_evidence(
        self,
        answer: str,
        retrieved_chunks: list[RetrievalResult],
        query: str | None = None,
    ) -> EvidenceMap:
        """Map citations from answer text against retrieved evidence chunks.

        1. Discovers all citation candidate tokens in the answer.
        2. Resolves candidates by doc_id (preferred) or filename + page against retrieved chunks.
        3. Validates page bounds against chunk total_pages metadata.
        4. Produces structured EvidenceCards with PDF anchor fragments.
        5. Normalizes answer text with numbered citation markers: [1], [2]...
        """
        if not answer.strip() or not retrieved_chunks:
            return EvidenceMap(
                evidence_cards=[],
                annotated_answer=answer,
                cited_pages_by_document={},
                unverified_references=[],
                total_citations=0,
                verified_count=0,
            )

        # Index retrieved chunks by (doc_id, page), (chunk_id), and (filename, page)
        chunk_by_doc_page: dict[tuple[str, int], list[RetrievalResult]] = {}
        chunk_by_file_page: dict[tuple[str, int], list[RetrievalResult]] = {}
        chunk_by_page: dict[int, list[RetrievalResult]] = {}

        for chunk in retrieved_chunks:
            doc_id = chunk.doc_id
            page = chunk.page
            fname = chunk.filename.lower()

            chunk_by_doc_page.setdefault((doc_id, page), []).append(chunk)
            chunk_by_file_page.setdefault((fname, page), []).append(chunk)
            chunk_by_page.setdefault(page, []).append(chunk)

        evidence_cards: list[EvidenceCard] = []
        unverified_refs: list[str] = []
        cited_pages: dict[str, set[int]] = {}

        # Track replacements to normalize answer text
        replacements: list[tuple[str, str]] = []
        seen_card_keys: dict[tuple[str, int], int] = {}  # (doc_id, page) -> citation_index

        for match in self.CITATION_PATTERN.finditer(answer):
            raw_token = match.group(0)
            p1, id_or_file1, id_or_file2, p2 = match.groups()

            page_num = int(p1) if p1 else int(p2)
            identifier = (id_or_file1 or id_or_file2 or "").strip()

            matched_chunk: RetrievalResult | None = None

            # 1. Match by doc_id + page (highest priority)
            if identifier and (identifier, page_num) in chunk_by_doc_page:
                matched_chunk = chunk_by_doc_page[(identifier, page_num)][0]

            # 2. Match by filename + page (handles duplicate filenames across doc_ids safely)
            elif identifier:
                ident_lower = identifier.lower()
                for (fname, p), chunks in chunk_by_file_page.items():
                    if p == page_num and (ident_lower in fname or fname in ident_lower):
                        matched_chunk = chunks[0]
                        break

            # 3. Match by page alone if only one document in retrieved chunks contains that page
            if matched_chunk is None and not identifier:
                candidates = chunk_by_page.get(page_num, [])
                if candidates:
                    # Pick candidate with highest similarity score
                    matched_chunk = sorted(candidates, key=lambda c: c.score, reverse=True)[0]

            if matched_chunk is not None:
                card_key = (matched_chunk.doc_id, matched_chunk.page)
                if card_key in seen_card_keys:
                    card_idx = seen_card_keys[card_key]
                else:
                    card_idx = len(evidence_cards) + 1
                    seen_card_keys[card_key] = card_idx

                    total_pg = int(matched_chunk.metadata.get("total_pages", matched_chunk.page))
                    is_valid_page = self.locator.validate_page_bounds(matched_chunk.page, total_pg)

                    snippet = self.locator.extract_preview_snippet(
                        text=matched_chunk.content,
                        query=query,
                        max_chars=180,
                    )

                    card = EvidenceCard(
                        citation_index=card_idx,
                        doc_id=matched_chunk.doc_id,
                        filename=matched_chunk.filename,
                        page=matched_chunk.page,
                        total_pages=total_pg,
                        chunk_id=matched_chunk.chunk_id,
                        chunk_index=matched_chunk.chunk_index,
                        snippet=snippet,
                        similarity_score=matched_chunk.score,
                        distance=matched_chunk.distance,
                        page_url_fragment=self.locator.build_page_url_fragment(matched_chunk.page),
                        is_verified=is_valid_page,
                        raw_reference=raw_token,
                    )
                    evidence_cards.append(card)
                    cited_pages.setdefault(matched_chunk.filename, set()).add(matched_chunk.page)

                replacements.append((raw_token, f"[{card_idx}]"))

            else:
                unverified_refs.append(raw_token)
                logger.warning("Unverified citation detected in answer: '{}'", raw_token)

        # Build normalized annotated answer
        annotated_answer = answer
        for original, normalized in replacements:
            annotated_answer = annotated_answer.replace(original, normalized)

        # Format sorted pages mapping
        sorted_cited_pages = {k: sorted(list(v)) for k, v in cited_pages.items()}

        return EvidenceMap(
            evidence_cards=evidence_cards,
            annotated_answer=annotated_answer,
            cited_pages_by_document=sorted_cited_pages,
            unverified_references=unverified_refs,
            total_citations=len(replacements) + len(unverified_refs),
            verified_count=len(evidence_cards),
        )

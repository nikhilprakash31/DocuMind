"""PDF page navigation, coordinate calculation, and preview excerpt extraction."""

import re


class PDFPageLocator:
    """Helper utilities for PDF coordinate resolution and excerpt formatting."""

    @staticmethod
    def validate_page_bounds(page: int, total_pages: int) -> bool:
        """Verify that page number is positive and within document page count."""
        return 1 <= page <= total_pages

    @staticmethod
    def build_page_url_fragment(page: int) -> str:
        """Generate PDF viewer URL anchor fragment (e.g. #page=2)."""
        return f"#page={max(1, page)}"

    @staticmethod
    def extract_preview_snippet(
        text: str,
        query: str | None = None,
        max_chars: int = 180,
    ) -> str:
        """Extract a coherent excerpt snippet from chunk text, prioritizing query keywords."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        if len(cleaned) <= max_chars:
            return cleaned

        if not query:
            return cleaned[:max_chars].rstrip() + "..."

        # Look for the first occurrence of any substantive query term (> 3 chars)
        terms = [re.escape(w.lower()) for w in re.findall(r"\w+", query) if len(w) > 3]
        if not terms:
            return cleaned[:max_chars].rstrip() + "..."

        pattern = re.compile(r"\b(" + "|".join(terms) + r")\b", re.IGNORECASE)
        match = pattern.search(cleaned)

        if not match:
            return cleaned[:max_chars].rstrip() + "..."

        # Center excerpt around match
        start_idx = max(0, match.start() - 40)
        end_idx = min(len(cleaned), start_idx + max_chars)

        snippet = cleaned[start_idx:end_idx].strip()
        prefix = "..." if start_idx > 0 else ""
        suffix = "..." if end_idx < len(cleaned) else ""

        return f"{prefix}{snippet}{suffix}"

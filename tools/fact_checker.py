"""
Fact Checker Tool

Verifies a financial claim by checking keyword overlap against
provided sources or web search results.
"""

import re
from tools.web_search import web_search


# ─── Configuration ────────────────────────────────────────────────────────────

# Words to ignore when matching claim against evidence
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "and", "but", "or", "not",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "you", "your", "he", "she", "his", "her",
    "than", "more", "most", "very", "also", "just", "over", "up",
}


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text, removing stop words."""
    words = re.findall(r"[a-zA-Z0-9]+\.?[0-9]*%?", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def _compute_overlap(claim_keywords: set, evidence_keywords: set) -> float:
    """Compute the fraction of claim keywords found in the evidence."""
    if not claim_keywords:
        return 0.0
    overlap = claim_keywords & evidence_keywords
    return len(overlap) / len(claim_keywords)


def fact_checker(claim: str, sources: list[str] = None) -> dict:
    """
    Verify a financial claim against provided sources or web search.

    Args:
        claim:   The claim to verify (e.g. "Apple revenue grew 8% in Q1 2025").
        sources: Optional list of text evidence to check against.
                 If not provided, web_search is used to fetch evidence.

    Returns:
        Dict with verification status, evidence, and confidence score.
    """
    try:
        claim_keywords = _extract_keywords(claim)

        if not claim_keywords:
            return {"error": "Could not extract meaningful keywords from claim."}

        # --- Gather evidence ---
        evidence_texts = []

        if sources:
            # Use provided sources directly
            evidence_texts = sources
        else:
            # Search the web for evidence
            search_results = web_search(claim, num_results=5)

            if isinstance(search_results, dict) and "error" in search_results:
                return {"error": f"Web search failed: {search_results['error']}"}

            if isinstance(search_results, list):
                for r in search_results:
                    text = f"{r.get('title', '')} {r.get('snippet', '')}"
                    evidence_texts.append(text)

        if not evidence_texts:
            return {
                "claim": claim,
                "status": "unverified",
                "evidence": [],
                "confidence": 0.0,
            }

        # --- Check each evidence source ---
        matches = []
        best_score = 0.0

        for evidence in evidence_texts:
            evidence_keywords = _extract_keywords(evidence)
            overlap = _compute_overlap(claim_keywords, evidence_keywords)

            if overlap > 0.3:  # At least 30% keyword overlap
                matches.append({
                    "text": evidence[:300],
                    "overlap_score": round(overlap, 3),
                })

            if overlap > best_score:
                best_score = overlap

        # --- Determine status ---
        confidence = round(best_score, 3)

        if confidence >= 0.6:
            status = "verified"
        elif confidence >= 0.3:
            status = "partially_verified"
        else:
            status = "unverified"

        return {
            "claim": claim,
            "status": status,
            "evidence": matches[:5],
            "confidence": confidence,
        }

    except Exception as e:
        return {"error": f"Fact check failed: {e}"}

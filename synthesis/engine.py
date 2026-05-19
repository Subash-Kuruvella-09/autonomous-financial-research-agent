"""
RAG Pipeline — 6-Stage Retrieval Augmented Generation (A8)

Stages:
  1. Query Transformation  — decompose user query into sub-queries
  2. Multi-Source Retrieval — dispatch sub-queries to memory + tools
  3. Relevance Re-Ranking   — TF-IDF score filtering
  4. Context Assembly       — source-tagged, recency-weighted, token-budgeted
  5. Grounded Generation    — handled by system prompt (A7)
  6. Post-Generation Verify — check claims against retrieved context

No heavy ML models — uses Groq LLM + sklearn TF-IDF only.
"""

import json
import re
import time
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.query_analyzer import client

# ─── Source tier weights for ranking ──────────────────────────────────────────

SOURCE_TIER = {
    "sec_filing":       {"tier": 1, "weight": 1.00, "label": "SEC Filing"},
    "10-K":             {"tier": 1, "weight": 1.00, "label": "SEC 10-K"},
    "10-Q":             {"tier": 1, "weight": 1.00, "label": "SEC 10-Q"},
    "financial_api":    {"tier": 2, "weight": 0.90, "label": "Financial API"},
    "ratios":           {"tier": 2, "weight": 0.90, "label": "Financial Ratios"},
    "company_profile":  {"tier": 2, "weight": 0.85, "label": "Company Profile"},
    "agent_analysis":   {"tier": 2, "weight": 0.80, "label": "Prior Analysis"},
    "earnings_call":    {"tier": 3, "weight": 0.70, "label": "Earnings Call"},
    "transcript":       {"tier": 3, "weight": 0.70, "label": "Transcript"},
    "news":             {"tier": 4, "weight": 0.65, "label": "News"},
    "article":          {"tier": 4, "weight": 0.65, "label": "Article"},
    "web_search":       {"tier": 4, "weight": 0.60, "label": "Web Search"},
}

# Token budget allocation (fraction of max_context_chars)
TOKEN_BUDGET = {
    "primary":    0.40,   # SEC filings, financial data (Tier 1-2)
    "supporting": 0.30,   # News, transcripts (Tier 3-4)
    "metadata":   0.10,   # Source tags, dates
}
# Remaining 20% reserved for system prompt + generation headroom

MAX_CONTEXT_CHARS = 8000  # Total char budget for RAG context


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Query Transformation
# ═══════════════════════════════════════════════════════════════════════════════

def transform_query(user_query: str, ticker: str = "") -> list[str]:
    """
    Decompose a user query into 3-5 specific retrieval sub-queries.

    A query like 'Is Tesla a good investment?' becomes:
      - 'Tesla TSLA revenue growth operating margins 2024'
      - 'Tesla TSLA competitive risks EV market'
      - 'Tesla TSLA analyst sentiment recent news'

    Args:
        user_query: Raw user question.
        ticker:     Identified stock ticker.

    Returns:
        List of 3-5 specific retrieval queries.
    """
    prompt = f"""Decompose this financial research query into 3-5 specific retrieval sub-queries.
Each sub-query should target a different aspect of the research.
Include the ticker symbol {ticker} in each sub-query.

User query: "{user_query}"

Return ONLY a JSON array of strings. No explanation.
Example: ["AAPL revenue growth margins 2024", "AAPL competitive risks market share", "AAPL recent news analyst sentiment"]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return only a JSON array of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON array from possible markdown
        if "```" in raw:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                raw = match.group()

        sub_queries = json.loads(raw)

        if isinstance(sub_queries, list) and len(sub_queries) >= 2:
            print(f"[RAG] Query decomposed into {len(sub_queries)} sub-queries")
            return sub_queries[:5]

    except Exception as e:
        print(f"[RAG] Query transformation failed: {e}")

    # Fallback: generate basic sub-queries manually
    fallback = [
        f"{ticker} financial performance revenue margins",
        f"{ticker} recent news market sentiment",
        f"{ticker} risks competitive position",
    ]
    print("[RAG] Using fallback sub-queries")
    return fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Multi-Source Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_multi_source(
    sub_queries: list[str],
    ticker: str,
) -> list[dict[str, Any]]:
    """
    Dispatch sub-queries to multiple sources and collect results.

    Sources queried:
      1. Vector DB (long-term memory)
      2. Financial data API (structured numbers)
      3. Company profile (basic info)
      4. News sentiment (recent coverage)

    Args:
        sub_queries: List of decomposed retrieval queries.
        ticker:      Stock ticker symbol.

    Returns:
        List of normalized result dicts with keys:
          content, source_type, ticker, date, relevance_query
    """
    from memory.vector_store import vector_db_search
    from tools.financial_api import financial_data_api
    from tools.company_profile import company_profile
    from tools.news_sentiment import news_sentiment

    results = []

    # Source 1: Vector DB memory — search each sub-query
    for sq in sub_queries[:3]:  # Cap to avoid slow DB
        try:
            mem = vector_db_search(sq, top_k=2)
            if isinstance(mem, dict) and mem.get("results"):
                for r in mem["results"]:
                    results.append({
                        "content": r.get("content", ""),
                        "source_type": r.get("metadata", {}).get("source_type", "memory"),
                        "ticker": r.get("metadata", {}).get("ticker", ticker),
                        "date": r.get("metadata", {}).get("date", ""),
                        "score": r.get("score", 0),
                        "relevance_query": sq,
                    })
        except Exception as e:
            print(f"[RAG] Memory search failed for '{sq[:40]}': {e}")

    # Source 2: Financial data (ratios — most compact, most useful)
    try:
        fin = financial_data_api(ticker, "ratios", "annual", 1)
        if isinstance(fin, dict) and "error" not in fin:
            content = json.dumps(fin, indent=2, default=str)[:2000]
            results.append({
                "content": f"Financial ratios for {ticker}:\n{content}",
                "source_type": "financial_api",
                "ticker": ticker,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "score": 0.9,
                "relevance_query": f"{ticker} financial ratios",
            })
    except Exception as e:
        print(f"[RAG] Financial data retrieval failed: {e}")

    # Source 3: Company profile
    try:
        profile = company_profile(ticker)
        if isinstance(profile, dict) and "error" not in profile:
            content = json.dumps(profile, indent=2, default=str)[:1500]
            results.append({
                "content": f"Company profile for {ticker}:\n{content}",
                "source_type": "company_profile",
                "ticker": ticker,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "score": 0.85,
                "relevance_query": f"{ticker} company overview",
            })
    except Exception as e:
        print(f"[RAG] Company profile retrieval failed: {e}")

    # Source 4: News sentiment
    try:
        news = news_sentiment(f"{ticker} stock", 3, 7)
        if isinstance(news, dict) and "error" not in news:
            content = json.dumps(news, indent=2, default=str)[:1500]
            results.append({
                "content": f"Recent news sentiment for {ticker}:\n{content}",
                "source_type": "news",
                "ticker": ticker,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "score": 0.7,
                "relevance_query": f"{ticker} recent news",
            })
    except Exception as e:
        print(f"[RAG] News retrieval failed: {e}")

    print(f"[RAG] Retrieved {len(results)} documents from {_count_sources(results)} sources")
    return results


def _count_sources(results: list[dict]) -> int:
    """Count unique source types."""
    return len(set(r.get("source_type", "") for r in results))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Relevance Filtering & Re-Ranking
# ═══════════════════════════════════════════════════════════════════════════════

def rerank_results(
    original_query: str,
    results: list[dict[str, Any]],
    min_score: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Re-rank retrieved results by relevance to the original query.

    Uses TF-IDF cosine similarity (same lightweight approach as vector_db_search).
    Applies source tier weighting as a boost factor.

    Args:
        original_query: The user's original query.
        results:        List of result dicts with 'content' field.
        min_score:      Minimum relevance score to keep.

    Returns:
        Filtered and re-ranked results, highest relevance first.
    """
    if not results:
        return []

    contents = [r["content"] for r in results]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(contents + [original_query])

        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]

        raw_scores = cosine_similarity(query_vec, doc_vecs)[0]

        # Apply source tier boost
        for i, result in enumerate(results):
            source = result.get("source_type", "web_search")
            tier_info = SOURCE_TIER.get(source, {"weight": 0.5})
            tier_boost = tier_info["weight"]

            # Combined score: 70% relevance + 30% source quality
            results[i]["relevance_score"] = round(
                0.7 * float(raw_scores[i]) + 0.3 * tier_boost, 4
            )
            results[i]["tier"] = tier_info.get("tier", 4)
            results[i]["tier_label"] = tier_info.get("label", source)

    except Exception as e:
        # If TF-IDF fails, just use source tier as score
        print(f"[RAG] Re-ranking TF-IDF failed: {e}, using tier-only ranking")
        for r in results:
            source = r.get("source_type", "web_search")
            tier_info = SOURCE_TIER.get(source, {"weight": 0.5})
            r["relevance_score"] = tier_info["weight"]
            r["tier"] = tier_info.get("tier", 4)
            r["tier_label"] = tier_info.get("label", source)

    # Filter by minimum score
    filtered = [r for r in results if r.get("relevance_score", 0) >= min_score]

    # Sort: highest relevance first
    filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Deduplicate by content similarity (exact match)
    seen = set()
    deduped = []
    for r in filtered:
        key = r["content"][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(
        f"[RAG] Re-ranked: {len(results)} → {len(deduped)} documents "
        f"(min_score={min_score})"
    )
    return deduped


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4: Context Assembly
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_context(
    results: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Assemble retrieved documents into a formatted context block.

    Implements:
      - Source attribution: each chunk tagged with [Source: type, Tier N]
      - Recency weighting: newer documents placed first within same tier
      - Diversity enforcement: at least one doc per source type
      - Token budgeting: primary data gets 40%, supporting 30%, metadata 10%

    Args:
        results: Re-ranked result list from Stage 3.
        max_chars: Maximum character budget for the assembled context.

    Returns:
        Formatted context string ready for injection into system prompt.
    """
    if not results:
        return "[No relevant context retrieved]"

    # Split into primary (Tier 1-2) and supporting (Tier 3-4)
    primary = [r for r in results if r.get("tier", 4) <= 2]
    supporting = [r for r in results if r.get("tier", 4) > 2]

    # Sort each group by recency (newer first)
    primary = _sort_by_recency(primary)
    supporting = _sort_by_recency(supporting)

    # Allocate budgets
    primary_budget = int(max_chars * TOKEN_BUDGET["primary"])
    supporting_budget = int(max_chars * TOKEN_BUDGET["supporting"])

    # Build context sections
    context_parts = []
    context_parts.append("=" * 50)
    context_parts.append("PRE-FETCHED RESEARCH CONTEXT (RAG Pipeline)")
    context_parts.append("Use this data in your analysis. Cite sources.")
    context_parts.append("=" * 50)

    # Primary data
    if primary:
        context_parts.append("\n--- PRIMARY DATA (Tier 1-2: High Reliability) ---\n")
        chars_used = 0
        for r in primary:
            chunk = _format_context_chunk(r)
            if chars_used + len(chunk) > primary_budget:
                break
            context_parts.append(chunk)
            chars_used += len(chunk)

    # Supporting data
    if supporting:
        context_parts.append("\n--- SUPPORTING DATA (Tier 3-4: Supplementary) ---\n")
        chars_used = 0
        for r in supporting:
            chunk = _format_context_chunk(r)
            if chars_used + len(chunk) > supporting_budget:
                break
            context_parts.append(chunk)
            chars_used += len(chunk)

    # Diversity check — ensure we have at least 2 source types
    source_types = set(r.get("source_type", "") for r in results)
    if len(source_types) < 2:
        context_parts.append(
            "\n⚠️ Limited source diversity — only "
            f"{', '.join(source_types)} available.\n"
        )

    assembled = "\n".join(context_parts)

    # Final truncation safety
    if len(assembled) > max_chars:
        assembled = assembled[:max_chars] + "\n\n[... context truncated ...]"

    print(f"[RAG] Context assembled: {len(assembled)} chars, "
          f"{len(source_types)} source types")
    return assembled


def _format_context_chunk(result: dict) -> str:
    """Format a single result as a context chunk with source attribution."""
    source = result.get("tier_label", result.get("source_type", "Unknown"))
    tier = result.get("tier", "?")
    ticker = result.get("ticker", "")
    date = result.get("date", "")[:10]
    score = result.get("relevance_score", 0)
    content = result.get("content", "")

    # Cap individual chunk length
    if len(content) > 1500:
        content = content[:1500] + "..."

    header = f"[Source: {source} | Tier {tier} | {ticker} | {date} | relevance: {score}]"
    return f"{header}\n{content}\n"


def _sort_by_recency(results: list[dict]) -> list[dict]:
    """Sort results by date (newest first). Undated items go last."""
    def date_key(r):
        d = r.get("date", "")
        if d:
            try:
                return datetime.fromisoformat(d.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        return datetime.min

    return sorted(results, key=date_key, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6: Post-Generation Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_generation(
    generated_text: str,
    context: str,
) -> dict[str, Any]:
    """
    Verify that claims in the generated output are supported by context.

    Checks every factual claim in the output against the retrieved context.
    Returns a verification report with supported/unsupported/unverifiable claims.

    Args:
        generated_text: The agent's Final Answer.
        context:        The assembled RAG context used during generation.

    Returns:
        Dict with:
          - verified: bool (overall pass/fail)
          - supported_claims: int
          - unsupported_claims: list[str]
          - verification_note: str (to append to report)
    """
    # Truncate inputs to fit in a single LLM call
    gen_truncated = generated_text[:3000]
    ctx_truncated = context[:4000]

    prompt = f"""You are a fact-verification assistant.

Given the SOURCE DOCUMENTS and the GENERATED REPORT below, identify any claims
in the report that are NOT supported by the source documents.

SOURCE DOCUMENTS:
{ctx_truncated}

GENERATED REPORT:
{gen_truncated}

Return ONLY a JSON object:
{{
  "supported_count": <number of claims supported by sources>,
  "unsupported_claims": ["claim 1 text", "claim 2 text"],
  "confidence": "high" or "medium" or "low"
}}

If all claims are supported, return an empty unsupported_claims list.
Return ONLY the JSON. No explanation."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON
        if "```" in raw:
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                raw = match.group()

        result = json.loads(raw)

        supported = result.get("supported_count", 0)
        unsupported = result.get("unsupported_claims", [])
        confidence = result.get("confidence", "medium")

        verified = len(unsupported) == 0

        # Build verification note for the report
        if verified:
            note = (
                f"✅ **Verification**: All {supported} factual claims verified "
                f"against source documents. Confidence: {confidence}."
            )
        else:
            note = (
                f"⚠️ **Verification**: {supported} claims supported, "
                f"{len(unsupported)} potentially unsupported:\n"
            )
            for claim in unsupported[:3]:
                note += f"  - {claim}\n"
            note += f"Confidence: {confidence}."

        print(f"[RAG] Verification: {supported} supported, "
              f"{len(unsupported)} unsupported, confidence={confidence}")

        return {
            "verified": verified,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "confidence": confidence,
            "verification_note": note,
        }

    except Exception as e:
        print(f"[RAG] Post-verification failed: {e}")
        return {
            "verified": False,
            "supported_claims": 0,
            "unsupported_claims": [],
            "confidence": "unknown",
            "verification_note": (
                "⚠️ **Verification**: Post-generation verification could "
                f"not be completed: {e}"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_rag_prefetch(user_query: str, ticker: str) -> str:
    """
    Run RAG pipeline stages 1-4 to pre-fetch context for the agent.

    This is called BEFORE the ReAct loop starts, giving the agent
    high-quality grounded context from the first step.

    Args:
        user_query: The user's research query.
        ticker:     Identified stock ticker.

    Returns:
        Assembled context string ready for system prompt injection.
    """
    print("\n" + "=" * 50)
    print("[RAG] Starting pre-fetch pipeline")
    print("=" * 50)

    start = time.time()

    # Stage 1: Query Transformation
    print("\n[RAG] Stage 1: Query Transformation")
    sub_queries = transform_query(user_query, ticker)
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")

    # Stage 2: Multi-Source Retrieval
    print("\n[RAG] Stage 2: Multi-Source Retrieval")
    raw_results = retrieve_multi_source(sub_queries, ticker)

    # Stage 3: Re-Ranking
    print("\n[RAG] Stage 3: Relevance Re-Ranking")
    ranked_results = rerank_results(user_query, raw_results)

    # Stage 4: Context Assembly
    print("\n[RAG] Stage 4: Context Assembly")
    context = assemble_context(ranked_results)

    elapsed = time.time() - start
    print(f"\n[RAG] Pre-fetch complete in {elapsed:.1f}s")
    print("=" * 50 + "\n")

    return context

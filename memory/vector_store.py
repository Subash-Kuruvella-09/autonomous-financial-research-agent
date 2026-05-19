"""
Vector DB Search Tool — Enhanced Memory Architecture

Searches the agent's long-term memory using TF-IDF + cosine similarity.
Supports rich metadata filtering: ticker, source_type, date, confidence, verified.

No heavy ML models — uses only sklearn.
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "vector_db.json")


def vector_db_search(
    query: str,
    top_k: int = 3,
    filter: dict = None,
    min_confidence: float = 0.0,
    verified_only: bool = False,
) -> dict:
    """
    Search stored research data using TF-IDF similarity.

    Supports metadata filtering by ticker, source_type, date, etc.
    and quality gates via confidence threshold and verification status.

    Args:
        query:           Search text.
        top_k:           Number of results to return.
        filter:          Optional metadata filter (e.g. {"ticker": "AAPL"}).
        min_confidence:  Minimum confidence score (0-1) to include results.
        verified_only:   If True, only return fact-checked records.

    Returns:
        {"query": str, "results": [...]} or {"error": str}.
    """
    try:
        if not os.path.exists(DB_FILE):
            return {"query": query, "results": []}

        with open(DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)

        if not db:
            return {"query": query, "results": []}

        # --- Apply metadata filters ---
        filtered = db

        # Key-value metadata match
        if filter:
            filtered = [
                r for r in filtered
                if all(
                    r.get("metadata", {}).get(k) == v
                    for k, v in filter.items()
                )
            ]

        # Confidence threshold
        if min_confidence > 0:
            filtered = [
                r for r in filtered
                if r.get("metadata", {}).get("confidence", 0) >= min_confidence
            ]

        # Verified-only filter
        if verified_only:
            filtered = [
                r for r in filtered
                if r.get("metadata", {}).get("verified") is True
            ]

        if not filtered:
            return {"query": query, "results": []}

        # --- Build TF-IDF matrix over stored docs + query ---
        contents = [r["content"] for r in filtered]

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(contents + [query])

        # Query vector is the last row
        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]

        # --- Compute cosine similarity ---
        scores = cosine_similarity(query_vec, doc_vecs)[0]

        # --- Rank and return top_k unique results ---
        top_indices = np.argsort(scores)[::-1]

        results = []
        seen_content = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            content = filtered[idx]["content"]

            # Deduplicate by content
            if content in seen_content:
                continue
            seen_content.add(content)

            meta = filtered[idx].get("metadata", {})
            results.append({
                "content": content,
                "metadata": {
                    "ticker": meta.get("ticker", ""),
                    "source_type": meta.get("source_type", ""),
                    "date": meta.get("date", ""),
                    "confidence": meta.get("confidence", 0),
                    "verified": meta.get("verified", False),
                },
                "score": round(float(scores[idx]), 4),
            })

            if len(results) >= top_k:
                break

        return {"query": query, "results": results}

    except Exception as e:
        return {"error": f"Search failed: {e}"}


def get_memory_stats() -> dict:
    """Get summary statistics about what's stored in memory."""
    try:
        if not os.path.exists(DB_FILE):
            return {"total_records": 0}

        with open(DB_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)

        tickers = set()
        source_types = set()
        verified_count = 0

        for record in db:
            meta = record.get("metadata", {})
            if meta.get("ticker"):
                tickers.add(meta["ticker"])
            if meta.get("source_type"):
                source_types.add(meta["source_type"])
            if meta.get("verified"):
                verified_count += 1

        return {
            "total_records": len(db),
            "tickers_covered": sorted(tickers),
            "source_types": sorted(source_types),
            "verified_records": verified_count,
        }

    except Exception:
        return {"total_records": 0}
"""
Vector DB Store Tool — Enhanced Memory Architecture

Stores research data into local long-term memory with enriched schema:
  - id, content, ticker, source_type, date
  - confidence, verified, researcher_session
  - Semantic chunking for different document types

No heavy ML models — TF-IDF embeddings computed at search time.
"""

import uuid
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "vector_db.json")
EPISODIC_FILE = os.path.join(os.path.dirname(__file__), "..", "episodic_memory.json")


# ─── Chunking Strategies ─────────────────────────────────────────────────────

def _chunk_transcript(content: str) -> list[str]:
    """Chunk earnings transcripts by speaker turns."""
    # Split on common speaker patterns (e.g. "CEO:", "Operator:", "Analyst:")
    import re
    turns = re.split(r'\n(?=[A-Z][a-zA-Z\s\.]+:)', content)
    chunks = []
    for turn in turns:
        turn = turn.strip()
        if len(turn) > 50:  # skip very short fragments
            chunks.append(turn[:1500])  # cap each chunk
    return chunks if chunks else [content[:3000]]


def _chunk_filing(content: str) -> list[str]:
    """Chunk SEC filings by section headers."""
    import re
    sections = re.split(r'\n(?=(?:Item|ITEM|Part|PART)\s+\d)', content)
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) > 100:
            chunks.append(section[:2000])
    return chunks if chunks else [content[:3000]]


def _chunk_news(content: str) -> list[str]:
    """Chunk news articles by paragraph."""
    paragraphs = content.split("\n\n")
    # Include headline context in each chunk
    headline = paragraphs[0][:200] if paragraphs else ""
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50:
            chunk = f"{headline}\n{para}" if para != headline else para
            chunks.append(chunk[:1500])
    return chunks if chunks else [content[:3000]]


def _chunk_default(content: str) -> list[str]:
    """Default chunking: split by sentences into ~500 char chunks."""
    if len(content) <= 500:
        return [content]

    import re
    sentences = re.split(r'(?<=[.!?])\s+', content)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > 500:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = f"{current} {sent}" if current else sent
    if current:
        chunks.append(current.strip())

    return chunks if chunks else [content[:3000]]


CHUNKERS = {
    "earnings_call": _chunk_transcript,
    "transcript": _chunk_transcript,
    "10-K": _chunk_filing,
    "10-Q": _chunk_filing,
    "8-K": _chunk_filing,
    "sec_filing": _chunk_filing,
    "news": _chunk_news,
    "article": _chunk_news,
}


from memory.episodic import record_episode

# ─── Main Store Function ─────────────────────────────────────────────────────

def vector_db_store(
    content: str,
    metadata: dict,
    confidence: float = 0.8,
    verified: bool = False,
    session_id: str = None,
) -> dict:
    """
    Store new research data into local long-term memory.

    Applies intelligent chunking based on source_type, and stores
    each chunk with enriched metadata per the A3.3 schema spec.

    Args:
        content:    Text to store.
        metadata:   Dict with keys: ticker, date, source_type.
        confidence: Agent's confidence in this info (0-1). Default 0.8.
        verified:   Whether fact-checked against multiple sources.
        session_id: Research session ID (auto-generated if not provided).

    Returns:
        {"status": "stored", "document_id": str, "chunks_stored": int}
        or {"error": str}.
    """
    try:
        # Load existing DB
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r", encoding="utf-8") as f:
                db = json.load(f)
        else:
            db = []

        # Generate session ID if not provided
        if not session_id:
            session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Determine source type for chunking
        source_type = metadata.get("source_type", "default")
        ticker = metadata.get("ticker", "UNKNOWN")

        # Apply intelligent chunking
        chunker = CHUNKERS.get(source_type, _chunk_default)
        chunks = chunker(content)

        # Store each chunk with full schema
        doc_ids = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{ticker.lower()}-{source_type}-{uuid.uuid4().hex[:8]}"

            record = {
                "id": doc_id,
                "content": chunk,
                "metadata": {
                    "ticker": ticker.upper(),
                    "source_type": source_type,
                    "date": metadata.get("date", datetime.now().isoformat()),
                    "confidence": confidence,
                    "verified": verified,
                    "researcher_session": session_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            db.append(record)
            doc_ids.append(doc_id)

        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)

        # Record episodic memory
        record_episode("vector_db_store", True, source_type)

        return {
            "status": "stored",
            "document_id": doc_ids[0] if len(doc_ids) == 1 else doc_ids,
            "chunks_stored": len(doc_ids),
        }

    except Exception as e:
        record_episode("vector_db_store", False)
        return {"error": f"Failed to store: {e}"}

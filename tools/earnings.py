"""
Earnings Call Transcript Retrieval Tool

Fetches and parses earnings call transcripts from the web
for a given company ticker, quarter, and year.
"""

import re
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


# ─── Domain & URL Configuration ──────────────────────────────────────────────

PREFERRED_DOMAINS = [
    "fool.com/earnings/call-transcript",
    "fool.com",
    "alphastreet.com",
    "nasdaq.com",
    "marketwatch.com",
    "seekingalpha.com",
]

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# URL fragments that indicate low-quality / non-transcript pages
BLOCKED_URL_PATTERNS = [
    "discussion",
    "forum",
    "reddit",
    "comments",
    "community",
    "boards",
    "thread",
    "how-to",
    "guide",
    "what-is",
    "opinion",
    "blog",
]


# ─── Content Rejection ───────────────────────────────────────────────────────

# Phrases in the page text that indicate it is NOT a transcript
REJECT_CONTENT_PHRASES = [
    "how to read",
    "step-by-step",
    "guide for investors",
    "what is an earnings call",
    "how to analyze",
    "tutorial",
    "beginner",
    "cheat sheet",
    "for dummies",
    "tips for reading",
]

# Phrases in the URL that suggest the page is a transcript
TRANSCRIPT_URL_HINTS = [
    "transcript",
    "earnings-call-transcript",
    "call-transcript",
]


# ─── Noise Filtering ─────────────────────────────────────────────────────────

NOISE_PHRASES = [
    # Operator / moderator noise
    "next question",
    "operator signoff",
    "operator, may we",
    "operator, can we",
    "operator, could we",
    "limit yourself to",
    "this does conclude",
    # Legal boilerplate
    "forward-looking statements",
    "safe harbor",
    "copyright",
    "all rights reserved",
    # Promotional / ads / footer
    "motley fool",
    "invest better",
    "stock recommendations",
    "portfolio guidance",
    "premium services",
    "premium advisory",
    "disclaimer",
    "sponsored",
    "advertisement",
    "affiliate",
    "disclosure",
    "the views and opinions",
    "not investment advice",
    "past performance",
    "returns as of",
    "fool.com",
    "seeking alpha",
    "join now",
    "free trial",
    "learn more",
    "click here",
    "read more",
    "related articles",
    "trending stocks",
    "editor's note",
    "this article",
    "originally published",
    "subscribe",
    "sign up",
    "newsletter",
    "premium picks",
    # Clickbait / recommendation boilerplate
    "average returns of all recommendations",
    "cost basis and return",
    "warren buffett",
    "forget innovation",
    "greatest contribution",
    "buy or sell before",
    "should you invest",
    "has positions in",
    "an earlier version of this story",
]

# Keywords that signal substantive financial discussion
BUSINESS_KEYWORDS = [
    "revenue", "growth", "margin", "earnings", "profit", "income",
    "guidance", "outlook", "forecast", "quarter", "year-over-year",
    "operating", "segment", "demand", "customers", "services",
    "product", "launch", "shipment", "inventory", "cash flow",
    "capital", "dividend", "buyback", "share", "billion", "million",
    "percent", "%", "increase", "decrease", "strong", "record",
    "pipeline", "investment", "strategy", "market", "opportunity",
]

# Speaker-style patterns that indicate a real transcript
# Matches lines like "Tim Cook -- CEO:", "Operator:", "John Smith -- Analyst:"
SPEAKER_PATTERN = re.compile(
    r"^[A-Z][a-zA-Z\s\.\,\-]{2,40}\s*(?:--|—|:)",
)

# Role keywords that commonly appear as speaker labels
SPEAKER_ROLES = [
    "operator", "ceo", "cfo", "coo", "cto",
    "chief executive", "chief financial", "chief operating",
    "president", "vice president",
    "analyst", "moderator",
    "investor relations",
]


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _is_blocked_url(url: str) -> bool:
    """Return True if the URL points to a forum, discussion, or guide page."""
    lower = url.lower()
    return any(pattern in lower for pattern in BLOCKED_URL_PATTERNS)


def _rank_urls(results: list[dict]) -> list[str]:
    """Return search-result URLs ordered by domain priority, then original rank."""
    prioritized = []
    remaining = []

    for domain in PREFERRED_DOMAINS:
        for result in results:
            url = result.get("href", "")
            if domain in url.lower() and url not in prioritized and not _is_blocked_url(url):
                prioritized.append(url)

    for result in results:
        url = result.get("href", "")
        if url and url not in prioritized and not _is_blocked_url(url):
            remaining.append(url)

    return prioritized + remaining


def _extract_visible_text(html: str) -> list[str]:
    """Parse HTML and extract visible text lines, removing scripts and styles."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = text.splitlines()

    return lines


def _is_noise(line_lower: str) -> bool:
    """Return True if the line matches a known noisy pattern."""
    return any(phrase in line_lower for phrase in NOISE_PHRASES)


def _has_business_content(line_lower: str) -> bool:
    """Return True if the line contains substantive financial keywords."""
    return any(kw in line_lower for kw in BUSINESS_KEYWORDS)


def _count_speaker_lines(lines: list[str]) -> int:
    """Count lines that look like speaker labels in a real transcript."""
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue

        # Check for named speaker pattern (e.g., "Tim Cook -- CEO:")
        if SPEAKER_PATTERN.match(stripped):
            count += 1
            continue

        # Check for role-based labels (e.g., "Operator:", "CEO:")
        lower = stripped.lower()
        for role in SPEAKER_ROLES:
            if lower.startswith(role) and (":" in stripped or "--" in stripped or "—" in stripped):
                count += 1
                break

    return count


def _has_reject_content(raw_lower: str) -> bool:
    """Return True if the page looks like a guide or educational article."""
    return any(phrase in raw_lower for phrase in REJECT_CONTENT_PHRASES)


def _clean_transcript_lines(lines: list[str]) -> list[str]:
    """Filter and clean extracted text lines to isolate transcript content."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 50:
            continue

        lower = stripped.lower()

        # Skip noisy boilerplate
        if _is_noise(lower):
            continue

        # Keep lines that look like speaker dialogue with substance
        has_speaker = ":" in stripped or "--" in stripped or "—" in stripped
        has_substance = _has_business_content(lower)

        if has_speaker and has_substance:
            cleaned.append(stripped)
        elif has_substance and len(stripped) > 80:
            # Long lines with financial keywords are likely from prepared remarks
            cleaned.append(stripped)

    return cleaned


# ─── Main Function ────────────────────────────────────────────────────────────

def earnings_transcript(ticker: str, quarter: str, year: int) -> dict:
    """
    Retrieve an earnings call transcript for a given company.

    Args:
        ticker:  Company ticker symbol (e.g. "AAPL").
        quarter: Fiscal quarter — one of Q1, Q2, Q3, Q4.
        year:    Calendar year (e.g. 2025).

    Returns:
        A dict containing the transcript text and metadata,
        or a dict with an "error" key on failure.
    """
    try:
        # --- Web search via DuckDuckGo (multiple queries for reliability) ---
        search_queries = [
            f"{ticker} {quarter} {year} earnings call transcript",
            f'"{ticker}" "{quarter} {year}" earnings call transcript',
            f"site:fool.com {ticker} {quarter} {year} earnings call transcript",
            f"site:alphastreet.com {ticker} {quarter} {year} earnings call transcript",
        ]

        results = []
        seen_urls = set()
        with DDGS() as ddgs:
            for q in search_queries:
                try:
                    for r in ddgs.text(q, max_results=8):
                        url = r.get("href", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append(r)
                except Exception:
                    continue

        if not results:
            return {"error": "No search results found for the query."}

        ranked_urls = _rank_urls(results)
        if not ranked_urls:
            return {"error": "Could not determine a valid URL from search results."}

        # --- Try each URL until one passes all validation gates ---
        last_error = None
        for source_url in ranked_urls:
            try:
                response = requests.get(source_url, headers=HTTP_HEADERS, timeout=15)
                response.raise_for_status()

                # --- Gate 1: Parse page ---
                raw_lines = _extract_visible_text(response.text)
                raw_text = "\n".join(raw_lines)
                raw_lower = raw_text.lower()

                # --- Gate 2: Year validation ---
                if str(year) not in raw_text:
                    last_error = f"Year {year} not found at {source_url}"
                    continue

                # --- Gate 3: Reject educational / guide content ---
                if _has_reject_content(raw_lower):
                    last_error = f"Educational content at {source_url}"
                    continue

                # --- Gate 4: Transcript structure validation ---
                speaker_count = _count_speaker_lines(raw_lines)
                if speaker_count < 3:
                    last_error = (
                        f"Not a transcript (only {speaker_count} speaker labels) "
                        f"at {source_url}"
                    )
                    continue

                # --- Gate 5: Clean and extract ---
                transcript_lines = _clean_transcript_lines(raw_lines)

                if not transcript_lines:
                    last_error = f"No transcript content after cleaning at {source_url}"
                    continue

                # Limit to first ~200 lines
                transcript_lines = transcript_lines[:200]
                transcript_text = "\n".join(transcript_lines)

                return {
                    "ticker": ticker.upper(),
                    "quarter": quarter.upper(),
                    "year": year,
                    "source_url": source_url,
                    "transcript": transcript_text,
                }

            except requests.exceptions.RequestException as e:
                last_error = f"{source_url} → {e}"
                continue

        return {"error": f"All URLs failed. Last: {last_error}"}

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

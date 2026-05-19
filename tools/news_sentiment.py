"""
News Sentiment Analysis Tool

Searches for recent news articles about a company or topic
and performs simple rule-based sentiment analysis.
"""

from datetime import datetime, timedelta
from ddgs import DDGS


# ─── Sentiment Keywords ──────────────────────────────────────────────────────

POSITIVE_KEYWORDS = [
    "growth", "profit", "record", "strong", "beat", "surge", "bullish",
    "upgrade", "outperform", "rally", "gain", "boost", "expand", "exceed",
    "optimistic", "upbeat", "positive", "soar", "breakthrough", "momentum",
    "recovery", "innovation", "dividend", "buyback", "revenue growth",
    "all-time high", "exceeded expectations", "better than expected",
    "raised guidance", "strong demand",
]

NEGATIVE_KEYWORDS = [
    "loss", "decline", "drop", "risk", "miss", "crash", "bearish",
    "downgrade", "underperform", "plunge", "fall", "cut", "weak", "warning",
    "pessimistic", "negative", "slump", "layoff", "lawsuit", "investigation",
    "recession", "default", "bankruptcy", "fraud", "sell-off", "missed",
    "lower guidance", "disappointing", "below expectations", "concern",
    "volatility", "debt", "shortfall",
]


def _compute_sentiment(text: str) -> tuple[int, str]:
    """
    Compute a simple sentiment score for a piece of text.

    Returns:
        (score, label) where score is +1, 0, or -1
        and label is "positive", "neutral", or "negative".
    """
    lower = text.lower()

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower)

    if pos_count > neg_count:
        return 1, "positive"
    elif neg_count > pos_count:
        return -1, "negative"
    else:
        return 0, "neutral"


def news_sentiment(
    query: str,
    num_articles: int = 10,
    lookback_days: int = 7,
) -> dict:
    """
    Search for recent news and analyze sentiment.

    Args:
        query:          Search query (e.g. "AAPL Apple stock").
        num_articles:   Number of articles to analyze.
        lookback_days:  How many days back to search.

    Returns:
        A dict with articles, sentiment scores, and overall sentiment,
        or {"error": "..."} on failure.
    """
    try:
        # --- Build date-filtered search query ---
        search_query = f"{query} stock news"

        # Calculate date range for DDGS timelimit parameter
        # DDGS supports: "d" (day), "w" (week), "m" (month)
        if lookback_days <= 1:
            timelimit = "d"
        elif lookback_days <= 7:
            timelimit = "w"
        else:
            timelimit = "m"

        # --- URL quality filters ---
        # URL patterns that indicate stock info pages, not news articles
        BLOCKED_URL_PATTERNS = [
            "/quote/", "/quote?", "/quotes/",
            "/market-activity/",
            "/stock-price/",
            "/stock-detail/",
            "/price-history/",
            "/historical-data/",
            "/symbol/",
            "/ticker/",
            "/stocks/",
            "/overview",
        ]

        # --- Search for news ---
        results = []
        seen_titles = set()

        with DDGS() as ddgs:
            search_results = ddgs.text(
                search_query,
                max_results=num_articles * 5,  # fetch extra to filter aggressively
                timelimit=timelimit,
            )

            for r in search_results:
                title = r.get("title", "").strip()
                snippet = r.get("body", "").strip()
                url = r.get("href", "").strip()

                if not title or not snippet:
                    continue

                url_lower = url.lower()

                # Skip stock quote / price / data pages
                if any(pattern in url_lower for pattern in BLOCKED_URL_PATTERNS):
                    continue

                # Skip site homepages (URL path is just "/")
                try:
                    from urllib.parse import urlparse
                    path = urlparse(url).path
                    if path in ("", "/"):
                        continue
                except Exception:
                    pass

                # Skip pages with generic stock-info or homepage titles
                title_lower = title.lower()
                if any(
                    phrase in title_lower
                    for phrase in [
                        "stock price, news, quote",
                        "stock price, quote, news",
                        "stock quote, chart",
                        "real-time quote",
                        "price & news",
                        "market data, news, trading tools",
                        "stock market news - financial news",
                        "price history",
                        "historical data",
                    ]
                ):
                    continue

                # Deduplicate by title
                if title_lower in seen_titles:
                    continue
                seen_titles.add(title_lower)

                # Compute sentiment on combined title + snippet
                combined_text = f"{title} {snippet}"
                score, label = _compute_sentiment(combined_text)

                # Determine if this looks like an actual article
                is_article = (
                    "/news/" in url_lower
                    or "/article/" in url_lower
                    or "/story/" in url_lower
                    or "/press-release/" in url_lower
                    or len(title) > 50  # long descriptive titles = articles
                )

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet[:300],
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "_is_article": is_article,
                })

        # Sort: real articles first, then others
        results.sort(key=lambda x: (not x["_is_article"],))

        # Remove internal flag and trim to requested count
        for r in results:
            r.pop("_is_article", None)
        results = results[:num_articles]

        if not results:
            return {"error": f"No news articles found for '{query}'."}

        # --- Calculate overall sentiment ---
        total_score = sum(a["sentiment_score"] for a in results)
        avg_score = round(total_score / len(results), 2)

        if avg_score > 0.2:
            overall_label = "positive"
        elif avg_score < -0.2:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        return {
            "query": query,
            "num_articles": len(results),
            "lookback_days": lookback_days,
            "articles": results,
            "overall_sentiment": overall_label,
            "score": avg_score,
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

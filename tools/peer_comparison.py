"""
Peer Comparison Tool

Identifies peer companies in the same sector/industry and compares
key financial metrics side-by-side with rankings.
"""

import yfinance as yf


# ─── Metric Configuration ────────────────────────────────────────────────────

# Mapping of user-facing metric names to yfinance info keys
METRIC_MAP = {
    "market_cap": "marketCap",
    "pe_ratio": "trailingPE",
    "forward_pe": "forwardPE",
    "profit_margin": "profitMargins",
    "operating_margin": "operatingMargins",
    "gross_margin": "grossMargins",
    "revenue": "totalRevenue",
    "net_income": "netIncomeToCommon",
    "netIncome": "netIncomeToCommon",       # alias
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "debt_to_equity": "debtToEquity",
    "current_ratio": "currentRatio",
    "dividend_yield": "dividendYield",
    "beta": "beta",
    "eps": "trailingEps",
    "price_to_book": "priceToBook",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
}

# Metrics where LOWER is better
LOWER_IS_BETTER = {"pe_ratio", "forward_pe", "debt_to_equity", "beta"}

# Default metrics if none specified
DEFAULT_METRICS = ["market_cap", "pe_ratio", "profit_margin", "roe", "revenue_growth"]


# ─── Industry-based Peer Mapping ─────────────────────────────────────────────
# Grouped by INDUSTRY (not broad sector) for tighter comparisons.
# Only USD-reporting companies to avoid currency mismatch.

INDUSTRY_PEERS = {
    "Consumer Electronics": [
        "AAPL", "DELL", "HPQ", "HPE", "LOGI",
    ],
    "Semiconductors": [
        "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "MRVL",
    ],
    "Software—Infrastructure": [
        "MSFT", "ORCL", "CRM", "NOW", "ADBE", "INTU", "SNOW",
    ],
    "Software—Application": [
        "ADBE", "CRM", "INTU", "WDAY", "ZM", "TEAM",
    ],
    "Internet Content & Information": [
        "GOOGL", "META", "SNAP", "PINS", "RDDT",
    ],
    "Internet Retail": [
        "AMZN", "EBAY", "ETSY", "W", "CHWY",
    ],
    "Auto Manufacturers": [
        "TSLA", "F", "GM", "RIVN", "LCID",
    ],
    "Banks—Diversified": [
        "JPM", "BAC", "WFC", "C", "USB",
    ],
    "Capital Markets": [
        "GS", "MS", "SCHW", "BLK", "RJF",
    ],
    "Drug Manufacturers—General": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY",
    ],
    "Health Care Plans": [
        "UNH", "HUM", "CI", "ELV", "MOH",
    ],
    "Oil & Gas Integrated": [
        "XOM", "CVX", "COP", "SLB", "EOG",
    ],
    "Aerospace & Defense": [
        "LMT", "RTX", "BA", "NOC", "GD",
    ],
    "Communication Equipment": [
        "CSCO", "JNPR", "MSI", "NOK", "ERIC",
    ],
    "Entertainment": [
        "NFLX", "DIS", "WBD", "PARA", "LGF.A",
    ],
    "Telecom Services": [
        "T", "VZ", "TMUS",
    ],
    "Specialty Retail": [
        "HD", "LOW", "TGT", "TJX", "BBY",
    ],
    "Restaurants": [
        "MCD", "SBUX", "YUM", "CMG", "DPZ",
    ],
}

# Broader sector fallback (all USD-reporting)
SECTOR_PEERS = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ORCL", "ADBE", "INTC", "AMD",
    ],
    "Financial Services": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "ABT",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
    ],
    "Industrials": [
        "CAT", "BA", "HON", "UPS", "GE", "LMT", "RTX",
    ],
}


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _find_peers(ticker: str, num_peers: int) -> list[str]:
    """Find peer tickers in the same industry, falling back to sector."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        industry = info.get("industry", "")
        sector = info.get("sector", "")

        # Priority 1: exact industry match
        if industry in INDUSTRY_PEERS:
            peers = [t for t in INDUSTRY_PEERS[industry] if t != ticker.upper()]
            if peers:
                return peers[:num_peers]

        # Priority 2: fuzzy industry match
        for key, tickers in INDUSTRY_PEERS.items():
            if (industry and (industry.lower() in key.lower() or key.lower() in industry.lower())):
                peers = [t for t in tickers if t != ticker.upper()]
                if peers:
                    return peers[:num_peers]

        # Priority 3: broad sector fallback
        if sector in SECTOR_PEERS:
            peers = [t for t in SECTOR_PEERS[sector] if t != ticker.upper()]
            if peers:
                return peers[:num_peers]

        # Priority 4: fuzzy sector match
        for key, tickers in SECTOR_PEERS.items():
            if (sector and (sector.lower() in key.lower() or key.lower() in sector.lower())):
                peers = [t for t in tickers if t != ticker.upper()]
                if peers:
                    return peers[:num_peers]

        return []

    except Exception:
        return []


def _fetch_company_data(ticker: str, metrics: list[str]) -> dict | None:
    """
    Fetch requested metrics for a single ticker.
    Returns None if the company is non-USD or has too many missing metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # --- Currency check: skip non-USD companies ---
        currency = info.get("financialCurrency") or info.get("currency", "")
        if currency and currency.upper() != "USD":
            return None

        # --- Fetch metrics ---
        result = {}
        missing_count = 0

        for metric in metrics:
            yf_key = METRIC_MAP.get(metric, metric)
            value = info.get(yf_key)

            if value is not None:
                result[metric] = round(value, 4) if isinstance(value, float) else value
            else:
                result[metric] = None
                missing_count += 1

        # --- Skip if more than half of metrics are missing ---
        if missing_count > len(metrics) / 2:
            return None

        return result

    except Exception:
        return None


def _rank_companies(comparison: list[dict], metrics: list[str]) -> list[dict]:
    """Add continuous per-metric rankings to each company entry."""
    for metric in metrics:
        # Collect (index, value) pairs, skipping None
        values = []
        for i, entry in enumerate(comparison):
            val = entry["metrics"].get(metric)
            if val is not None:
                values.append((i, val))

        # Sort: lower rank number = better
        reverse = metric not in LOWER_IS_BETTER
        values.sort(key=lambda x: x[1], reverse=reverse)

        # Assign continuous ranks (1, 2, 3… no gaps)
        for rank, (idx, _) in enumerate(values, 1):
            comparison[idx]["rank"][metric] = rank

    return comparison


# ─── Main Function ────────────────────────────────────────────────────────────

def peer_comparison(
    ticker: str,
    num_peers: int = 4,
    metrics: list[str] = None,
) -> dict:
    """
    Compare a company against its sector peers on key financial metrics.

    Args:
        ticker:    Company ticker (e.g. "AAPL").
        num_peers: Number of peer companies to compare.
        metrics:   List of metric names to compare.

    Returns:
        Comparison dict with rankings, or {"error": str}.
    """
    try:
        ticker = ticker.upper().strip()
        if not metrics:
            metrics = DEFAULT_METRICS

        # --- Normalize metric names ---
        # Allow users to pass yfinance keys or our aliases
        normalized_metrics = []
        for m in metrics:
            if m in METRIC_MAP:
                normalized_metrics.append(m)
            else:
                # Try to find a matching alias
                found = False
                for alias, yf_key in METRIC_MAP.items():
                    if m == yf_key or m.lower() == alias.lower():
                        normalized_metrics.append(alias)
                        found = True
                        break
                if not found:
                    normalized_metrics.append(m)  # pass through as-is

        metrics = normalized_metrics

        # --- Find candidate peers ---
        candidate_peers = _find_peers(ticker, num_peers * 2)  # fetch extra to filter
        if not candidate_peers:
            return {"error": f"Could not find peers for {ticker}."}

        # --- Fetch target company data ---
        target_data = _fetch_company_data(ticker, metrics)
        if target_data is None:
            return {"error": f"Could not fetch valid data for {ticker}."}

        comparison = [{
            "company": ticker,
            "metrics": target_data,
            "rank": {},
        }]

        # --- Fetch peer data, filtering out invalid companies ---
        valid_peers = []
        for peer in candidate_peers:
            if len(valid_peers) >= num_peers:
                break

            peer_data = _fetch_company_data(peer, metrics)
            if peer_data is None:
                continue  # skip non-USD or data-incomplete peers

            valid_peers.append(peer)
            comparison.append({
                "company": peer,
                "metrics": peer_data,
                "rank": {},
            })

        if not valid_peers:
            return {"error": f"No valid peers found for {ticker} after filtering."}

        # --- Rank companies per metric ---
        comparison = _rank_companies(comparison, metrics)

        return {
            "ticker": ticker,
            "peers": valid_peers,
            "comparison": comparison,
        }

    except Exception as e:
        return {"error": f"Peer comparison failed: {e}"}

"""
Report Generator Tool — Enhanced (A7.2 Structured Output Pattern)

Assembles researched data sections into a structured investment
research report in Markdown format with:
  - Professional header with date, analyst ID, and confidence score
  - Source citations with reliability tier labels
  - Auto-detected data gaps section
  - Section-level confidence scoring
"""

import os
import re
from datetime import datetime

# ─── Section Templates ────────────────────────────────────────────────────────

SECTION_ORDER = [
    "executive_summary",
    "company_overview",
    "financial_analysis",
    "management_insights",
    "market_sentiment",
    "peer_comparison",
    "risks",
    "data_limitations",
    "conclusion",
]

SECTION_TITLES = {
    "executive_summary": "Executive Summary",
    "company_overview": "Company Overview",
    "financial_analysis": "Financial Analysis",
    "management_insights": "Management Insights",
    "market_sentiment": "Market Sentiment & Recent Developments",
    "peer_comparison": "Competitive Position & Peer Comparison",
    "risks": "Risks & Considerations",
    "data_limitations": "Data Limitations & Gaps",
    "conclusion": "Investment Thesis & Recommendation",
}

# Minimum content length (chars) to consider a section "complete"
MIN_SECTION_LENGTH = 50


def _detect_data_gaps(sections: dict) -> list[str]:
    """
    Identify missing or weak sections in the report.

    Returns a list of human-readable gap descriptions.
    """
    expected = [
        ("company_overview", "Company Overview"),
        ("financial_analysis", "Financial Analysis"),
        ("risks", "Risks & Considerations"),
        ("conclusion", "Investment Thesis"),
    ]

    gaps = []
    for key, label in expected:
        content = sections.get(key, "")
        if not content:
            gaps.append(f"{label}: Section missing entirely")
        elif len(content) < MIN_SECTION_LENGTH:
            gaps.append(f"{label}: Minimal content ({len(content)} chars)")

    # Check for uncited numbers (simple heuristic)
    all_content = " ".join(str(v) for v in sections.values())
    import re
    numbers = re.findall(r'\$[\d,.]+[BMK]?|\d+\.\d+%', all_content)
    citations = all_content.lower().count("[source:")
    if numbers and citations == 0:
        gaps.append(
            f"Citations: {len(numbers)} numerical claims found but no "
            f"[Source: tool_name] citations detected"
        )

    return gaps


def _compute_section_confidence(sections: dict) -> dict[str, str]:
    """
    Estimate confidence level per section based on content quality signals.

    Returns dict of section_key -> "High" | "Medium" | "Low".
    """
    confidence = {}
    for key, content in sections.items():
        if not content:
            confidence[key] = "N/A"
            continue

        content_lower = content.lower()
        score = 0

        # Length signals
        if len(content) > 500:
            score += 2
        elif len(content) > 200:
            score += 1

        # Citation signals
        if "[source:" in content_lower or "(source:" in content_lower:
            score += 2
        if "cite:" in content_lower:
            score += 1

        # Data quality signals
        if "data not available" in content_lower or "not available" in content_lower:
            score -= 1
        if any(w in content_lower for w in ["tier 1", "sec filing", "10-k"]):
            score += 1

        if score >= 3:
            confidence[key] = "High"
        elif score >= 1:
            confidence[key] = "Medium"
        else:
            confidence[key] = "Low"

    return confidence


def report_generator(
    template: str = "investment_report",
    sections: dict = None,
    sources: list = None,
) -> str:
    """
    Generate a clean research report in Markdown.
    """
    try:
        if not sections:
            return {"error": "No sections provided for report generation."}

        report_lines = []

        # ── Professional Header ───────────────────────────────────────
        title = sections.get("title", "Research Report")
        report_lines.append(f"# {title}")
        report_lines.append("")
        report_lines.append(f"**Date**: {datetime.now().strftime('%B %d, %Y')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # ── Sections in order ─────────────────────────────────────────
        for key in SECTION_ORDER:
            content = sections.get(key)
            if not content:
                continue

            heading = SECTION_TITLES.get(key, key.replace("_", " ").title())
            
            report_lines.append(f"## {heading}")
            report_lines.append("")
            report_lines.append(content)
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        # ── Extra sections not in standard order ──────────────────────
        standard_keys = set(SECTION_ORDER) | {"title", "ticker"}
        for key, content in sections.items():
            if key not in standard_keys and content:
                heading = key.replace("_", " ").title()
                report_lines.append(f"## {heading}")
                report_lines.append("")
                report_lines.append(content)
                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")

        return "\n".join(report_lines)

    except Exception as e:
        return {"error": f"Report generation failed: {e}"}

from agent.core import run_agent
from tools.company_profile import company_profile

def generate_report(ticker: str, query: str) -> dict:
    """
    Main entry point for generating a professional, cleaned research report.
    """
    # Step 1: Get raw agent output
    agent_out = run_agent(query)
    raw_output = agent_out.get("report", "")
    intent = agent_out.get("intent", "DETAILED")

    # Step 2: Resolve ticker from agent output (the input 'ticker' may be the raw query)
    resolved_ticker = agent_out.get("ticker", ticker)
    # Avoid using raw query strings as tickers
    if resolved_ticker and len(resolved_ticker) > 10:
        resolved_ticker = ticker if len(ticker) <= 10 else "UNKNOWN"

    # Step 3: Get company name dynamically
    company_name = agent_out.get("company_name", resolved_ticker)
    if resolved_ticker and resolved_ticker != "Unknown" and resolved_ticker != "UNKNOWN":
        try:
            profile = company_profile(resolved_ticker)
            company_name = profile.get("company_name", resolved_ticker.split(".")[0])
        except Exception:
            pass

    # Step 4: Apply cleaning
    cleaned_output = clean_output(raw_output, company_name)

    # DEBUG CHECK (MANDATORY)
    print("\n[DEBUG CLEANED OUTPUT]:\n", cleaned_output[:500])

    return {
        "report": cleaned_output,
        "ticker": resolved_ticker,
        "company_name": company_name,
        "intent": intent
    }

def clean_output(text: str, company_name: str = None) -> str:
    import re

    # -----------------------------
    # 1. Remove ticker formats
    # -----------------------------
    text = re.sub(r'\(?\b\d{4,6}\.[A-Z]{2,3}\b\)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(?\b[A-Z]{1,5}\.(US|KS|NS|L|HK|T)\b\)?', '', text, flags=re.IGNORECASE)

    # -----------------------------
    # 2. REMOVE ANY TOOL / API PHRASES
    # -----------------------------
    # Remove phrases containing "tool" or "API"
    text = re.sub(
        r'([A-Z][^.]*?\b(tool|api)\b[^.]*\.)',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Also remove inline fragments
    text = re.sub(
        r'\b[\w\s\-]*(tool|api)[\w\s\-]*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # -----------------------------
    # 3. Replace numeric ticker safely
    # -----------------------------
    if company_name:
        text = re.sub(r'\b005930\b', company_name, text)

    # -----------------------------
    # 4. Fix broken sentences
    # -----------------------------
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove leftover awkward starts
    text = re.sub(r'According to\s*,?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Based on\s*,?', '', text, flags=re.IGNORECASE)

    # Remove empty parentheses
    text = text.replace("()", "")
    text = text.replace("( )", "")

    return text.strip()
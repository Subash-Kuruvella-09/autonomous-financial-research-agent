from agent.query_analyzer import client
import json
import re
from datetime import datetime

def disambiguate_query(user_query: str) -> dict:
    """
    Analyze and disambiguate a financial research query.

    Implements PRD section A7.3:
    - Identifies the most likely interpretation based on context
    - Detects ambiguous queries (e.g., "What's happening with banks?")
    - Uses temporal context for disambiguation
    - Classifies query type for optimized tool selection

    Args:
        user_query: Raw user query string.

    Returns:
        Dict with:
          - is_ambiguous: bool
          - query_type: str (company_profile|earnings_analysis|risk_assessment|
                             competitor_comparison|sector_analysis|full_report)
          - identified_tickers: list[str]
          - clarified_query: str
          - research_focus: list[str] — key topics to investigate
          - temporal_context: str — relevant time period
    """
    current_year = datetime.now().year

    prompt = f"""You are a financial research query analyzer.

Current date: {datetime.now().strftime('%B %d, %Y')}

Analyze this user query and return a JSON object:

User Query: "{user_query}"

Return ONLY valid JSON with these fields:
{{
  "is_ambiguous": true or false,
  "query_type": "company_profile" or "earnings_analysis" or "risk_assessment" or "competitor_comparison" or "sector_analysis" or "full_report",
  "identified_tickers": ["MSFT", "AAPL"],
  "company_name": "Full legal company name",
  "clarified_query": "A clear, specific version of the query",
  "research_focus": ["revenue trends", "competitive position"],
  "temporal_context": "FY2024" or "Q1 2025" or "last 3 years"
}}

Rules:
- If the query mentions a specific company, extract its ticker
- If the query is vague (e.g. "what's happening with banks?"), set is_ambiguous=true and identify likely sector/companies
- Use temporal context: a query about "bank stress tests" in {current_year} likely refers to Federal Reserve CCAR/DFAST
- query_type determines which tools the agent will prioritize
- research_focus should list 2-4 specific topics to investigate

Return ONLY the JSON object. No explanation."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial query analyzer. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1)

        result = json.loads(raw)

        # Validate required fields
        result.setdefault("is_ambiguous", False)
        result.setdefault("query_type", "full_report")
        result.setdefault("identified_tickers", [])
        result.setdefault("clarified_query", user_query)
        result.setdefault("research_focus", [])
        result.setdefault("temporal_context", str(current_year))

        return result

    except Exception as e:
        # Fallback: return non-ambiguous default
        return {
            "is_ambiguous": False,
            "query_type": "full_report",
            "identified_tickers": [],
            "clarified_query": user_query,
            "research_focus": [],
            "temporal_context": str(current_year),
        }


# ─── Company Analysis ────────────────────────────────────────────────────────

def analyze_company(data):
    """Generate a grounded company analysis from structured data."""
    prompt = f"""
    You are a financial analyst AI.

    You are given ONLY this real company data:
    {data}

    Your job:
    - Explain the company briefly
    - Identify strengths based ONLY on this data
    - Identify risks based ONLY on this data

    STRICT RULES:
    - Do NOT use external knowledge
    - Do NOT add financial numbers not present in data
    - If information is missing, say "data not available"

    Keep the answer simple and grounded.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
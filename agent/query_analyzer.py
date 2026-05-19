"""
LLM Services — Groq client and query processing utilities.

Provides:
  - Groq client initialization
  - Ticker extraction from natural language
  - Query disambiguation with temporal context (A7.3)
  - Company analysis prompting
"""

import json
import os
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

# Load env
load_dotenv()

# Initialize client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─── Ticker Extraction ───────────────────────────────────────────────────────

def get_ticker_from_llm(company_name: str) -> str:
    """Extract stock ticker symbol from a company name or query."""
    prompt = f"""
    What is the stock ticker symbol for the company: {company_name}?

    Only return the ticker symbol.
    No explanation.

    Examples:
    Tesla → TSLA
    Apple → AAPL
    Microsoft → MSFT
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# ─── Query Disambiguation (A7.3) ─────────────────────────────────────────────


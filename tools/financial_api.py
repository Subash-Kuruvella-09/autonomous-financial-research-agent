"""
Financial Data API Tool

Retrieves structured financial data (income statement, balance sheet,
cash flow, key ratios) for a given ticker using Yahoo Finance.
"""

import yfinance as yf


# ─── Configuration ────────────────────────────────────────────────────────────

VALID_STATEMENTS = {"income", "balance", "cashflow", "ratios"}
VALID_PERIODS = {"annual", "quarterly"}


# ─── Field Extraction ─────────────────────────────────────────────────────────

# Important fields to keep from each statement type
INCOME_FIELDS = [
    "Total Revenue", "Cost Of Revenue", "Gross Profit",
    "Operating Income", "Operating Expense",
    "Net Income", "EBITDA", "EBIT",
    "Basic EPS", "Diluted EPS",
    "Total Expenses", "Interest Expense",
    "Tax Provision",
]

BALANCE_FIELDS = [
    "Total Assets", "Current Assets",
    "Total Liabilities Net Minority Interest", "Current Liabilities",
    "Stockholders Equity",
    "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
    "Total Debt", "Net Debt", "Long Term Debt",
    "Goodwill And Other Intangible Assets",
    "Total Capitalization",
]

CASHFLOW_FIELDS = [
    "Operating Cash Flow", "Free Cash Flow",
    "Capital Expenditure",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Repurchase Of Capital Stock",
    "Cash Dividends Paid",
    "Change In Cash Supplemental As Reported",
    "Issuance Of Debt", "Repayment Of Debt",
]


def _dataframe_to_records(df, wanted_fields: list[str], num_periods: int) -> list[dict]:
    """Convert a yfinance DataFrame to a list of clean dicts with selected fields."""
    if df is None or df.empty:
        return []

    records = []
    # yfinance returns columns as dates, rows as metrics
    columns = list(df.columns)[:num_periods]

    for col in columns:
        record = {"date": str(col.date()) if hasattr(col, "date") else str(col)}
        for field in wanted_fields:
            if field in df.index:
                value = df.at[field, col]
                # Convert numpy types to native Python
                if hasattr(value, "item"):
                    value = value.item()
                record[field] = value
        records.append(record)

    return records


def _compute_ratios(stock: yf.Ticker, period: str, years: int) -> list[dict]:
    """Compute key financial ratios from available data."""
    info = stock.info or {}

    # Build a single-period ratios snapshot from info (yfinance doesn't
    # provide historical ratios as a DataFrame)
    ratios = {
        "date": "latest",
        # Valuation
        "PE Ratio": info.get("trailingPE"),
        "Forward PE": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Price to Sales": info.get("priceToSalesTrailing12Months"),
        "Price to Book": info.get("priceToBook"),
        "Enterprise Value / EBITDA": info.get("enterpriseToEbitda"),
        "Enterprise Value / Revenue": info.get("enterpriseToRevenue"),
        # Profitability
        "Profit Margin": info.get("profitMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "Gross Margin": info.get("grossMargins"),
        "Return on Equity": info.get("returnOnEquity"),
        "Return on Assets": info.get("returnOnAssets"),
        # Dividend
        "Dividend Yield": info.get("dividendYield"),
        "Payout Ratio": info.get("payoutRatio"),
        # Debt
        "Debt to Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        # Per Share
        "Book Value per Share": info.get("bookValue"),
        "Revenue per Share": info.get("revenuePerShare"),
        "Free Cash Flow per Share": None,
        # Market
        "Market Cap": info.get("marketCap"),
        "Enterprise Value": info.get("enterpriseValue"),
        "Beta": info.get("beta"),
    }

    # Compute FCF per share if possible
    try:
        fcf = info.get("freeCashflow")
        shares = info.get("sharesOutstanding")
        if fcf and shares:
            ratios["Free Cash Flow per Share"] = round(fcf / shares, 2)
    except Exception:
        pass

    # Remove None values for cleanliness
    ratios = {k: v for k, v in ratios.items() if v is not None}

    return [ratios]


# ─── Main Function ────────────────────────────────────────────────────────────

def financial_data_api(
    ticker: str,
    statement_type: str = "income",
    period: str = "annual",
    years: int = 3,
) -> dict:
    """
    Retrieve structured financial data for a company.

    Args:
        ticker:         Company ticker symbol (e.g. "AAPL").
        statement_type: One of "income", "balance", "cashflow", "ratios".
        period:         "annual" or "quarterly".
        years:          Number of periods to return.

    Returns:
        A dict with the financial data, or {"error": "..."} on failure.
    """
    try:
        # --- Input validation ---
        ticker = ticker.upper().strip()
        statement_type = statement_type.lower().strip()
        period = period.lower().strip()

        if statement_type not in VALID_STATEMENTS:
            return {
                "error": (
                    f"Invalid statement_type '{statement_type}'. "
                    f"Must be one of: {', '.join(sorted(VALID_STATEMENTS))}"
                )
            }

        if period not in VALID_PERIODS:
            return {
                "error": (
                    f"Invalid period '{period}'. "
                    f"Must be one of: {', '.join(sorted(VALID_PERIODS))}"
                )
            }

        # --- Fetch data from Yahoo Finance ---
        stock = yf.Ticker(ticker)

        # For quarterly, request more periods since yfinance returns last 4
        num_periods = years if period == "annual" else years * 4

        if statement_type == "income":
            df = stock.quarterly_financials if period == "quarterly" else stock.financials
            data = _dataframe_to_records(df, INCOME_FIELDS, num_periods)

        elif statement_type == "balance":
            df = stock.quarterly_balance_sheet if period == "quarterly" else stock.balance_sheet
            data = _dataframe_to_records(df, BALANCE_FIELDS, num_periods)

        elif statement_type == "cashflow":
            df = stock.quarterly_cashflow if period == "quarterly" else stock.cashflow
            data = _dataframe_to_records(df, CASHFLOW_FIELDS, num_periods)

        elif statement_type == "ratios":
            data = _compute_ratios(stock, period, years)

        if not data:
            return {"error": f"No {statement_type} data found for {ticker}."}

        return {
            "ticker": ticker,
            "statement_type": statement_type,
            "period": period,
            "data": data,
        }

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

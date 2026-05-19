import yfinance as yf

def company_profile(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return {"error": f"No profile found for {ticker}"}

        ceo = "N/A"
        company_officers = info.get("companyOfficers", [])
        for officer in company_officers:
            title = officer.get("title", "").lower()
            if "ceo" in title or "chief executive" in title:
                ceo = officer.get("name", "N/A")
                break

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0.0),
            "ceo": ceo,
            "description": info.get("longBusinessSummary", "N/A")
        }
    except Exception as e:
        return {"error": str(e)}

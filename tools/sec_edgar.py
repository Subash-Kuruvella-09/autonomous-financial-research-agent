import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Subash Agent (your_email@example.com)"
}

def sec_filing_search(ticker: str, filing_type: str, year: int = None):
    try:
        # ---------------------------
        # Step 1: Get CIK
        # ---------------------------
        url = "https://www.sec.gov/files/company_tickers.json"
        res = requests.get(url, headers=HEADERS)

        if res.status_code != 200:
            return {"error": "Failed to fetch company list"}

        data = res.json()

        cik = None
        for company in data.values():
            if company["ticker"].lower() == ticker.lower():
                cik = str(company["cik_str"]).zfill(10)
                break

        if not cik:
            return {"error": "CIK not found"}

        # ---------------------------
        # Step 2: Get filings
        # ---------------------------
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(filings_url, headers=HEADERS)

        if res.status_code != 200:
            return {"error": "Failed to fetch filings"}

        filings_data = res.json()
        recent = filings_data["filings"]["recent"]

        # ---------------------------
        # Step 3: Find filing
        # ---------------------------
        for i in range(min(len(recent["form"]), 100)):
            form = recent["form"][i]
            date = recent["filingDate"][i]

            if filing_type.lower() in form.lower():

                if year and not date.startswith(str(year)):
                    continue

                accession = recent["accessionNumber"][i]
                primary_doc = recent["primaryDocument"][i]

                accession_clean = accession.replace("-", "")
                cik_int = str(int(cik))

                # Fix document name
                if primary_doc.endswith(".xml"):
                    primary_doc = primary_doc.replace(".xml", ".htm")

                if "ix?doc=" in primary_doc:
                    primary_doc = primary_doc.split("doc=")[-1]

                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_doc}"

                # ---------------------------
                # Step 4: Fetch document
                # ---------------------------
                doc_res = requests.get(doc_url, headers=HEADERS)
                if doc_res.status_code != 200:
                    continue

                # ---------------------------
                # Step 5: Parse HTML
                # ---------------------------
                soup = BeautifulSoup(doc_res.text, "html.parser")

                for tag in soup(["script", "style"]):
                    tag.decompose()

                text = soup.get_text(separator="\n")

                # ---------------------------
                # Step 6: Clean text
                # ---------------------------
                clean_lines = []

                for line in text.split("\n"):
                    line = line.strip()

                    if not line:
                        continue

                    # remove XBRL junk
                    if any(x in line.lower() for x in [
                        "us-gaap", "xbrli", "iso4217",
                        "linkbase", "schema", "contextref", "unitref"
                    ]):
                        continue

                    # remove technical tags
                    if ":" in line and len(line.split()) < 6:
                        continue

                    # remove junk lines
                    if sum(c.isalpha() for c in line) < 20:
                        continue

                    if len(line) < 60:
                        continue

                    clean_lines.append(line)

                # ---------------------------
                # Step 7: IMPORTANT SECTION FILTER 🔥
                # ---------------------------
                important_sections = []

                for line in clean_lines:
                    if any(keyword in line.lower() for keyword in [
                        "business", "competition", "risk", "market",
                        "products", "services", "supply", "economic"
                    ]):
                        important_sections.append(line)

                final_text = "\n".join(important_sections[:50])

                # fallback if nothing found
                if not final_text:
                    final_text = "\n".join(clean_lines[:50])

                # ---------------------------
                # Step 8: Return
                # ---------------------------
                return {
                    "ticker": ticker.upper(),
                    "filing_type": form,
                    "filing_date": date,
                    "accession_number": accession,
                    "filing_url": doc_url,
                    "filing_summary": summarize_filing(final_text),
                    "raw_text": final_text
                }

        return {"error": "Filing not found"}

    except Exception as e:
        return {"error": str(e)}


def summarize_filing(text: str):
    lines = text.split("\n")

    business = []
    products_services = []
    competition = []   # 🔥 THIS WAS MISSING
    risks = []

    for line in lines:
        l = line.lower()

        if "designs" in l or "manufactures" in l:
            business.append(line)

        elif "product" in l or "service" in l:
            products_services.append(line)

        elif any(word in l for word in [
            "competition", "competitive", "competitors"
        ]):
            competition.append(line)

        elif "risk" in l or "adverse" in l or "uncertainty" in l:
            risks.append(line)

    # remove duplicates + limit
    competition = list(dict.fromkeys(competition))[:5]

    return {
        "business": "\n".join(business[:3]),
        "products_services": "\n".join(products_services[:3]),
        "competition": "\n".join(competition),
        "risks": "\n".join(risks[:5])
    }
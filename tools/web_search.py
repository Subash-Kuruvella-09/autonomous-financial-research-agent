from ddgs import DDGS

def web_search(query: str, num_results: int = 10, date_range: str = None):
    try:
        results = []
        seen_urls = set()

        search_query = f"{query} stock news"

        BAD_DOMAINS = [
            "wikipedia",
            "grokipedia",
            "apple.com/newsroom"
        ]

        with DDGS() as ddgs:
            search_results = ddgs.text(search_query, max_results=30)

            for r in search_results:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")

                if not title or not url:
                    continue

                url_lower = url.lower()

                # skip bad domains
                if any(bad in url_lower for bad in BAD_DOMAINS):
                    continue

                # remove duplicates
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                results.append({
                    "title": title.strip(),
                    "url": url.strip(),
                    "snippet": snippet.strip() if snippet else ""
                })

                if len(results) >= num_results:
                    break

        if not results:
            return {"error": "No results found"}

        return results

    except Exception as e:
        return {"error": str(e)}
from duckduckgo_search import DDGS


def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return ""
        parts = []
        for r in results[:3]:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            parts.append(f"Title: {title}\nURL: {href}\nSummary: {body}")
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""

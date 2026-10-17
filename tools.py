import hashlib
from pathlib import Path
from ddgs import DDGS
from langchain_core.tools import tool

CACHE_DIR = Path(".tool_cache")
CACHE_DIR.mkdir(exist_ok=True)

@tool
def web_serach(query: str) -> str:
    """Search DuckDuckGo for real-time information, recent news, or factual facts.

    Args:
        query: Clear, specific search terms to look up.
    """
    query_clean = query.lower().strip()
    query_hash = hashlib.md5(query_clean.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{query_hash}.json"

    if cache_file.exists():
        print(f"[CACHE HIT] Returning cached result for: '{query_clean}'")
        return cache_file.read_text(encoding="utf-8")

    print(f"[CACHE MISS] Executing real search for: '{query_clean}'")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query_clean, max_results=3))
    except Exception as e:
        error_msg = f"Error: Web search failed due to exception: {str(e)}"
        print(f"[SEARCH ERROR] {error_msg}")
        return error_msg  

    if not results:
        print(f"[SEARCH BLANK] No results returned for '{query_clean}'. Not caching.")
        return "No search results found."

    formatted_results = "\n\n".join(
        [f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}\nLink: {r.get('href', '')}" for r in results]
    )

    cache_file.write_text(formatted_results, encoding="utf-8")
    return formatted_results